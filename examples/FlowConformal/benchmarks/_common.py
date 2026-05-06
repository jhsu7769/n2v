"""Shared pipeline runner for PoC benchmarks with curved output sets.

Unlike golden-path (identity / rotated linear), these benchmarks have no
analytical exact reach-set volume. The ground-truth 1-alpha probabilistic
reachset volume is estimated from the Star-union pushforward:

    vol_exact(1-alpha) ~= (1 - alpha) * vol(Star_union)

because P_X is uniform on B(x_0, eps) and the Star union is the exact
deterministic pushforward of that input box. The smallest 1-alpha reachset
under a uniform distribution on a set is the set itself times (1-alpha)
mass — this is the tightness floor any conformal score is measured against.

Compares hyperrect / ball / flow scores against the Star-union reference.

Backward-compat shim
====================

This module also acts as a thin backward-compatibility shim for the
flow-conformal+AMLS verification pipeline, which now lives in
``n2v.probabilistic.verify_flow``. The shim re-exports the moved
internals and overrides ``run_verification_pipeline`` to default
``use_falsifier=True`` so existing example scripts (acasxu_sweep.py,
phase5c_probe_sweep.py, etc.) keep producing the same results.

For new code, prefer the library API:

    from n2v.probabilistic import verify_flow
    result = verify_flow(network=..., ..., use_falsifier=False)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.utils import compute_exact_reach
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore  # noqa: F401 (available to callers)
from n2v.probabilistic.flow.sampling import sample_l_inf_ball
from n2v.probabilistic.flow.scores import BallScore, FlowScore, HyperrectScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.volume import (
    compute_mc_bbox, exact_volume_2d, star_union_volume_mc,
)
from n2v.sets.halfspace import HalfSpace
# Re-export the moved verify_flow internals so legacy callers (ablation
# scripts, tests) that imported these as ``examples.FlowConformal.benchmarks._common.<name>``
# continue to work. The implementations now live in n2v.
from n2v.probabilistic.verify_flow import (  # noqa: F401
    _MaxEnsembleFlowScore,
    _WhitenedNetwork,
    _WhiteningFlowScore,
    _certify_spec_on_flow_v2,
    _extract_min_worst_max_margin,
    _flow_unsat_pipeline,
    _forward,
    _sat_result,
    _train_flow,
    _train_flow_tight,
    _whiten_halfspace,
    run_verification_pipeline as _verify_flow_pipeline,
)


@dataclass
class MethodResult:
    name: str
    threshold: float
    volume: float
    volume_se: float
    empirical_coverage: float
    fit_time_s: float


def exact_star_union_volume(net, x_center: np.ndarray, radius: float,
                            output_dim: int, n_mc: int = 500_000,
                            seed: int = 42) -> tuple[float, list]:
    """Star-union ground-truth volume (an exact deterministic over-approx of
    f_#P_X's support; the 1-alpha reachset is smaller by (1-alpha)).

    Returns (volume_mean, stars). The MC estimate is used because the Star
    union can have thousands of overlapping polytopes whose exact volume
    requires inclusion-exclusion.
    """
    reach = compute_exact_reach(net, x_center, radius, output_dim=output_dim)
    stars = reach['stars']
    if output_dim == 2:
        # 2D has a cheap rasterization method, which we use as the ground-
        # truth reference rather than MC on a box (the 2D Star union is a
        # measure-zero manifold in some cases, so MC-on-a-box would give 0).
        y_bbox = compute_mc_bbox(net, x_center, radius, output_dim=output_dim,
                                 n_samples=5000, pad=1.0)
        vol = exact_volume_2d(stars, (y_bbox[0].numpy(), y_bbox[1].numpy()),
                              resolution=500)
        return float(vol), stars
    ve = star_union_volume_mc(
        stars, n_samples=n_mc, batch_size=25_000, seed=seed,
        contains_method='algebraic',
    )
    return float(ve.mean), stars


def run_pipeline(
    net,
    x_center: np.ndarray,
    radius: float,
    output_dim: int,
    star_union_volume: float,
    alpha: float = 0.01,
    n_train: int = 10_000,
    n_calib: int = 2_000,
    n_test: int = 2_000,
    seed: int = 0,
    flow_epochs: int = 2000,
    n_mc_volume: int = 400_000,
    flow_config: str = 'base',
    infer_solver: str = 'rk4',
    infer_atol: float = 1e-5,
    infer_rtol: float = 1e-5,
    infer_steps: int = 30,
    flow_score_class=FlowScore,
    flow_score_infer_batch_size: int = 65536,
) -> dict:
    """Run the flow-conformal pipeline.

    flow_config:
      'base'  — hidden=128, L=4, concat time, uniform time (original).
      'tight' — hidden=256, L=6, sinusoidal time, logit-normal time
                (experiment (c)+(e)).
    infer_solver / atol / rtol / steps:
      Control the ODE solver used when scoring y_calib, y_test and MC
      samples. 'rk4' is a fast fixed-step solver (30 steps is plenty if the
      flow has converged). 'dopri5' with atol/rtol ~ 1e-4 is 2-3x slower but
      more accurate for a poorly-converged flow (experiment (d)).
    flow_score_class: callable producing a NonconformityScore around the
        trained flow. Defaults to the naive ``FlowScore``. Pass
        ``LogDetFlowScore`` for the log-density-corrected variant.
        The constructor is called with the same kwargs we currently pass
        to FlowScore (t, n_steps, method, atol, rtol, batch_size).
    flow_score_infer_batch_size: chunk size for score evaluations on MC
        points. 65536 is the current production default; reduce if GPU
        memory is tight.
    """
    ell = int(math.ceil((n_calib + 1) * (1 - alpha)))
    torch.manual_seed(seed)
    dim_in = x_center.shape[0]
    x_center_t = torch.as_tensor(x_center, dtype=torch.float32)

    x_tr = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_train, seed=seed, dim=dim_in,
    )
    x_ca = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_calib,
        seed=seed + 1_000_000, dim=dim_in,
    )
    x_te = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_test,
        seed=seed + 2_000_000, dim=dim_in,
    )
    y_tr = _forward(net, x_tr)
    y_ca = _forward(net, x_ca)
    y_te = _forward(net, x_te)

    y_all = torch.cat([y_tr, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)

    results: list[MethodResult] = []
    for name, builder in (
        ('hyperrect', lambda: HyperrectScore(
            center=y_ca.mean(dim=0),
            scales=y_ca.std(dim=0).clamp(min=1e-8),
        )),
        ('ball', lambda: BallScore(center=y_ca.mean(dim=0))),
    ):
        t0 = time.time()
        score_fn = builder()
        thresh = calibrate(score_fn(y_ca), ell).item()
        s = ProbabilisticSet(
            score_fn=score_fn, threshold=thresh,
            m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
        )
        vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
        cov = s.contains(y_te).float().mean().item()
        results.append(MethodResult(
            name=name, threshold=thresh, volume=vol, volume_se=se,
            empirical_coverage=cov, fit_time_s=time.time() - t0,
        ))

    # Flow
    t0 = time.time()
    if flow_config == 'base':
        flow = _train_flow(y_tr, output_dim, flow_epochs, seed)
    elif flow_config == 'tight':
        flow = _train_flow_tight(y_tr, output_dim, flow_epochs, seed)
    else:
        raise ValueError(f"unknown flow_config {flow_config!r}")
    train_time = time.time() - t0

    score_fn = flow_score_class(
        flow, t=1.0, n_steps=infer_steps, method=infer_solver,
        batch_size=flow_score_infer_batch_size,
        atol=infer_atol, rtol=infer_rtol,
    )
    t1 = time.time()
    thresh = calibrate(score_fn(y_ca), ell).item()
    s = ProbabilisticSet(
        score_fn=score_fn, threshold=thresh,
        m=n_calib, ell=ell, epsilon=alpha, dim=output_dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
    cov = s.contains(y_te).float().mean().item()
    infer_time = time.time() - t1
    results.append(MethodResult(
        name='flow', threshold=thresh, volume=vol, volume_se=se,
        empirical_coverage=cov, fit_time_s=train_time + infer_time,
    ))

    return {
        'results': results,
        'bbox': bbox,
        'y_train': y_tr, 'y_calib': y_ca, 'y_test': y_te,
        'star_union_volume': star_union_volume,
        'alpha': alpha,
        'flow_train_time_s': train_time,
        'flow_infer_time_s': infer_time,
    }


def print_report(bundle: dict):
    results = bundle['results']
    su = bundle['star_union_volume']
    alpha = bundle['alpha']
    floor = (1 - alpha) * su  # tightness floor for any 1-alpha reachset
    print(f"\n  Star-union volume      = {su:.4f}")
    print(f"  (1-alpha)*Star-union   = {floor:.4f}  <- tightness floor")
    print(f"  alpha = {alpha}  coverage floor = {1 - alpha}")
    print(f"  {'method':<10} {'vol':>10} {'+/-SE':>10} {'vol/floor':>10} "
          f"{'cov':>8} {'fit(s)':>8}")
    for r in results:
        ratio = r.volume / floor if floor > 0 else float('nan')
        print(f"  {r.name:<10} {r.volume:>10.4f} {r.volume_se:>10.4f} "
              f"{ratio:>10.3f} {r.empirical_coverage:>8.4f} {r.fit_time_s:>8.1f}")
    print(f"  (flow: train {bundle['flow_train_time_s']:.1f}s, "
          f"infer {bundle['flow_infer_time_s']:.1f}s)")


# --- Verification pipeline shim ---------------------------------------
#
# The flow-conformal+AMLS verification pipeline implementation now lives
# in :mod:`n2v.probabilistic.verify_flow`. The function below is a thin
# backward-compat wrapper that defaults ``use_falsifier=True`` so existing
# example scripts (acasxu_sweep.py, phase5c_probe_sweep.py, etc.) keep
# producing the same falsifier-on-by-default behavior they were built
# against. New code should prefer the n2v library API directly.


def run_verification_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    use_falsifier: bool = True,  # legacy default (Stage-1 falsifier ON)
    **kwargs,
) -> dict:
    """Backward-compat shim around :func:`n2v.probabilistic.verify_flow`.

    Defaults ``use_falsifier=True`` to preserve the falsifier-on-by-
    default behavior expected by existing example scripts. The library
    API at ``n2v.probabilistic.verify_flow.run_verification_pipeline``
    defaults ``use_falsifier=False``.

    For new code prefer:

        from n2v.probabilistic import verify_flow
        result = verify_flow(network=..., ..., use_falsifier=False)
    """
    return _verify_flow_pipeline(
        network=network,
        input_lb=input_lb,
        input_ub=input_ub,
        spec=spec,
        use_falsifier=use_falsifier,
        **kwargs,
    )

