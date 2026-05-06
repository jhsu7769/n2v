"""Exp 3 (synthetic validation): 3D banana benchmark — REACHABLE-SAT spec.

Runs the AMLS-locked Phase 5d pipeline on ``ThreeBlobClassifier3D`` with
a halfspace spec that genuinely intersects the reach set
(``y[0] >= 0.5``, encoded as ``-y[0] <= -0.5``). Phase 2 documented this
spec as a known SAT case for the analogous ``RotatedBananaNet`` 2D net;
on the 3D classifier the empirical y[0] range easily exceeds 0.5
(see :func:`run_one_seed`). The falsifier is OFF per Exp 3 plan, so
the verifier should return ``UNKNOWN`` (AMLS detects unsafe samples →
cannot certify safe). Validates honest abstention.

Output CSV columns:
    seed, verdict, q, volume_estimate, volume_ratio_vs_exact,
    coverage_empirical, train_s, verify_s, total_s

Smoke usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_3d_banana_sat \\
        --smoke

Full usage (5 seeds; ~5 min/seed):
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_3d_banana_sat
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.networks import ThreeBlobClassifier3D
from examples.FlowConformal.benchmarks._common import (
    exact_star_union_volume, run_verification_pipeline,
)
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.halfspace import HalfSpace


_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'exp3_3d_banana_sat_ours.csv'

# >>> LOCKED Phase 5d config <<<
_VERIFICATION_METHOD = 'amls'
_ALPHA = 0.001
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_SCENARIO_BETA = 0.001

_X_CENTER = np.zeros(3)
_RADIUS = 1.0
_OUTPUT_DIM = 3


def _build_sat_spec() -> HalfSpace:
    """Reachable halfspace ``y[0] >= 0.5`` for the 3D banana classifier.

    Phase 2 documented ``y[0] <= 0.5`` (i.e. ``y[0] > 0.5`` SAFE / SAT
    when encoded as UNSAFE = ``y[0] >= 0.5``) as a SAT-witnessing spec
    for the analogous 2D banana. Empirically the 3D classifier's
    ``y[0]`` ranges roughly over ``[-18, +22]`` on inputs in
    ``[-1, 1]^3`` (see _build_sat_spec docstring; verified in smoke
    runs), so ``y[0] >= 0.5`` is easily reachable.

    Encoding convention (per ``project_sat_unsat_convention``):
    ``G y <= g`` defines the UNSAFE region. We want unsafe = ``y[0]
    >= 0.5``, i.e. ``-y[0] <= -0.5``.
    """
    G = np.array([[-1.0, 0.0, 0.0]])  # -y[0] <= -0.5  <=>  y[0] >= 0.5
    g = np.array([[-0.5]])
    return HalfSpace(G, g)


def _flow_volume(result: dict, y_train: torch.Tensor, output_dim: int,
                 n_mc: int = 200_000) -> tuple[float, float]:
    """MC-estimate the volume of the calibrated reach set ``{y : score(y) <= q}``."""
    y_lo = y_train.min(dim=0).values
    y_hi = y_train.max(dim=0).values
    pad = 0.05 * (y_hi - y_lo).clamp(min=1e-6)
    bbox = (y_lo - pad, y_hi + pad)
    s = ProbabilisticSet(
        score_fn=result['score_fn'], threshold=result['q'],
        m=8000, ell=7999, epsilon=_ALPHA, dim=output_dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc, bounding_box=bbox)
    return float(vol), float(se)


def _sample_outputs(net, x_center: np.ndarray, radius: float,
                    n_samples: int, seed: int) -> torch.Tensor:
    """Uniform input-box sampling pushed through the network."""
    from n2v.probabilistic.flow.sampling import sample_l_inf_ball
    x_center_t = torch.as_tensor(x_center, dtype=torch.float32)
    x = sample_l_inf_ball(
        x_center=x_center_t, radius=radius, n_samples=n_samples,
        seed=seed, dim=x_center.shape[0],
    )
    with torch.no_grad():
        return net(x)


def run_one_seed(seed: int, *, smoke: bool = False) -> dict:
    """Run one seed of the AMLS pipeline on the SAT-spec 3D banana."""
    torch.manual_seed(seed)
    net = ThreeBlobClassifier3D().eval()
    spec = _build_sat_spec()
    lb = (_X_CENTER - _RADIUS).astype(np.float64)
    ub = (_X_CENTER + _RADIUS).astype(np.float64)

    flow_epochs = 200 if smoke else _FLOW_EPOCHS
    n_train = 1_000 if smoke else _N_TRAIN
    scen_n = 200 if smoke else _SCENARIO_N

    t0 = time.time()
    result = run_verification_pipeline(
        network=net,
        input_lb=lb, input_ub=ub, spec=spec,
        alpha=_ALPHA,
        n_train=n_train, flow_epochs=flow_epochs,
        flow_config='base',
        scenario_n_samples=scen_n, scenario_beta=_SCENARIO_BETA,
        verification_method=_VERIFICATION_METHOD,
        seed=seed,
        use_falsifier=False,  # Exp 3: no falsifier per finalized plan
        sat_backend='none',
    )
    total_s = time.time() - t0

    if result['verdict'] in ('UNSAT', 'UNKNOWN') and result.get('score_fn') is not None:
        y_train = _sample_outputs(net, _X_CENTER, _RADIUS, n_train, seed)
        n_mc_vol = 50_000 if smoke else 200_000
        vol, _ = _flow_volume(result, y_train, _OUTPUT_DIM, n_mc=n_mc_vol)
    else:
        vol = float('nan')

    return {
        'seed': seed,
        'verdict': result['verdict'],
        'q': result.get('q'),
        'volume_estimate': vol,
        'coverage_empirical': result.get('coverage_empirical'),
        'train_s': result.get('flow_train_time_s'),
        'verify_s': result.get('verification_time_s'),
        'total_s': total_s,
        'amls_levels_used': result.get('amls_levels_used', ''),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='1 seed, reduced epochs; ~30-60s')
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    print('Computing Star-union ground truth (SAT spec; '
          'used only for ratio reporting)...', flush=True)
    if args.smoke:
        torch.manual_seed(0)
        gt_net = ThreeBlobClassifier3D().eval()
        star_vol, stars = exact_star_union_volume(
            gt_net, x_center=_X_CENTER, radius=_RADIUS,
            output_dim=_OUTPUT_DIM, n_mc=20_000,
        )
    else:
        torch.manual_seed(0)
        gt_net = ThreeBlobClassifier3D().eval()
        star_vol, stars = exact_star_union_volume(
            gt_net, x_center=_X_CENTER, radius=_RADIUS,
            output_dim=_OUTPUT_DIM,
        )
    floor = (1.0 - _ALPHA) * star_vol
    print(f'  n_stars = {len(stars)}  Star-union vol = {star_vol:.4f}  '
          f'(1-alpha)*vol = {floor:.4f}', flush=True)

    seeds = [0] if args.smoke else list(range(args.seeds))
    out_path = _OUT_CSV if not args.smoke else _OUT_CSV.with_suffix('.smoke.csv')

    fields = ['seed', 'verdict', 'q', 'volume_estimate',
              'volume_ratio_vs_exact', 'coverage_empirical',
              'train_s', 'verify_s', 'total_s', 'amls_levels_used']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            print(f'[seed={seed}] running...', flush=True)
            row = run_one_seed(seed, smoke=args.smoke)
            v = row['volume_estimate']
            ratio = (v / floor) if (floor > 0 and v == v) else float('nan')
            row['volume_ratio_vs_exact'] = ratio
            print(f'  verdict={row["verdict"]} q={row["q"]} '
                  f'vol={v if v != v else f"{v:.4f}"} '
                  f'ratio={ratio:.3f} cov={row["coverage_empirical"]} '
                  f'total_s={row["total_s"]:.1f}', flush=True)
            writer.writerow({k: row.get(k, '') for k in fields})
            f.flush()
    print(f'\nWrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
