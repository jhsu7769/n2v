"""Flow-conformal + AMLS verification pipeline (Plan B integration).

Public API:
    run_verification_pipeline: main entry point. Falsifier is OPT-IN via
        ``use_falsifier=True`` (default OFF — pure flow-conformal+AMLS).

This is the modern flow-matching-based probabilistic verifier with an
optional falsifier-first stage. The legacy
``n2v.probabilistic.verify`` (Hashemi-style naive / clipping_block
surrogates) is unchanged and remains the parallel API for that approach.

Default behavior:
    use_falsifier=False — Stage 1 disabled, the pipeline goes directly
    to flow training + scenario / AMLS verification, returning UNSAT or
    UNKNOWN only (never SAT).

For backward compatibility with existing example scripts that relied on
Stage 1 running by default, the shim at
``examples.FlowConformal.benchmarks._common.run_verification_pipeline``
overrides this default to ``use_falsifier=True`` and is the path used by
the ACAS-Xu sweep / Phase 5 ablation harnesses.

The implementation here is deliberately self-contained: imports of
example-side helpers (``spec_summary``, ``certify_spec_on_flow``) are
deferred to call time so this module is importable even when the
examples package is not on ``sys.path``.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch
from scipy.stats import beta as _beta_dist

from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.probabilistic.flow.train import train_flow
from n2v.sets.halfspace import HalfSpace
from n2v.utils.falsify import falsify


# ---- Whitening glue --------------------------------------------------
#
# ACAS-Xu networks produce outputs whose per-dim std is 1e-4 to 1e-2 on
# the tight VNN-LIB input boxes. A flow trained with OT-CFM cannot bridge
# that ~1000x scale gap in finite time — it converges to a near-identity
# that leaves ||phi(y)|| dominated by the data-space offset ||mu||.
#
# The fix below pre-whitens y in the pipeline before the flow sees it,
# trains with the model's internal standardize_outputs disabled (to avoid
# double-whitening), and transforms the halfspace spec into whitened
# coordinates so verify_spec_on_flow operates in the same frame. All
# joint conformal-scenario guarantees carry over unchanged because the
# whitening transform (y - mu) / sigma is a deterministic invertible
# affine map and the flow is the interesting non-linear part downstream.


class _WhitenedNetwork:
    """Callable wrapper around a network that whitens its outputs.

    ``whitened_network(x) = (network(x) - mu) / sigma`` (per-dim).

    Used to hand scenario-verify / preimage-search a network whose
    outputs live in the same whitened coordinates as the flow.
    """

    def __init__(self, net, y_mean: torch.Tensor, y_std: torch.Tensor):
        self.net = net
        self.y_mean = y_mean
        self.y_std = y_std

    def __call__(self, x):
        y = self.net(x)
        dev = y.device
        return (y - self.y_mean.to(dev)) / self.y_std.to(dev)

    def eval(self):
        if hasattr(self.net, 'eval'):
            self.net.eval()
        return self

    def parameters(self):
        if hasattr(self.net, 'parameters'):
            yield from self.net.parameters()


class _WhiteningFlowScore:
    """Score function that whitens its input before delegating.

    Lets callers (e.g. volume validation) keep passing raw network
    outputs: whitening happens transparently before the underlying
    :class:`FlowScore` operates.
    """

    def __init__(self, base_score_fn, y_mean: torch.Tensor,
                 y_std: torch.Tensor):
        self.base = base_score_fn
        self.y_mean = y_mean
        self.y_std = y_std

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        dev = y.device
        y_w = (y - self.y_mean.to(dev)) / self.y_std.to(dev)
        return self.base(y_w)

    @property
    def flow_model(self):
        return self.base.flow_model


class _MaxEnsembleFlowScore:
    """Conservative ensemble score: max over a list of FlowScore instances.

    Score(y) = max_j score_j(y), where each score_j is a single-flow
    FlowScore. Higher score = "more out-of-distribution" under at least
    one flow. The conservative max-score makes calibrated q LARGER than
    any single-flow q (since max >= each individual), trading tightness
    for stability against flow-init variance.
    """

    def __init__(self, score_fns: list):
        if len(score_fns) == 0:
            raise ValueError("score_fns must be non-empty")
        self.score_fns = score_fns

    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        scores = torch.stack([s(y) for s in self.score_fns], dim=0)
        return scores.max(dim=0).values

    @property
    def flow_model(self):
        # Echo the first flow's model — the wrapper used by external
        # callers (volume validation etc.) only needs *some* underlying
        # model handle for diagnostics.
        return self.score_fns[0].flow_model


def _whiten_halfspace(spec: HalfSpace, y_mean: np.ndarray,
                      y_std: np.ndarray) -> HalfSpace:
    """Transform ``G @ y <= g`` to the equivalent constraint on whitened
    coordinates ``y_w = (y - mu) / sigma``:

        G @ y <= g
        G @ (sigma * y_w + mu) <= g
        (G * sigma) @ y_w <= g - G @ mu
    """
    sigma = np.asarray(y_std, dtype=np.float64).flatten()
    mu = np.asarray(y_mean, dtype=np.float64).flatten()
    G_white = spec.G * sigma[None, :]  # row-wise elementwise scale
    g_white = spec.g.flatten() - spec.G @ mu
    return HalfSpace(G_white, g_white.reshape(-1, 1))


# ---- Flow training helpers -------------------------------------------


def _forward(net, x):
    with torch.no_grad():
        # Push x to network's device so callers that .to('cuda') the
        # network get GPU forward passes for sample generation /
        # calibration / test. CPU-only callers see no behavior change
        # (the .to() is a no-op when device matches).
        # Some ONNX-converted networks (e.g. ACAS Xu via onnx2torch)
        # have only buffers, no nn.Parameters; check both before
        # falling back to CPU.
        try:
            target_device = next(net.parameters()).device
        except StopIteration:
            try:
                target_device = next(net.buffers()).device
            except StopIteration:
                target_device = torch.device('cpu')
        x_t = torch.as_tensor(x, dtype=torch.float32).to(target_device)
        return net(x_t)


def _train_flow(y_train: torch.Tensor, dim: int, n_epochs: int, seed: int,
                batch_size: int = 2048, sinkhorn_iters: int = 10,
                hidden: int = 128, n_layers: int = 4,
                time_embed: str = 'concat',
                time_sampling: str = 'uniform',
                internal_standardize: bool = True,
                return_losses: bool = False,
                coupling: str = 'sinkhorn',
                use_ema: bool = True):
    """Production-grade OT-CFM. Runs GPU-end-to-end.

    ``internal_standardize``: pass-through to ``train_flow``'s
    ``standardize_outputs`` argument. Callers that pre-whiten the
    training data externally (e.g. ``run_verification_pipeline``) must
    pass False to avoid double-whitening and to keep the flow operating
    end-to-end in whitened coordinates rather than data coordinates.

    ``return_losses``: if True, return ``(FlowODE, losses)`` tuple
    instead of just the FlowODE; the per-epoch loss list is used by
    callers that want to record the final training loss.

    ``coupling``: pass-through to ``train_flow``'s ``coupling`` argument.
    Default 'sinkhorn' matches pre-exposure behavior bit-identically.

    ``use_ema``: pass-through to ``train_flow``'s ``use_ema`` argument.
    Default True matches pre-exposure behavior bit-identically.
    """
    torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vf = VelocityField(dim=dim, hidden=hidden, n_layers=n_layers,
                       activation='silu', time_embed=time_embed).to(device)
    y_train = y_train.to(device)
    vf, losses = train_flow(
        vf, y_train, n_epochs=n_epochs, batch_size=batch_size, lr=1e-3,
        coupling=coupling, sinkhorn_reg='auto', sinkhorn_iters=sinkhorn_iters,
        use_ema=use_ema, standardize_outputs=internal_standardize,
        time_sampling=time_sampling,
    )
    vf.eval()
    flow = FlowODE(vf)
    if return_losses:
        return flow, losses
    return flow


def _train_flow_tight(y_train: torch.Tensor, dim: int, n_epochs: int,
                      seed: int, internal_standardize: bool = True,
                      return_losses: bool = False,
                      coupling: str = 'sinkhorn',
                      use_ema: bool = True):
    """Higher-capacity, longer-training config for ThreeBlobClassifier3D-
    class multimodal output distributions.

    hidden=256, L=6, sinusoidal time embedding, logit-normal time sampling
    (concentrates t near 0.5 where interpolation is hardest). Meets the
    (c)+(e) experiment spec. Training cost scales linearly with n_epochs.
    """
    return _train_flow(
        y_train, dim=dim, n_epochs=n_epochs, seed=seed,
        batch_size=2048, sinkhorn_iters=10,
        hidden=256, n_layers=6,
        time_embed='sinusoidal', time_sampling='logit_normal',
        internal_standardize=internal_standardize,
        return_losses=return_losses,
        coupling=coupling,
        use_ema=use_ema,
    )


# ---- SAT result and scenario helpers ---------------------------------


def _sat_result(cex_x, cex_y, counterexample_source, sat_backend_time_s,
                spec_summary_str) -> dict:
    """Build the result dict for a SAT verdict (no flow training ran)."""
    return {
        'verdict': 'SAT',
        'counterexample': {'x': cex_x, 'y': cex_y},
        'counterexample_source': counterexample_source,
        # Probabilistic certificate fields: not applicable for SAT
        'epsilon_total': None, 'delta_total': None,
        'epsilon_1': None, 'delta_1': None,
        'epsilon_2': None, 'delta_2': None,
        # Flow fields: not populated (flow never trained)
        'flow': None, 'score_fn': None,
        'q': None, 'coverage_empirical': None,
        'y_mean': None, 'y_std': None,
        'scenario_result': None,
        # Timing fields
        'flow_train_time_s': 0.0,
        'flow_train_loss_final': None,
        'verification_time_s': 0.0,
        'sat_backend_time_s': sat_backend_time_s,
        'total_time_s': sat_backend_time_s,
        'spec_summary': spec_summary_str,
        'flow_ensemble_size': None,
    }


def _extract_min_worst_max_margin(scenario_result: dict) -> 'float | None':
    """Pull min over (group, halfspace) of worst_max_margin from a
    scenario_result dict (output of ``certify_spec_on_flow``).

    Returns None when there are no per-group / per-hs entries.
    """
    per_group = scenario_result.get('per_group_results') or []
    margins: list[float] = []
    for gr in per_group:
        for hs_res in (getattr(gr, 'per_hs_results', None) or []):
            wm = getattr(hs_res, 'worst_max_margin', None)
            if wm is not None:
                margins.append(float(wm))
    if not margins:
        return None
    return min(margins)


def _certify_spec_on_flow_v2(
    flow_ode,
    threshold_q: float,
    spec,
    *,
    n_samples: int,
    beta_2: float,
    base_seed: int,
    n_restarts: int = 5,
    adaptive_threshold: 'float | None' = None,
    adaptive_n_samples: 'int | None' = None,
) -> dict:
    """Multi-restart wrapper around ``certify_spec_on_flow`` (C0 / scenario_v2).

    Runs ``certify_spec_on_flow`` ``n_restarts`` times with seeds
    ``base_seed + i * 1_000_000`` and ``sampling_strategy='qmc+antithetic'``.
    Aggregates by taking the run with the SMALLEST min-worst-max-margin
    across all (group, halfspace) entries — the most conservative
    hypothesis test result. UNSAT iff every restart certifies UNSAT
    (which corresponds to all min-margins being positive). The aggregate
    epsilon_2 is the Bonferroni union bound: K * per-restart epsilon.
    Effective sample count is reported as K * N.

    Returns a dict in the same shape as ``certify_spec_on_flow``'s output
    plus an ``aggregate_metadata`` field describing the K runs.
    """
    # Lazy import: certify_spec_on_flow lives in the example-side _spec
    # helper (it composes load_vnnlib outputs into scenario_verify
    # calls). Importing it here keeps the n2v.probabilistic.verify_flow
    # module free of import-time dependencies on examples/.
    from examples.FlowConformal.benchmarks._spec import certify_spec_on_flow

    if n_restarts < 1:
        raise ValueError(f'n_restarts must be >= 1, got {n_restarts}')

    runs = []
    for k in range(n_restarts):
        run_seed = base_seed + k * 1_000_000
        r = certify_spec_on_flow(
            flow_ode=flow_ode, threshold_q=threshold_q, spec=spec,
            n_samples=n_samples, beta_2=beta_2, seed=run_seed,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
            sampling_strategy='qmc+antithetic',
        )
        runs.append(r)

    # Min margin across runs (most conservative).
    per_run_min_margin = []
    for r in runs:
        m = _extract_min_worst_max_margin(r)
        per_run_min_margin.append(m if m is not None else float('inf'))
    aggregate_min_margin = min(per_run_min_margin)
    worst_run_idx = int(np.argmin(per_run_min_margin))
    worst_run = runs[worst_run_idx]

    # UNSAT iff EVERY restart certified UNSAT (and the most conservative
    # min-margin is positive). Either condition implies the other in
    # practice — we report the conservative AND.
    all_unsat = all(r['unsat_certified'] for r in runs)
    unsat_certified = bool(all_unsat and aggregate_min_margin > 0)

    # Bonferroni: per-restart epsilon * K.
    epsilon_2_total = sum(r['epsilon_2'] for r in runs)

    return {
        'unsat_certified': unsat_certified,
        'certifying_group_idx': worst_run.get('certifying_group_idx'),
        'epsilon_2': epsilon_2_total,
        'delta_2': 1.0 - beta_2,
        'n_samples_used': n_samples * n_restarts,
        'per_group_results': worst_run['per_group_results'],
        'spec_summary': worst_run.get('spec_summary'),
        'aggregate_metadata': {
            'method': 'scenario_v2',
            'n_restarts': n_restarts,
            'per_run_min_margin': per_run_min_margin,
            'aggregate_min_margin': aggregate_min_margin,
            'worst_run_idx': worst_run_idx,
            'all_unsat': all_unsat,
        },
    }


# ---- Stage-2 (flow training + scenario / AMLS verify) ----------------


def _calibrate_flow_for_spec(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    alpha: float = 0.001,
    m: int = 8000,
    ell: int = 7999,
    n_train: int = 10_000,
    flow_epochs: int = 5000,
    flow_config: str = 'tight',
    seed: int = 0,
    flow_seed: 'int | None' = None,
    cal_seed: 'int | None' = None,
    infer_solver: str = 'rk4',
    infer_steps: int = 30,
    flow_ensemble_size: int = 1,
    flow_standardize: 'bool | None' = None,
    flow_coupling: 'str | None' = None,
    flow_use_ema: 'bool | None' = None,
) -> dict:
    """Steps 1-7 of the flow-conformal pipeline.

    Sample, whiten, train flow, calibrate q, compute empirical coverage,
    compute Hashemi double-step delta_1, parse + whiten spec. Returns
    everything :func:`_verify_with_calibration` needs to dispatch any
    verification method against the *same* (flow, q) tuple.

    Used by the verification-method ablation so the ablation only varies
    the verifier, not the calibration.
    """
    from examples.FlowConformal.benchmarks._spec import spec_summary

    flow_seed = flow_seed if flow_seed is not None else seed
    cal_seed = cal_seed if cal_seed is not None else seed

    lb_t = torch.as_tensor(input_lb, dtype=torch.float32)
    ub_t = torch.as_tensor(input_ub, dtype=torch.float32)
    x_tr = _sample_box(lb_t, ub_t, n_samples=n_train, seed=flow_seed)
    x_ca = _sample_box(lb_t, ub_t, n_samples=m, seed=cal_seed + 1_000_000)
    x_te = _sample_box(lb_t, ub_t, n_samples=2_000, seed=cal_seed + 2_000_000)
    y_tr = _forward(network, x_tr)
    y_ca = _forward(network, x_ca)
    y_te = _forward(network, x_te)

    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std
    y_te_w = (y_te - y_mean) / y_std

    if flow_ensemble_size < 1:
        raise ValueError(
            f"flow_ensemble_size must be >= 1, got {flow_ensemble_size}")
    t0 = time.time()
    output_dim = y_tr_w.shape[1]

    _internal_standardize = (
        bool(flow_standardize) if flow_standardize is not None else False
    )
    _coupling = flow_coupling if flow_coupling is not None else 'sinkhorn'
    _use_ema = bool(flow_use_ema) if flow_use_ema is not None else True

    def _train_one(seed_j: int):
        if flow_config == 'base':
            return _train_flow(
                y_tr_w, output_dim, flow_epochs, seed_j,
                internal_standardize=_internal_standardize,
                return_losses=True,
                coupling=_coupling,
                use_ema=_use_ema)
        if flow_config == 'tight':
            return _train_flow_tight(
                y_tr_w, output_dim, flow_epochs, seed_j,
                internal_standardize=_internal_standardize,
                return_losses=True,
                coupling=_coupling,
                use_ema=_use_ema)
        raise ValueError(f"unknown flow_config {flow_config!r}")

    flows = []
    flow_losses = None
    for j in range(flow_ensemble_size):
        seed_j = flow_seed + j * 1_000_000
        f_j, losses_j = _train_one(seed_j)
        f_j = f_j.to('cpu').eval()
        flows.append(f_j)
        if j == 0:
            flow_losses = losses_j
    flow = flows[0]
    flow_train_loss_final = float(flow_losses[-1]) if flow_losses else None
    train_time = time.time() - t0

    per_flow_scores = [
        FlowScore(f, t=1.0, n_steps=infer_steps, method=infer_solver,
                  batch_size=65536)
        for f in flows
    ]
    if flow_ensemble_size == 1:
        base_score_fn = per_flow_scores[0]
    else:
        base_score_fn = _MaxEnsembleFlowScore(per_flow_scores)
    calib_scores = base_score_fn(y_ca_w)
    q = calibrate(calib_scores, ell).item()

    s = ProbabilisticSet(
        score_fn=base_score_fn, threshold=q,
        m=m, ell=ell, epsilon=alpha, dim=output_dim,
    )
    coverage_empirical = s.contains(y_te_w).float().mean().item()

    delta_1 = 1.0 - float(_beta_dist.cdf(1.0 - alpha, ell, m + 1 - ell))
    epsilon_1 = alpha

    from n2v.utils.verify_specification import (
        _parse_property_groups, distribute_and_of_or_of_and,
    )

    y_mean_np = y_mean.detach().cpu().numpy()
    y_std_np = y_std.detach().cpu().numpy()

    raw_groups = _parse_property_groups(spec)
    raw_groups = distribute_and_of_or_of_and(raw_groups)
    whitened_groups = [
        [_whiten_halfspace(hs, y_mean_np, y_std_np) for hs in group]
        for group in raw_groups
    ]

    return {
        'flow': flow, 'flows': flows,
        'base_score_fn': base_score_fn,
        'q': q,
        'y_mean': y_mean, 'y_std': y_std,
        'y_mean_np': y_mean_np, 'y_std_np': y_std_np,
        'whitened_groups': whitened_groups,
        'output_dim': output_dim,
        'coverage_empirical': coverage_empirical,
        'epsilon_1': epsilon_1, 'delta_1': delta_1,
        'flow_train_time_s': train_time,
        'flow_losses': flow_losses,
        'flow_train_loss_final': flow_train_loss_final,
        'flow_ensemble_size': flow_ensemble_size,
        'alpha': alpha, 'm': m, 'ell': ell,
        'spec': spec,
        'spec_summary': spec_summary(spec),
    }


def _verify_with_calibration(
    calib: dict,
    *,
    verification_method: str = 'scenario',
    scenario_n_samples: int = 10_000,
    scenario_beta: float = 0.001,
    scenario_seed: 'int | None' = None,
    adaptive_threshold: float = 0.5,
    adaptive_n_samples: int = 20000,
    sampling_strategy: str = 'uniform',
    sat_backend_time_s: float = 0.0,
    amls_quantile: 'float | None' = None,
    amls_n_mcmc_steps: 'int | None' = None,
    amls_mcmc_step_size: 'float | None' = None,
    amls_n_samples_per_level: 'int | None' = None,
    amls_max_levels: 'int | None' = None,
    amls_bounded_eps_2_target: 'float | None' = None,
    amls_bounded_adaptive_step: bool = False,
) -> dict:
    """Steps 8-9 of the flow-conformal pipeline.

    Dispatches the named verifier against a pre-built ``calib`` bundle
    from :func:`_calibrate_flow_for_spec` and assembles the joint
    (epsilon, delta) certificate. The same calib can be reused across
    multiple verification methods so the ablation only varies the
    verifier.
    """
    from examples.FlowConformal.benchmarks._spec import (
        certify_spec_on_flow, spec_summary,
    )

    if scenario_seed is None:
        scenario_seed = 0

    flow = calib['flow']
    q = calib['q']
    whitened_groups = calib['whitened_groups']
    spec = calib['spec']
    alpha = calib['alpha']
    base_score_fn = calib['base_score_fn']
    y_mean = calib['y_mean']
    y_std = calib['y_std']
    y_mean_np = calib['y_mean_np']
    y_std_np = calib['y_std_np']
    coverage_empirical = calib['coverage_empirical']
    epsilon_1 = calib['epsilon_1']
    delta_1 = calib['delta_1']
    train_time = calib['flow_train_time_s']
    flow_train_loss_final = calib['flow_train_loss_final']
    flow_losses = calib['flow_losses']
    flow_ensemble_size = calib['flow_ensemble_size']

    # AMLS-bounded mutates the flow's device (calls flow_ode.to(dev)).
    # When a single calibration is reused across multiple verifiers,
    # subsequent CPU-only verifiers (is_tilted, derived) hit a device-
    # mismatch error. Snap back to CPU per verifier call.
    if hasattr(flow, 'to'):
        flow = flow.to('cpu').eval()

    t1 = time.time()
    if verification_method == 'scenario':
        scenario_result = certify_spec_on_flow(
            flow_ode=flow,
            threshold_q=q,
            spec=whitened_groups,
            n_samples=scenario_n_samples,
            beta_2=scenario_beta,
            seed=scenario_seed,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
            sampling_strategy=sampling_strategy,
        )
    elif verification_method == 'scenario_v2':
        scenario_result = _certify_spec_on_flow_v2(
            flow_ode=flow,
            threshold_q=q,
            spec=whitened_groups,
            n_samples=scenario_n_samples,
            beta_2=scenario_beta,
            base_seed=scenario_seed,
            n_restarts=5,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
        )
    elif verification_method == 'amls':
        from n2v.probabilistic.flow.amls import amls_certify_spec
        _amls_quantile = amls_quantile if amls_quantile is not None else 0.1
        _amls_max_levels = (
            amls_max_levels if amls_max_levels is not None else 30)
        _amls_n_mcmc_steps = (
            amls_n_mcmc_steps if amls_n_mcmc_steps is not None else 10)
        _amls_mcmc_step_size = (
            amls_mcmc_step_size
            if amls_mcmc_step_size is not None else 0.3)
        _amls_n_samples_per_level = (
            amls_n_samples_per_level
            if amls_n_samples_per_level is not None
            else scenario_n_samples)
        amls_result = amls_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_samples_per_level=_amls_n_samples_per_level,
            quantile=_amls_quantile,
            max_levels=_amls_max_levels,
            n_mcmc_steps=_amls_n_mcmc_steps,
            mcmc_step_size=_amls_mcmc_step_size,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': amls_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'amls_result': amls_result,
            'amls_detected_unsafe': amls_result.detected_any,
            'amls_levels_used': max(
                (r.levels_used for grp in amls_result.per_hs_results for r in grp),
                default=0,
            ),
        }
    elif verification_method in ('amls_bounded', 'amls_bounded_union'):
        from n2v.probabilistic.flow.amls_bounded import (
            amls_bounded_certify_spec,
            amls_bounded_certify_spec_union,
        )
        _amls_quantile = amls_quantile if amls_quantile is not None else 0.1
        _amls_max_levels = (
            amls_max_levels if amls_max_levels is not None else 30)
        _amls_n_mcmc_steps = (
            amls_n_mcmc_steps if amls_n_mcmc_steps is not None else 10)
        _amls_mcmc_step_size = (
            amls_mcmc_step_size
            if amls_mcmc_step_size is not None else 0.3)
        _amls_n_samples_per_level = (
            amls_n_samples_per_level
            if amls_n_samples_per_level is not None
            else scenario_n_samples)
        _eps_2_target = (
            amls_bounded_eps_2_target
            if amls_bounded_eps_2_target is not None else alpha)
        _certify_fn = (
            amls_bounded_certify_spec_union
            if verification_method == 'amls_bounded_union'
            else amls_bounded_certify_spec
        )
        amls_b_result = _certify_fn(
            flow_ode=flow,
            spec_groups=whitened_groups,
            q=q,
            eps_2_target=_eps_2_target,
            n_samples_per_level=_amls_n_samples_per_level,
            quantile=_amls_quantile,
            max_levels=_amls_max_levels,
            n_mcmc_steps=_amls_n_mcmc_steps,
            mcmc_step_size=_amls_mcmc_step_size,
            adaptive_step=amls_bounded_adaptive_step,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': amls_b_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': float(amls_b_result.eps_2_upper),
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'amls_bounded_result': amls_b_result,
            'amls_bounded_detected_unsafe': amls_b_result.detected_any,
            'amls_bounded_eps_2_upper': float(amls_b_result.eps_2_upper),
            'amls_bounded_eps_2_target': float(_eps_2_target),
            'amls_bounded_adaptive_step': amls_bounded_adaptive_step,
            'amls_bounded_mode': verification_method,
            'amls_levels_used': max(
                (r.levels_used
                 for grp in amls_b_result.per_hs_results for r in grp),
                default=0,
            ),
        }
    elif verification_method == 'derived':
        from n2v.probabilistic.flow.langevin_sampler import (
            langevin_certify_spec,
        )
        chain_count = 100
        step_total = max(1, int(scenario_n_samples) // chain_count)
        warmup = max(50, step_total // 4)
        sample_steps = max(50, step_total - warmup)
        lang_result = langevin_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_chains=chain_count,
            n_warmup=warmup,
            n_samples=sample_steps,
            step_size=0.05,
            lambda_tilt=5.0,
            use_mala=False,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': lang_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'derived_result': lang_result,
            'derived_detected_unsafe': lang_result.detected_any,
            'derived_min_final_phi': min(
                (r.final_phi for grp in lang_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
            'derived_mean_grad_norm': max(
                (r.mean_grad_norm for grp in lang_result.per_hs_results
                 for r in grp), default=0.0,
            ),
        }
    elif verification_method == 'raw_mc_uniform':
        from n2v.probabilistic.flow.raw_mc_uniform import (
            raw_mc_certify_spec,
        )
        _eps_2_target = (
            amls_bounded_eps_2_target
            if amls_bounded_eps_2_target is not None else alpha)
        raw_mc_result = raw_mc_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            q=q,
            eps_2_target=_eps_2_target,
            n_samples=scenario_n_samples,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': raw_mc_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': float(raw_mc_result.eps_2_upper),
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'raw_mc_result': raw_mc_result,
            'raw_mc_detected_unsafe': raw_mc_result.detected_any,
            'raw_mc_eps_2_upper': float(raw_mc_result.eps_2_upper),
            'raw_mc_eps_2_target': float(_eps_2_target),
        }
    elif verification_method == 'is_tilted':
        from n2v.probabilistic.flow.importance_sampling import (
            is_tilted_certify_spec,
        )
        is_result = is_tilted_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_samples=scenario_n_samples,
            lambda_tilt=5.0,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': is_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'is_result': is_result,
            'is_detected_unsafe': is_result.detected_any,
            'is_min_final_phi': min(
                (r.final_phi for grp in is_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
            'is_min_ess': min(
                (r.ess for grp in is_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
        }
    else:
        raise ValueError(
            f'unsupported verification_method: {verification_method!r}')
    verify_time = time.time() - t1

    epsilon_2 = scenario_result['epsilon_2']
    delta_2 = scenario_result['delta_2']
    epsilon_total = 1.0 - (1.0 - epsilon_1) * (1.0 - epsilon_2)
    delta_total = delta_1 * delta_2

    if verification_method == 'amls' and scenario_result.get(
            'amls_detected_unsafe', False):
        verdict = 'UNKNOWN'
    elif verification_method == 'is_tilted' and scenario_result.get(
            'is_detected_unsafe', False):
        verdict = 'UNKNOWN'
    elif verification_method == 'derived' and scenario_result.get(
            'derived_detected_unsafe', False):
        verdict = 'UNKNOWN'
    else:
        verdict = 'UNSAT' if scenario_result['unsat_certified'] else 'UNKNOWN'

    score_fn = _WhiteningFlowScore(base_score_fn, y_mean.cpu(), y_std.cpu())

    out = {
        'verdict': verdict,
        'epsilon_total': epsilon_total,
        'delta_total': delta_total,
        'epsilon_1': epsilon_1, 'delta_1': delta_1,
        'epsilon_2': epsilon_2, 'delta_2': delta_2,
        'counterexample': None,
        'counterexample_source': None,
        'flow_train_time_s': train_time,
        'flow_train_loss_final': flow_train_loss_final,
        'flow_train_loss_curve': (
            [float(x) for x in flow_losses] if flow_losses else []
        ),
        'verification_time_s': verify_time,
        'sat_backend_time_s': sat_backend_time_s,
        'total_time_s': train_time + verify_time + sat_backend_time_s,
        'coverage_empirical': coverage_empirical,
        'q': q,
        'flow': flow,
        'score_fn': score_fn,
        'y_mean': y_mean_np,
        'y_std': y_std_np,
        'scenario_result': scenario_result,
        'spec_summary': spec_summary(spec),
        'flow_ensemble_size': flow_ensemble_size,
    }
    if verification_method == 'amls':
        out['amls_detected_unsafe'] = scenario_result.get(
            'amls_detected_unsafe', False)
        out['amls_levels_used'] = scenario_result.get('amls_levels_used', 0)
    if verification_method in ('amls_bounded', 'amls_bounded_union'):
        out['amls_bounded_eps_2_upper'] = scenario_result.get(
            'amls_bounded_eps_2_upper', float('inf'))
        out['amls_bounded_detected_unsafe'] = scenario_result.get(
            'amls_bounded_detected_unsafe', False)
        out['amls_levels_used'] = scenario_result.get('amls_levels_used', 0)
    if verification_method == 'raw_mc_uniform':
        out['raw_mc_eps_2_upper'] = scenario_result.get(
            'raw_mc_eps_2_upper', float('inf'))
        out['raw_mc_detected_unsafe'] = scenario_result.get(
            'raw_mc_detected_unsafe', False)
    if verification_method == 'is_tilted':
        out['is_detected_unsafe'] = scenario_result.get(
            'is_detected_unsafe', False)
        out['is_min_final_phi'] = scenario_result.get(
            'is_min_final_phi', float('inf'))
        out['is_min_ess'] = scenario_result.get(
            'is_min_ess', float('inf'))
    if verification_method == 'derived':
        out['derived_detected_unsafe'] = scenario_result.get(
            'derived_detected_unsafe', False)
        out['derived_min_final_phi'] = scenario_result.get(
            'derived_min_final_phi', float('inf'))
        out['derived_mean_grad_norm'] = scenario_result.get(
            'derived_mean_grad_norm', 0.0)
    return out


def _flow_unsat_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    alpha: float = 0.001,
    m: int = 8000,
    ell: int = 7999,
    scenario_n_samples: int = 10_000,
    scenario_beta: float = 0.001,
    n_train: int = 10_000,
    flow_epochs: int = 5000,
    flow_config: str = 'tight',
    seed: int = 0,
    flow_seed: 'int | None' = None,
    cal_seed: 'int | None' = None,
    scenario_seed: 'int | None' = None,
    infer_solver: str = 'rk4',
    infer_steps: int = 30,
    adaptive_threshold: float = 0.5,
    adaptive_n_samples: int = 20000,
    sampling_strategy: str = 'uniform',
    flow_ensemble_size: int = 1,
    sat_backend_time_s: float = 0.0,
    verification_method: str = 'scenario',
    # AMLS hyperparameter overrides (None = use legacy hardcoded defaults).
    amls_quantile: 'float | None' = None,
    amls_n_mcmc_steps: 'int | None' = None,
    amls_mcmc_step_size: 'float | None' = None,
    amls_n_samples_per_level: 'int | None' = None,
    amls_max_levels: 'int | None' = None,
    # Bounded-AMLS overrides (used only when verification_method=='amls_bounded').
    amls_bounded_eps_2_target: 'float | None' = None,
    amls_bounded_adaptive_step: bool = False,
    # Flow-training knob overrides (None = use legacy hardcoded defaults).
    flow_standardize: 'bool | None' = None,
    flow_coupling: 'str | None' = None,
    flow_use_ema: 'bool | None' = None,
) -> dict:
    """Flow-training + scenario-verify pipeline, returning UNSAT or
    UNKNOWN only.

    This helper contains the original Phase-4 steps 1–9 of
    :func:`run_verification_pipeline` (sample, whiten, train, calibrate,
    coverage, delta_1, parse spec, scenario verify, joint certificate).
    It does NOT call any falsifier. The top-level
    :func:`run_verification_pipeline` runs the falsifier first; this
    helper is only invoked when the falsifier did not produce a
    validated counterexample.

    The ``sat_backend_time_s`` argument is the wall-clock cost of the
    Stage-1 falsifier (already paid by the caller); it is rolled into
    the returned ``total_time_s`` so timings stay accurate end-to-end.
    """
    # Lazy import: spec_summary lives in the example-side _spec helper.
    from examples.FlowConformal.benchmarks._spec import (
        certify_spec_on_flow, spec_summary,
    )

    flow_seed = flow_seed if flow_seed is not None else seed
    cal_seed = cal_seed if cal_seed is not None else seed
    scenario_seed = scenario_seed if scenario_seed is not None else seed

    # 1. Sample input box.
    lb_t = torch.as_tensor(input_lb, dtype=torch.float32)
    ub_t = torch.as_tensor(input_ub, dtype=torch.float32)
    x_tr = _sample_box(lb_t, ub_t, n_samples=n_train, seed=flow_seed)
    x_ca = _sample_box(lb_t, ub_t, n_samples=m, seed=cal_seed + 1_000_000)
    x_te = _sample_box(lb_t, ub_t, n_samples=2_000, seed=cal_seed + 2_000_000)
    y_tr = _forward(network, x_tr)
    y_ca = _forward(network, x_ca)
    y_te = _forward(network, x_te)

    # 2. Whiten. The flow operates end-to-end on coordinates
    #   y_w = (y - y_mean) / y_std
    # and everything downstream (calibration, scenario-verify, preimage
    # search, spec) is transformed into the same frame. See the module-
    # level comment "Whitening glue".
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std
    y_te_w = (y_te - y_mean) / y_std

    # 3. Train flow on pre-whitened data (no internal double-whitening).
    # When flow_ensemble_size > 1, train K flows with seeds offset by
    # j*1_000_000 and use a conservative max-score for calibration.
    if flow_ensemble_size < 1:
        raise ValueError(
            f"flow_ensemble_size must be >= 1, got {flow_ensemble_size}")
    t0 = time.time()
    output_dim = y_tr_w.shape[1]

    # Resolve flow-training kwargs, falling back to the legacy hardcoded
    # defaults when the caller did not override them. Default behavior is
    # bit-identical when all overrides are None.
    _internal_standardize = (
        bool(flow_standardize) if flow_standardize is not None else False
    )
    _coupling = flow_coupling if flow_coupling is not None else 'sinkhorn'
    _use_ema = bool(flow_use_ema) if flow_use_ema is not None else True

    def _train_one(seed_j: int):
        if flow_config == 'base':
            return _train_flow(
                y_tr_w, output_dim, flow_epochs, seed_j,
                internal_standardize=_internal_standardize,
                return_losses=True,
                coupling=_coupling,
                use_ema=_use_ema)
        if flow_config == 'tight':
            return _train_flow_tight(
                y_tr_w, output_dim, flow_epochs, seed_j,
                internal_standardize=_internal_standardize,
                return_losses=True,
                coupling=_coupling,
                use_ema=_use_ema)
        raise ValueError(f"unknown flow_config {flow_config!r}")

    flows = []
    flow_losses = None
    for j in range(flow_ensemble_size):
        seed_j = flow_seed + j * 1_000_000
        f_j, losses_j = _train_one(seed_j)
        # Move each flow to CPU so downstream scenario_verify (which
        # constructs CPU latent samples and uses CPU target_fn) is
        # device-consistent. FlowScore already cross-device-handles, so
        # calibration still works.
        f_j = f_j.to('cpu').eval()
        flows.append(f_j)
        if j == 0:
            flow_losses = losses_j  # report the first flow's loss curve
    flow = flows[0]
    flow_train_loss_final = float(flow_losses[-1]) if flow_losses else None
    train_time = time.time() - t0

    # 4. Calibrate on whitened calibration samples.
    per_flow_scores = [
        FlowScore(f, t=1.0, n_steps=infer_steps, method=infer_solver,
                  batch_size=65536)
        for f in flows
    ]
    if flow_ensemble_size == 1:
        base_score_fn = per_flow_scores[0]
    else:
        base_score_fn = _MaxEnsembleFlowScore(per_flow_scores)
    calib_scores = base_score_fn(y_ca_w)
    q = calibrate(calib_scores, ell).item()

    # 5. Empirical coverage (diagnostic) on whitened test samples.
    s = ProbabilisticSet(
        score_fn=base_score_fn, threshold=q,
        m=m, ell=ell, epsilon=alpha, dim=output_dim,
    )
    coverage_empirical = s.contains(y_te_w).float().mean().item()

    # 6. Hashemi double-step confidence δ_1.
    delta_1 = 1.0 - float(_beta_dist.cdf(1.0 - alpha, ell, m + 1 - ell))
    epsilon_1 = alpha

    # 7. Normalize spec into canonical list[list[HalfSpace]] and whiten
    # each HalfSpace in place. Every layer of scenario-verify (1/2/3)
    # operates in whitened coordinates.
    #
    # AMLS-bounded gates per HalfSpace's rare-event probability against
    # ``eps_2_target``. For multi-group AND-of-OR specs the per-group
    # halfspace masses can each exceed ``eps_2_target`` while the true
    # AND-conjunction unsafe mass is much smaller (see lsnc_relu which
    # has 0% AND-conjunction mass empirically but per-group masses up
    # to 23%). To make the AMLS estimates reflect the actual unsafe-
    # region probability, distribute AND-of-OR-of-AND into a single
    # OR-of-AND-conjunction group before passing to AMLS. This is
    # mathematically equivalent (each compound disjunct's mass is the
    # cross-group AND-conjunction mass) but lets the per-halfspace gate
    # bound the right quantity.
    from n2v.utils.verify_specification import (
        _parse_property_groups, distribute_and_of_or_of_and,
    )

    y_mean_np = y_mean.detach().cpu().numpy()
    y_std_np = y_std.detach().cpu().numpy()

    raw_groups = _parse_property_groups(spec)
    raw_groups = distribute_and_of_or_of_and(raw_groups)
    whitened_groups = [
        [_whiten_halfspace(hs, y_mean_np, y_std_np) for hs in group]
        for group in raw_groups
    ]

    # 8. UNSAT-certification via the three-layer dispatcher. This lane
    # is strictly UNSAT-only — it never returns SAT. SAT detection
    # happens in Stage 1 of the top-level pipeline.
    t1 = time.time()
    if verification_method == 'scenario':
        scenario_result = certify_spec_on_flow(
            flow_ode=flow,
            threshold_q=q,
            spec=whitened_groups,
            n_samples=scenario_n_samples,
            beta_2=scenario_beta,
            seed=scenario_seed,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
            sampling_strategy=sampling_strategy,
        )
    elif verification_method == 'scenario_v2':
        scenario_result = _certify_spec_on_flow_v2(
            flow_ode=flow,
            threshold_q=q,
            spec=whitened_groups,
            n_samples=scenario_n_samples,
            beta_2=scenario_beta,
            base_seed=scenario_seed,
            n_restarts=5,
            adaptive_threshold=adaptive_threshold,
            adaptive_n_samples=adaptive_n_samples,
        )
    elif verification_method == 'amls':
        # C1 (Adaptive Multilevel Splitting): rare-event estimator
        # targeted at unsafe-region detection.
        from n2v.probabilistic.flow.amls import amls_certify_spec
        _amls_quantile = (
            amls_quantile if amls_quantile is not None else 0.1
        )
        _amls_max_levels = (
            amls_max_levels if amls_max_levels is not None else 30
        )
        _amls_n_mcmc_steps = (
            amls_n_mcmc_steps if amls_n_mcmc_steps is not None else 10
        )
        _amls_mcmc_step_size = (
            amls_mcmc_step_size if amls_mcmc_step_size is not None else 0.3
        )
        _amls_n_samples_per_level = (
            amls_n_samples_per_level
            if amls_n_samples_per_level is not None
            else scenario_n_samples
        )
        amls_result = amls_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_samples_per_level=_amls_n_samples_per_level,
            quantile=_amls_quantile,
            max_levels=_amls_max_levels,
            n_mcmc_steps=_amls_n_mcmc_steps,
            mcmc_step_size=_amls_mcmc_step_size,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': amls_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'amls_result': amls_result,
            'amls_detected_unsafe': amls_result.detected_any,
            'amls_levels_used': max(
                (r.levels_used for grp in amls_result.per_hs_results for r in grp),
                default=0,
            ),
        }
    elif verification_method in ('amls_bounded', 'amls_bounded_union'):
        # Bounded AMLS: same level-splitting machinery, restricted to the
        # conformal latent ball ||z|| <= q. See
        # docs/research/2026-04-28-bounded-amls-design.md for the
        # soundness story. Verdict requires both (a) no detection in the
        # ball AND (b) the asymptotic upper bound pi_upper <=
        # eps_2_target so the joint multiplicative bound 1-(1-α)(1-ε_2)
        # is meaningful.
        #
        # ``amls_bounded_union`` runs ONE chain per group on
        # phi_union(y) = min_j phi_halfspace_j(y) instead of
        # len(group) per-halfspace chains. Equivalent for
        # single-halfspace groups (e.g. ACAS Xu); ~K× faster and
        # mathematically tighter for K-disjunct OR groups (cifar100
        # with K=99 other-class halfspaces, Exp 4 multi-class).
        from n2v.probabilistic.flow.amls_bounded import (
            amls_bounded_certify_spec,
            amls_bounded_certify_spec_union,
        )
        _amls_quantile = (
            amls_quantile if amls_quantile is not None else 0.1
        )
        _amls_max_levels = (
            amls_max_levels if amls_max_levels is not None else 30
        )
        _amls_n_mcmc_steps = (
            amls_n_mcmc_steps if amls_n_mcmc_steps is not None else 10
        )
        _amls_mcmc_step_size = (
            amls_mcmc_step_size if amls_mcmc_step_size is not None else 0.3
        )
        _amls_n_samples_per_level = (
            amls_n_samples_per_level
            if amls_n_samples_per_level is not None
            else scenario_n_samples
        )
        # Default eps_2 target = alpha so the joint multiplicative
        # bound becomes 1 - (1-α)(1-α) ≈ 2α.
        _eps_2_target = (
            amls_bounded_eps_2_target
            if amls_bounded_eps_2_target is not None else alpha
        )
        _certify_fn = (
            amls_bounded_certify_spec_union
            if verification_method == 'amls_bounded_union'
            else amls_bounded_certify_spec
        )
        amls_b_result = _certify_fn(
            flow_ode=flow,
            spec_groups=whitened_groups,
            q=q,
            eps_2_target=_eps_2_target,
            n_samples_per_level=_amls_n_samples_per_level,
            quantile=_amls_quantile,
            max_levels=_amls_max_levels,
            n_mcmc_steps=_amls_n_mcmc_steps,
            mcmc_step_size=_amls_mcmc_step_size,
            adaptive_step=amls_bounded_adaptive_step,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': amls_b_result.unsat_certified,
            'certifying_group_idx': None,
            # The (1-β_2) upper bound on the worst per-halfspace
            # rare-event probability. Drives the joint
            # multiplicative ε_total computation downstream.
            'epsilon_2': float(amls_b_result.eps_2_upper),
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'amls_bounded_result': amls_b_result,
            'amls_bounded_detected_unsafe': amls_b_result.detected_any,
            'amls_bounded_eps_2_upper': float(amls_b_result.eps_2_upper),
            'amls_bounded_eps_2_target': float(_eps_2_target),
            'amls_bounded_adaptive_step': amls_bounded_adaptive_step,
            'amls_bounded_mode': verification_method,
            'amls_levels_used': max(
                (r.levels_used
                 for grp in amls_b_result.per_hs_results for r in grp),
                default=0,
            ),
        }
    elif verification_method == 'derived':
        from n2v.probabilistic.flow.langevin_sampler import (
            langevin_certify_spec,
        )
        chain_count = 100
        step_total = max(1, int(scenario_n_samples) // chain_count)
        warmup = max(50, step_total // 4)
        sample_steps = max(50, step_total - warmup)
        lang_result = langevin_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_chains=chain_count,
            n_warmup=warmup,
            n_samples=sample_steps,
            step_size=0.05,
            lambda_tilt=5.0,
            use_mala=False,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': lang_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'derived_result': lang_result,
            'derived_detected_unsafe': lang_result.detected_any,
            'derived_min_final_phi': min(
                (r.final_phi for grp in lang_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
            'derived_mean_grad_norm': max(
                (r.mean_grad_norm for grp in lang_result.per_hs_results
                 for r in grp), default=0.0,
            ),
        }
    elif verification_method == 'raw_mc_uniform':
        # Brute-force baseline: one pass of uniform Monte Carlo on
        # ||z|| <= q, per-group union mass + Clopper-Pearson upper
        # bound. Same gate as amls_bounded_union (no detection AND
        # pi_upper <= eps_2_target) so it composes with the joint
        # 1 - (1-α)(1-ε_2) certificate downstream.
        from n2v.probabilistic.flow.raw_mc_uniform import (
            raw_mc_certify_spec,
        )
        _eps_2_target = (
            amls_bounded_eps_2_target
            if amls_bounded_eps_2_target is not None else alpha
        )
        raw_mc_result = raw_mc_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            q=q,
            eps_2_target=_eps_2_target,
            n_samples=scenario_n_samples,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': raw_mc_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': float(raw_mc_result.eps_2_upper),
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'raw_mc_result': raw_mc_result,
            'raw_mc_detected_unsafe': raw_mc_result.detected_any,
            'raw_mc_eps_2_upper': float(raw_mc_result.eps_2_upper),
            'raw_mc_eps_2_target': float(_eps_2_target),
        }
    elif verification_method == 'is_tilted':
        from n2v.probabilistic.flow.importance_sampling import (
            is_tilted_certify_spec,
        )
        is_result = is_tilted_certify_spec(
            flow_ode=flow,
            spec_groups=whitened_groups,
            n_samples=scenario_n_samples,
            lambda_tilt=5.0,
            beta=scenario_beta,
            seed=scenario_seed,
        )
        scenario_result = {
            'unsat_certified': is_result.unsat_certified,
            'certifying_group_idx': None,
            'epsilon_2': 0.0,
            'delta_2': 1.0 - scenario_beta,
            'n_samples_used': scenario_n_samples,
            'per_group_results': [],
            'spec_summary': spec_summary(spec),
            'is_result': is_result,
            'is_detected_unsafe': is_result.detected_any,
            'is_min_final_phi': min(
                (r.final_phi for grp in is_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
            'is_min_ess': min(
                (r.ess for grp in is_result.per_hs_results for r in grp),
                default=float('inf'),
            ),
        }
    else:
        raise ValueError(
            f'unsupported verification_method: {verification_method!r}')
    verify_time = time.time() - t1

    # 9. Joint (ε, δ) certificate.
    epsilon_2 = scenario_result['epsilon_2']
    delta_2 = scenario_result['delta_2']
    epsilon_total = 1.0 - (1.0 - epsilon_1) * (1.0 - epsilon_2)
    delta_total = delta_1 * delta_2

    # AMLS / IS detection invalidates UNSAT regardless of the
    # unsat_certified flag (a witness sample landed in U). Map detection
    # -> UNKNOWN; this mirrors the operational semantics described in
    # the AMLS / IS lit docs.
    if verification_method == 'amls' and scenario_result.get(
            'amls_detected_unsafe', False):
        verdict = 'UNKNOWN'
    elif verification_method == 'is_tilted' and scenario_result.get(
            'is_detected_unsafe', False):
        verdict = 'UNKNOWN'
    elif verification_method == 'derived' and scenario_result.get(
            'derived_detected_unsafe', False):
        verdict = 'UNKNOWN'
    else:
        verdict = 'UNSAT' if scenario_result['unsat_certified'] else 'UNKNOWN'

    # Wrap score_fn so external callers can still pass RAW network
    # outputs; whitening happens inside the wrapper.
    score_fn = _WhiteningFlowScore(base_score_fn, y_mean.cpu(), y_std.cpu())

    out = {
        'verdict': verdict,
        'epsilon_total': epsilon_total,
        'delta_total': delta_total,
        'epsilon_1': epsilon_1, 'delta_1': delta_1,
        'epsilon_2': epsilon_2, 'delta_2': delta_2,
        'counterexample': None,
        'counterexample_source': None,
        'flow_train_time_s': train_time,
        'flow_train_loss_final': flow_train_loss_final,
        'flow_train_loss_curve': (
            [float(x) for x in flow_losses] if flow_losses else []
        ),
        'verification_time_s': verify_time,
        'sat_backend_time_s': sat_backend_time_s,
        'total_time_s': train_time + verify_time + sat_backend_time_s,
        'coverage_empirical': coverage_empirical,
        'q': q,
        'flow': flow,
        'score_fn': score_fn,
        'y_mean': y_mean_np,
        'y_std': y_std_np,
        'scenario_result': scenario_result,
        'spec_summary': spec_summary(spec),
        'flow_ensemble_size': flow_ensemble_size,
    }
    if verification_method == 'amls':
        out['amls_detected_unsafe'] = scenario_result.get(
            'amls_detected_unsafe', False)
        out['amls_levels_used'] = scenario_result.get('amls_levels_used', 0)
    if verification_method in ('amls_bounded', 'amls_bounded_union'):
        out['amls_bounded_eps_2_upper'] = scenario_result.get(
            'amls_bounded_eps_2_upper', float('inf'))
        out['amls_bounded_detected_unsafe'] = scenario_result.get(
            'amls_bounded_detected_unsafe', False)
        out['amls_levels_used'] = scenario_result.get('amls_levels_used', 0)
    if verification_method == 'raw_mc_uniform':
        out['raw_mc_eps_2_upper'] = scenario_result.get(
            'raw_mc_eps_2_upper', float('inf'))
        out['raw_mc_detected_unsafe'] = scenario_result.get(
            'raw_mc_detected_unsafe', False)
    if verification_method == 'is_tilted':
        out['is_detected_unsafe'] = scenario_result.get(
            'is_detected_unsafe', False)
        out['is_min_final_phi'] = scenario_result.get(
            'is_min_final_phi', float('inf'))
        out['is_min_ess'] = scenario_result.get(
            'is_min_ess', float('inf'))
    if verification_method == 'derived':
        out['derived_detected_unsafe'] = scenario_result.get(
            'derived_detected_unsafe', False)
        out['derived_min_final_phi'] = scenario_result.get(
            'derived_min_final_phi', float('inf'))
        out['derived_mean_grad_norm'] = scenario_result.get(
            'derived_mean_grad_norm', 0.0)
    return out


# ---- Public entry point ---------------------------------------------


def run_verification_pipeline(
    network,
    input_lb: np.ndarray,
    input_ub: np.ndarray,
    spec,
    *,
    alpha: float = 0.001,
    m: int = 8000,
    ell: int = 7999,
    scenario_n_samples: int = 10_000,
    scenario_beta: float = 0.001,
    n_train: int = 10_000,
    flow_epochs: int = 5000,
    flow_config: str = 'tight',
    seed: int = 0,
    flow_seed: 'int | None' = None,
    cal_seed: 'int | None' = None,
    scenario_seed: 'int | None' = None,
    infer_solver: str = 'rk4',
    infer_steps: int = 30,
    adaptive_threshold: float = 0.5,
    adaptive_n_samples: int = 20000,
    sampling_strategy: str = 'uniform',
    flow_ensemble_size: int = 1,
    use_falsifier: bool = False,
    sat_backend: str = 'random+pgd+apgd',
    sat_backend_kwargs: dict | None = None,
    verification_method: str = 'scenario',
    # AMLS hyperparameter overrides (None = use legacy hardcoded defaults).
    amls_quantile: 'float | None' = None,
    amls_n_mcmc_steps: 'int | None' = None,
    amls_mcmc_step_size: 'float | None' = None,
    amls_n_samples_per_level: 'int | None' = None,
    amls_max_levels: 'int | None' = None,
    # Bounded-AMLS overrides (used only when verification_method=='amls_bounded').
    amls_bounded_eps_2_target: 'float | None' = None,
    amls_bounded_adaptive_step: bool = False,
    # Flow-training knob overrides (None = use legacy hardcoded defaults).
    flow_standardize: 'bool | None' = None,
    flow_coupling: 'str | None' = None,
    flow_use_ema: 'bool | None' = None,
) -> dict:
    """Flow-conformal+AMLS verification pipeline (Plan B library entry point).

    By default this entry point is **falsifier-free**: it skips Stage-1
    falsification entirely and runs flow training + scenario / AMLS
    verification, returning only UNSAT or UNKNOWN. Pass
    ``use_falsifier=True`` to opt back into a Stage-1 falsifier that can
    return SAT by finding a real counterexample.

    Pipeline stages:
      Stage 1 (opt-in via ``use_falsifier=True``): Run the falsifier on
        the raw network. If a counterexample is found and validated,
        return SAT immediately — flow training is skipped entirely.
      Stage 2 (always): Train the flow, calibrate, run scenario / AMLS
        verify. UNSAT is certified iff the verification step succeeds;
        otherwise UNKNOWN.

    ``flow_seed`` / ``cal_seed`` / ``scenario_seed`` (optional): override
    the seed for the flow-training, calibration-sampling, and
    scenario-verify stages independently. Each defaults to ``seed`` when
    unset, so existing callers passing only ``seed=k`` are unaffected.

    The returned dict's populated fields depend on `verdict`:
      - SAT: counterexample populated; flow/certificate fields None
      - UNSAT or UNKNOWN: certificate fields populated; counterexample None

    ``sampling_strategy`` (``'uniform'`` default, ``'qmc'`` optional)
    controls how latent samples are drawn for scenario-verify. Soundness
    note: under ``'qmc'`` samples come from the full N(0, I_d), not the
    conformal level-set ``||z|| <= q``, so the joint composition
    ``epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)`` (derived under
    'uniform') may not be tight. QMC is currently experimental; do not
    rely on the joint epsilon for sound certification under QMC.
    """
    # Validate verification_method up front. Phase A candidates extend
    # this tuple as they land (C0 'scenario_v2', C1 'amls',
    # C2 'is_tilted', C3 'derived').
    if verification_method not in (
            'scenario', 'scenario_v2', 'amls', 'amls_bounded',
            'amls_bounded_union', 'is_tilted', 'derived',
            'raw_mc_uniform'):
        raise ValueError(
            f'unsupported verification_method: {verification_method!r}')

    # --- Stage 1: Falsifier (opt-in; fast SAT path) ---
    sat_backend_time = 0.0
    if use_falsifier and sat_backend not in (None, 'none'):
        # Lazy import for spec_summary used in SAT-result construction.
        from examples.FlowConformal.benchmarks._spec import spec_summary

        t_sat = time.time()
        # Default falsifier budget: bumped from n2v.utils.falsify's
        # internal defaults. Caller can override via sat_backend_kwargs.
        default_kwargs = {'n_restarts': 30, 'n_steps': 200}
        fals_kwargs = {**default_kwargs, **(sat_backend_kwargs or {})}
        try:
            fals_result, fals_cex = falsify(
                model=network, lb=input_lb, ub=input_ub, property=spec,
                method=sat_backend, seed=seed, **fals_kwargs,
            )
        except Exception as e:
            fals_result, fals_cex = 2, None
            print(f'[run_verification_pipeline] falsify({sat_backend}) '
                  f'raised {type(e).__name__}: {e}', file=sys.stderr)
        sat_backend_time = time.time() - t_sat

        if fals_result == 0 and fals_cex is not None:
            cex_x, cex_y = fals_cex
            # Independent post-hoc validation: re-run real network and
            # verify the spec is satisfied at the real output.
            from n2v.utils.falsify import (
                _detect_model_device,
                _extract_halfspace_groups,
                _output_satisfies_property,
            )
            with torch.no_grad():
                _device = _detect_model_device(network)
                y_check = network(
                    torch.as_tensor(
                        np.asarray(cex_x).reshape(1, *input_lb.shape),
                        dtype=torch.float32,
                    ).to(_device)
                ).cpu().numpy().flatten()
            groups_for_check = _extract_halfspace_groups(spec)
            if _output_satisfies_property(y_check, groups_for_check):
                return _sat_result(
                    cex_x=np.asarray(cex_x).flatten(),
                    cex_y=y_check,
                    counterexample_source=sat_backend,
                    sat_backend_time_s=sat_backend_time,
                    spec_summary_str=spec_summary(spec),
                )
            print(f'[run_verification_pipeline] WARNING: falsifier '
                  f'returned a counterexample, but post-hoc spec check on '
                  f'the real network output disagreed. Ignoring it and '
                  f'proceeding to flow + scenario-verify.', file=sys.stderr)

    # --- Stage 2: Flow training + scenario / AMLS verify ---
    return _flow_unsat_pipeline(
        network=network,
        input_lb=input_lb, input_ub=input_ub, spec=spec,
        alpha=alpha, m=m, ell=ell,
        scenario_n_samples=scenario_n_samples, scenario_beta=scenario_beta,
        n_train=n_train, flow_epochs=flow_epochs, flow_config=flow_config,
        seed=seed,
        flow_seed=flow_seed, cal_seed=cal_seed, scenario_seed=scenario_seed,
        infer_solver=infer_solver, infer_steps=infer_steps,
        adaptive_threshold=adaptive_threshold,
        adaptive_n_samples=adaptive_n_samples,
        sampling_strategy=sampling_strategy,
        flow_ensemble_size=flow_ensemble_size,
        sat_backend_time_s=sat_backend_time,
        verification_method=verification_method,
        amls_quantile=amls_quantile,
        amls_n_mcmc_steps=amls_n_mcmc_steps,
        amls_mcmc_step_size=amls_mcmc_step_size,
        amls_n_samples_per_level=amls_n_samples_per_level,
        amls_max_levels=amls_max_levels,
        amls_bounded_eps_2_target=amls_bounded_eps_2_target,
        amls_bounded_adaptive_step=amls_bounded_adaptive_step,
        flow_standardize=flow_standardize,
        flow_coupling=flow_coupling,
        flow_use_ema=flow_use_ema,
    )


__all__ = [
    'run_verification_pipeline',
    # Internal helpers re-exported for backward-compat with the legacy
    # examples.FlowConformal.benchmarks._common shim and ablation scripts.
    '_flow_unsat_pipeline',
    '_certify_spec_on_flow_v2',
    '_extract_min_worst_max_margin',
    '_sat_result',
    '_train_flow',
    '_train_flow_tight',
    '_forward',
    '_WhitenedNetwork',
    '_WhiteningFlowScore',
    '_MaxEnsembleFlowScore',
    '_whiten_halfspace',
]
