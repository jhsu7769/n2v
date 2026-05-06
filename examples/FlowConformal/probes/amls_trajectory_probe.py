"""AMLS chain-trajectory probe (read-only diagnostic).

PURPOSE
-------
Re-runs the production pipeline on the metaroom false-UNSAT instances
and captures, *per AMLS level*, the diagnostic trajectory:

  - tau_k          : the level threshold at this iteration
  - phi_min        : the smallest phi value across the population
  - phi_q_quantile : the q-quantile of phi values (= the unclamped tau)
  - was_clamped    : True if the level used the ``tau_prev - 1e-12``
                     fallback (the chain's quantile didn't decrease)
  - frac_in_U      : fraction of population with phi <= 0 at this level
  - mh_accept_rate : MCMC acceptance rate during this level's mutation

What we're checking: when AMLS exhausts ``max_levels=30`` without
detection, does the chain genuinely descend (real geometric progress)
or does it stall (forced tau decrease via the 1e-12 fallback)? Stalling
silently invalidates the ``pi_upper = quantile^K`` extrapolation.

DESIGN
------
This probe MUST NOT modify any production code. To capture per-level
state, we monkey-patch ``amls_bounded_estimate_halfspace_mass`` for
the duration of the probe with an instrumented copy that records the
trajectory to a global list. The instrumented function is a near-clone
of the production function with logging inlined; it produces the same
result (verified post-hoc by the run reproducing the production
``pi_upper`` and verdict).

Output: ``examples/FlowConformal/probes/outputs/amls_trajectory_<benchmark>_<vnnlib>.csv``
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
import n2v.probabilistic.flow.amls_bounded as amls_mod
from n2v.probabilistic.flow.amls import (
    _phi_halfspace_torch,
    _push_through_flow,
    _resolve_device,
)
from n2v.probabilistic.flow.amls_bounded import (
    AMLSBoundedResult,
    _phi_union_torch,
)
from n2v.probabilistic.flow.scenario_verify import (
    sample_truncated_gaussian_ball,
)
from n2v.probabilistic.verify_flow import run_verification_pipeline

_OUT_DIR = Path(__file__).resolve().parent / 'outputs'
_TRAJECTORY: List[Dict[str, Any]] = []
_CALL_INDEX = 0
# Probe-level override: when True, the instrumented union function
# forces ``adaptive_step=True`` regardless of what the pipeline passes.
# Used to test whether the Roberts-Tweedie scaling fixes the MCMC
# acceptance-rate collapse seen in metaroom idx 14.
_FORCE_ADAPTIVE_STEP = False
# Probe-level override: when not None, force ``mcmc_step_size`` to this
# value (the production default is 0.3). Used to test whether shrinking
# the proposal scale fixes the late-level acceptance-rate collapse.
_FORCE_STEP_SIZE: Optional[float] = None
# Probe-level override: when not None, force ``n_mcmc_steps`` to this
# value (production default 10). More steps give the chain more chances
# to mix into the new level set before the next quantile cut.
_FORCE_N_MCMC_STEPS: Optional[int] = None


def instrumented_estimate(
    flow_ode,
    halfspace,
    *,
    q: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedResult:
    """Instrumented clone of ``amls_bounded_estimate_halfspace_mass``.

    Identical algorithm to the production function — same RNG, same
    branches, same output. Adds a per-level trajectory log into the
    module-global ``_TRAJECTORY`` list, tagged with the current
    ``_CALL_INDEX`` so multiple per-instance halfspaces are
    distinguishable.
    """
    global _CALL_INDEX
    call_idx = _CALL_INDEX
    _CALL_INDEX += 1

    if q <= 0:
        raise ValueError(f'q must be positive, got {q}')

    dev = _resolve_device(device)
    G_np = np.asarray(halfspace.G, dtype=np.float64)
    g_np = np.asarray(halfspace.g, dtype=np.float64).flatten()
    dim = G_np.shape[1]
    N = n_samples_per_level
    dtype = torch.float32

    G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
    g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)

    eff_step = (mcmc_step_size * min(1.0, q / math.sqrt(max(dim, 1)))
                if adaptive_step else mcmc_step_size)
    q_sq_t = torch.tensor(q * q, device=dev, dtype=dtype)

    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    gen = torch.Generator(device=dev).manual_seed(seed_int)
    if seed is not None:
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)

    z_np = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=N)
    z = torch.as_tensor(z_np.astype(np.float32), device=dev, dtype=dtype)

    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_halfspace_torch(y, G_t, g_t)

    best_idx_t = torch.argmin(phi_t)
    best_phi_t = phi_t[best_idx_t].detach().clone()
    best_y_t = y[best_idx_t].detach().clone()

    # Log level 0
    _TRAJECTORY.append({
        'call_idx': call_idx,
        'level': 0,
        'tau_k': float('inf'),
        'phi_min': float(phi_t.min().item()),
        'phi_q_quantile': float(torch.quantile(phi_t, quantile).item()),
        'phi_max': float(phi_t.max().item()),
        'phi_mean': float(phi_t.mean().item()),
        'was_clamped': False,
        'frac_in_U': float((phi_t <= 0.0).float().mean().item()),
        'mh_accept_rate': float('nan'),
        'event': 'init',
    })

    in_U_mask = phi_t <= 0.0
    if bool(in_U_mask.any().item()):
        n_in = int(in_U_mask.sum().item())
        pi_hat = n_in / N
        from scipy.stats import beta as _beta_dist
        if n_in == N:
            pi_upper = 1.0
        else:
            pi_upper = float(_beta_dist.ppf(1.0 - beta, n_in + 1, N - n_in))
        _TRAJECTORY[-1]['event'] = 'detect_at_init'
        return AMLSBoundedResult(
            pi_hat=pi_hat,
            pi_upper=pi_upper,
            levels_used=0,
            final_unsafe_count=n_in,
            detected_unsafe=True,
            final_phi=float(best_phi_t.item()),
            worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
            adaptive_step_used=adaptive_step,
        )

    tau_prev = math.inf
    K = 0
    for level in range(max_levels):
        K = level + 1

        tau_k_unclamped = float(torch.quantile(phi_t, quantile).item())
        tau_k = tau_k_unclamped
        was_clamped = False
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
            was_clamped = True

        tau_k_t = torch.tensor(tau_k, device=dev, dtype=dtype)

        if tau_k <= 0.0:
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** (K - 1)) * frac_in_U
            from scipy.stats import norm
            sigma2 = K * (1.0 - quantile) / (quantile * N)
            log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                            norm.ppf(1.0 - beta) * math.sqrt(sigma2))
            pi_upper = math.exp(log_pi_upper)
            _TRAJECTORY.append({
                'call_idx': call_idx, 'level': K,
                'tau_k': tau_k, 'phi_min': float(phi_t.min().item()),
                'phi_q_quantile': tau_k_unclamped,
                'phi_max': float(phi_t.max().item()),
                'phi_mean': float(phi_t.mean().item()),
                'was_clamped': was_clamped, 'frac_in_U': frac_in_U,
                'mh_accept_rate': float('nan'),
                'event': 'tau_dropped_to_zero',
            })
            best_idx_t = torch.argmin(phi_t)
            best_phi_now = float(phi_t[best_idx_t].item())
            if best_phi_now < float(best_phi_t.item()):
                best_phi_t = phi_t[best_idx_t].detach().clone()
                best_y_t = y[best_idx_t].detach().clone()
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=bool(in_U.any().item()),
                final_phi=float(best_phi_t.item()),
                worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
                adaptive_step_used=adaptive_step,
            )

        keep_count = max(1, int(round(N * quantile)))
        _kept_phi, keep_idx_t = torch.topk(
            phi_t, k=keep_count, largest=False, sorted=False,
        )

        sample_indices = torch.randint(
            0, keep_count, (N,), device=dev, generator=gen,
        )
        chosen_t = keep_idx_t[sample_indices]
        z_cur = z.index_select(0, chosen_t).clone()
        y_cur = y.index_select(0, chosen_t).clone()
        phi_cur = phi_t.index_select(0, chosen_t).clone()

        n_accept_total = 0
        n_proposals_total = 0
        for _step in range(n_mcmc_steps):
            eta = torch.randn(N, dim, device=dev, dtype=dtype, generator=gen)
            z_prop = z_cur + eff_step * eta
            z_prop_norm_sq = (z_prop * z_prop).sum(dim=1)
            in_ball = z_prop_norm_sq <= q_sq_t
            log_alpha = 0.5 * (
                (z_cur * z_cur).sum(dim=1) - z_prop_norm_sq
            )
            u = torch.rand(N, device=dev, dtype=dtype, generator=gen)
            log_u = torch.log(u)
            mh_pass = log_u < log_alpha
            y_prop = _push_through_flow(
                flow_ode, z_prop, t=t, n_steps=n_ode_steps,
                method=ode_method, atol=ode_atol, rtol=ode_rtol,
            )
            phi_prop = _phi_halfspace_torch(y_prop, G_t, g_t)
            level_pass = phi_prop <= tau_k_t
            accept = mh_pass & level_pass & in_ball

            cur_best_idx = torch.argmin(phi_prop)
            cur_best_phi = phi_prop[cur_best_idx]
            improve = cur_best_phi < best_phi_t
            best_phi_t = torch.where(improve, cur_best_phi, best_phi_t)
            best_y_t = torch.where(
                improve.unsqueeze(-1), y_prop[cur_best_idx], best_y_t,
            )

            accept_b = accept.unsqueeze(-1)
            z_cur = torch.where(accept_b, z_prop, z_cur)
            y_cur = torch.where(accept_b, y_prop, y_cur)
            phi_cur = torch.where(accept, phi_prop, phi_cur)

            n_accept_total += int(accept.sum().item())
            n_proposals_total += N

        z = z_cur
        y = y_cur
        phi_t = phi_cur
        tau_prev = tau_k

        # Log this level
        _TRAJECTORY.append({
            'call_idx': call_idx, 'level': K,
            'tau_k': tau_k,
            'phi_min': float(phi_t.min().item()),
            'phi_q_quantile': tau_k_unclamped,
            'phi_max': float(phi_t.max().item()),
            'phi_mean': float(phi_t.mean().item()),
            'was_clamped': was_clamped,
            'frac_in_U': float((phi_t <= 0.0).float().mean().item()),
            'mh_accept_rate': (n_accept_total / max(n_proposals_total, 1)),
            'event': 'level_complete',
        })

        if bool((phi_t <= 0.0).any().item()):
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** K) * frac_in_U
            from scipy.stats import norm
            sigma2 = (K + 1) * (1.0 - quantile) / (quantile * N)
            log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                            norm.ppf(1.0 - beta) * math.sqrt(sigma2))
            pi_upper = math.exp(log_pi_upper)
            _TRAJECTORY[-1]['event'] = 'detect_at_level_end'
            return AMLSBoundedResult(
                pi_hat=pi_hat,
                pi_upper=min(1.0, pi_upper),
                levels_used=K + 1,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=True,
                final_phi=float(best_phi_t.item()),
                worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
                adaptive_step_used=adaptive_step,
            )

    # Exhausted
    pi_hat = quantile ** K
    from scipy.stats import norm
    sigma2 = K * (1.0 - quantile) / (quantile * N)
    log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                    norm.ppf(1.0 - beta) * math.sqrt(sigma2))
    pi_upper = min(1.0, math.exp(log_pi_upper))
    _TRAJECTORY[-1]['event'] = 'exhausted_no_detection'
    return AMLSBoundedResult(
        pi_hat=pi_hat,
        pi_upper=pi_upper,
        levels_used=K,
        final_unsafe_count=0,
        detected_unsafe=bool(float(best_phi_t.item()) <= 0.0),
        final_phi=float(best_phi_t.item()),
        worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
        adaptive_step_used=adaptive_step,
    )


def instrumented_estimate_union(
    flow_ode,
    halfspaces: List,
    *,
    q: float,
    n_samples_per_level: int = 2000,
    quantile: float = 0.1,
    max_levels: int = 30,
    n_mcmc_steps: int = 10,
    mcmc_step_size: float = 0.3,
    adaptive_step: bool = False,
    beta: float = 0.001,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: Optional[Union[str, torch.device]] = None,
) -> AMLSBoundedResult:
    """Instrumented clone of ``amls_bounded_estimate_union_mass``.

    Identical algorithm to the production function — chain targets
    ``phi_union(y) = min_j phi_halfspace(y, G_j, g_j)`` so a single
    AMLS chain estimates the union mass over all halfspaces.
    Logs per-level trajectory into ``_TRAJECTORY``.
    """
    global _CALL_INDEX
    call_idx = _CALL_INDEX
    _CALL_INDEX += 1

    # Probe overrides: force adaptive_step / mcmc_step_size /
    # n_mcmc_steps when the operator set them on the CLI. Let us A/B
    # test MCMC tuning against the production defaults without touching
    # the pipeline code path.
    if _FORCE_ADAPTIVE_STEP:
        adaptive_step = True
    if _FORCE_STEP_SIZE is not None:
        mcmc_step_size = _FORCE_STEP_SIZE
    if _FORCE_N_MCMC_STEPS is not None:
        n_mcmc_steps = _FORCE_N_MCMC_STEPS

    if not halfspaces:
        raise ValueError('halfspaces must be non-empty')
    if q <= 0:
        raise ValueError(f'q must be positive, got {q}')

    dev = _resolve_device(device)
    dtype = torch.float32

    halfspaces_torch = []
    dim = None
    for hs in halfspaces:
        G_np = np.asarray(hs.G, dtype=np.float64)
        g_np = np.asarray(hs.g, dtype=np.float64).flatten()
        if dim is None:
            dim = G_np.shape[1]
        G_t = torch.from_numpy(G_np.astype(np.float32)).to(dev)
        g_t = torch.from_numpy(g_np.astype(np.float32)).to(dev)
        halfspaces_torch.append((G_t, g_t))

    N = n_samples_per_level
    eff_step = (mcmc_step_size * min(1.0, q / math.sqrt(max(dim, 1)))
                if adaptive_step else mcmc_step_size)
    q_sq_t = torch.tensor(q * q, device=dev, dtype=dtype)

    if hasattr(flow_ode, 'to'):
        flow_ode = flow_ode.to(dev)

    seed_int = 0 if seed is None else (int(seed) & 0x7FFFFFFF)
    gen = torch.Generator(device=dev).manual_seed(seed_int)
    if seed is not None:
        torch.manual_seed(seed_int)
        np.random.seed(seed_int)

    z_np = sample_truncated_gaussian_ball(q=q, dim=dim, n_samples=N)
    z = torch.as_tensor(z_np.astype(np.float32), device=dev, dtype=dtype)

    y = _push_through_flow(
        flow_ode, z, t=t, n_steps=n_ode_steps,
        method=ode_method, atol=ode_atol, rtol=ode_rtol,
    )
    phi_t = _phi_union_torch(y, halfspaces_torch)

    best_idx_t = torch.argmin(phi_t)
    best_phi_t = phi_t[best_idx_t].detach().clone()
    best_y_t = y[best_idx_t].detach().clone()

    _TRAJECTORY.append({
        'call_idx': call_idx, 'level': 0,
        'tau_k': float('inf'),
        'phi_min': float(phi_t.min().item()),
        'phi_q_quantile': float(torch.quantile(phi_t, quantile).item()),
        'phi_max': float(phi_t.max().item()),
        'phi_mean': float(phi_t.mean().item()),
        'was_clamped': False,
        'frac_in_U': float((phi_t <= 0.0).float().mean().item()),
        'mh_accept_rate': float('nan'),
        'event': 'init',
    })

    in_U_mask = phi_t <= 0.0
    if bool(in_U_mask.any().item()):
        n_in = int(in_U_mask.sum().item())
        pi_hat = n_in / N
        from scipy.stats import beta as _beta_dist
        pi_upper = (1.0 if n_in == N
                    else float(_beta_dist.ppf(1.0 - beta, n_in + 1, N - n_in)))
        _TRAJECTORY[-1]['event'] = 'detect_at_init'
        return AMLSBoundedResult(
            pi_hat=pi_hat, pi_upper=pi_upper, levels_used=1,
            final_unsafe_count=n_in, detected_unsafe=True,
            final_phi=float(best_phi_t.item()),
            worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
            adaptive_step_used=adaptive_step,
        )

    tau_prev = math.inf
    K = 0
    for level in range(max_levels):
        K = level + 1

        tau_k_unclamped = float(torch.quantile(phi_t, quantile).item())
        tau_k = tau_k_unclamped
        was_clamped = False
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
            was_clamped = True
        tau_k_dev = torch.tensor(tau_k, device=dev, dtype=dtype)

        if tau_k <= 0.0:
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** (K - 1)) * frac_in_U
            from scipy.stats import norm
            sigma2 = K * (1.0 - quantile) / (quantile * N)
            log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                            norm.ppf(1.0 - beta) * math.sqrt(sigma2))
            pi_upper = math.exp(log_pi_upper)
            _TRAJECTORY.append({
                'call_idx': call_idx, 'level': K, 'tau_k': tau_k,
                'phi_min': float(phi_t.min().item()),
                'phi_q_quantile': tau_k_unclamped,
                'phi_max': float(phi_t.max().item()),
                'phi_mean': float(phi_t.mean().item()),
                'was_clamped': was_clamped, 'frac_in_U': frac_in_U,
                'mh_accept_rate': float('nan'),
                'event': 'tau_dropped_to_zero',
            })
            best_idx_t = torch.argmin(phi_t)
            best_phi_now = float(phi_t[best_idx_t].item())
            if best_phi_now < float(best_phi_t.item()):
                best_phi_t = phi_t[best_idx_t].detach().clone()
                best_y_t = y[best_idx_t].detach().clone()
            return AMLSBoundedResult(
                pi_hat=pi_hat, pi_upper=min(1.0, pi_upper), levels_used=K,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=bool(in_U.any().item()),
                final_phi=float(best_phi_t.item()),
                worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
                adaptive_step_used=adaptive_step,
            )

        keep_count = max(1, int(round(N * quantile)))
        _kept_phi, keep_idx_t = torch.topk(
            phi_t, k=keep_count, largest=False, sorted=False,
        )
        sample_indices = torch.randint(
            0, keep_count, (N,), device=dev, generator=gen,
        )
        chosen_t = keep_idx_t[sample_indices]
        z_cur = z.index_select(0, chosen_t).clone()
        y_cur = y.index_select(0, chosen_t).clone()
        phi_cur = phi_t.index_select(0, chosen_t).clone()

        n_accept_total = 0
        n_proposals_total = 0
        for _step in range(n_mcmc_steps):
            eta = torch.randn(N, dim, device=dev, dtype=dtype, generator=gen)
            z_prop = z_cur + eff_step * eta
            z_prop_norm_sq = (z_prop * z_prop).sum(dim=1)
            in_ball = z_prop_norm_sq <= q_sq_t
            log_alpha = 0.5 * (
                (z_cur * z_cur).sum(dim=1) - z_prop_norm_sq
            )
            u = torch.rand(N, device=dev, dtype=dtype, generator=gen)
            log_u = torch.log(u)
            mh_pass = log_u < log_alpha
            y_prop = _push_through_flow(
                flow_ode, z_prop, t=t, n_steps=n_ode_steps,
                method=ode_method, atol=ode_atol, rtol=ode_rtol,
            )
            phi_prop = _phi_union_torch(y_prop, halfspaces_torch)
            level_pass = phi_prop <= tau_k_dev
            accept = mh_pass & level_pass & in_ball

            cur_best_idx = torch.argmin(phi_prop)
            cur_best_phi = phi_prop[cur_best_idx]
            improve = cur_best_phi < best_phi_t
            best_phi_t = torch.where(improve, cur_best_phi, best_phi_t)
            best_y_t = torch.where(
                improve.unsqueeze(-1), y_prop[cur_best_idx], best_y_t,
            )
            accept_b = accept.unsqueeze(-1)
            z_cur = torch.where(accept_b, z_prop, z_cur)
            y_cur = torch.where(accept_b, y_prop, y_cur)
            phi_cur = torch.where(accept, phi_prop, phi_cur)

            n_accept_total += int(accept.sum().item())
            n_proposals_total += N

        z = z_cur
        y = y_cur
        phi_t = phi_cur
        tau_prev = tau_k

        _TRAJECTORY.append({
            'call_idx': call_idx, 'level': K, 'tau_k': tau_k,
            'phi_min': float(phi_t.min().item()),
            'phi_q_quantile': tau_k_unclamped,
            'phi_max': float(phi_t.max().item()),
            'phi_mean': float(phi_t.mean().item()),
            'was_clamped': was_clamped,
            'frac_in_U': float((phi_t <= 0.0).float().mean().item()),
            'mh_accept_rate': (n_accept_total / max(n_proposals_total, 1)),
            'event': 'level_complete',
        })

        if bool((phi_t <= 0.0).any().item()):
            in_U = phi_t <= 0.0
            frac_in_U = float(in_U.float().mean().item())
            pi_hat = (quantile ** K) * frac_in_U
            from scipy.stats import norm
            sigma2 = (K + 1) * (1.0 - quantile) / (quantile * N)
            log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                            norm.ppf(1.0 - beta) * math.sqrt(sigma2))
            pi_upper = math.exp(log_pi_upper)
            _TRAJECTORY[-1]['event'] = 'detect_at_level_end'
            return AMLSBoundedResult(
                pi_hat=pi_hat, pi_upper=min(1.0, pi_upper),
                levels_used=K + 1,
                final_unsafe_count=int(in_U.sum().item()),
                detected_unsafe=True,
                final_phi=float(best_phi_t.item()),
                worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
                adaptive_step_used=adaptive_step,
            )

    pi_hat = quantile ** K
    from scipy.stats import norm
    sigma2 = K * (1.0 - quantile) / (quantile * N)
    log_pi_upper = (math.log(max(pi_hat, 1e-300)) +
                    norm.ppf(1.0 - beta) * math.sqrt(sigma2))
    pi_upper = min(1.0, math.exp(log_pi_upper))
    _TRAJECTORY[-1]['event'] = 'exhausted_no_detection'
    return AMLSBoundedResult(
        pi_hat=pi_hat, pi_upper=pi_upper, levels_used=K,
        final_unsafe_count=0,
        detected_unsafe=bool(float(best_phi_t.item()) <= 0.0),
        final_phi=float(best_phi_t.item()),
        worst_y=best_y_t.detach().cpu().numpy().astype(np.float64),
        adaptive_step_used=adaptive_step,
    )


def run_probe(benchmark: str, instance_idx: int, output_csv: Path) -> None:
    """Run the probe on one instance, capture per-level AMLS trajectory."""
    global _TRAJECTORY, _CALL_INDEX
    _TRAJECTORY = []
    _CALL_INDEX = 0

    instances = list_instances(benchmark)
    if not (0 <= instance_idx < len(instances)):
        raise IndexError(
            f'instance_idx {instance_idx} out of [0, {len(instances)})')
    onnx_rel, vnn_rel, vnncomp_t = instances[instance_idx]
    print(f'[probe] {benchmark} idx={instance_idx}: {vnn_rel} (t={vnncomp_t}s)',
          flush=True)

    network, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[benchmark]

    # Monkey-patch BOTH production functions for the duration of this call:
    # ``_halfspace_mass`` (used by ``amls_bounded_certify_spec`` — per-HS
    # path) and ``_union_mass`` (used by ``amls_bounded_certify_spec_union``
    # — single chain on phi_min, used by classification benchmarks like
    # metaroom). Patching only the halfspace path silently misses union
    # benchmarks (probe captures 0 rows).
    orig_hs = amls_mod.amls_bounded_estimate_halfspace_mass
    orig_union = amls_mod.amls_bounded_estimate_union_mass
    amls_mod.amls_bounded_estimate_halfspace_mass = instrumented_estimate
    amls_mod.amls_bounded_estimate_union_mass = instrumented_estimate_union
    try:
        # We only use the FIRST input box (single-box benchmarks).
        lb, ub = boxes[0]
        result = run_verification_pipeline(
            network=network, input_lb=lb, input_ub=ub, spec=spec,
            alpha=cfg['alpha'],
            n_train=cfg['n_train'],
            flow_epochs=cfg['flow_epochs'],
            flow_config=cfg['flow_config'],
            scenario_n_samples=cfg['scenario_n_samples'],
            scenario_beta=0.001,
            verification_method=cfg['verification_method'],
            amls_max_levels=cfg['amls_max_levels'],
            seed=47,
            use_falsifier=False,  # we want to see AMLS behavior
        )
    finally:
        amls_mod.amls_bounded_estimate_halfspace_mass = orig_hs
        amls_mod.amls_bounded_estimate_union_mass = orig_union

    print(f'[probe] verdict={result["verdict"]}  '
          f'eps_2={result.get("amls_bounded_eps_2_upper")}  '
          f'detected={result.get("amls_bounded_detected_unsafe")}  '
          f'levels={result.get("amls_levels_used")}',
          flush=True)
    print(f'[probe] captured {len(_TRAJECTORY)} trajectory rows from '
          f'{_CALL_INDEX} per-halfspace AMLS calls', flush=True)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['benchmark', 'instance_idx', 'vnn_rel', 'verdict', 'call_idx',
              'level', 'event', 'tau_k', 'phi_q_quantile',
              'phi_min', 'phi_mean', 'phi_max', 'was_clamped',
              'frac_in_U', 'mh_accept_rate']
    with open(output_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in _TRAJECTORY:
            w.writerow({
                'benchmark': benchmark, 'instance_idx': instance_idx,
                'vnn_rel': vnn_rel, 'verdict': result['verdict'],
                **row,
            })
    print(f'[probe] wrote {output_csv}')

    # Print a per-call summary so the operator can scan terminal output.
    print('\n[probe] per-call summary:')
    by_call: Dict[int, List[Dict[str, Any]]] = {}
    for row in _TRAJECTORY:
        by_call.setdefault(row['call_idx'], []).append(row)
    for ci, rows in sorted(by_call.items()):
        n_levels = max(r['level'] for r in rows)
        n_clamped = sum(1 for r in rows if r['was_clamped'])
        first = rows[0]
        last = rows[-1]
        print(f'  call_idx={ci:>3}  levels={n_levels:>2}  '
              f'clamped={n_clamped:>2}/{n_levels}  '
              f'phi_min: {first["phi_min"]:>10.4g} -> {last["phi_min"]:>10.4g}  '
              f'final event={last["event"]}')


def main():
    global _FORCE_ADAPTIVE_STEP, _FORCE_STEP_SIZE, _FORCE_N_MCMC_STEPS
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--output-csv', type=Path, default=None)
    p.add_argument('--adaptive-step', action='store_true',
                   help='Force adaptive_step=True (Roberts-Tweedie scaling).')
    p.add_argument('--mcmc-step-size', type=float, default=None,
                   help='Override mcmc_step_size (default 0.3).')
    p.add_argument('--n-mcmc-steps', type=int, default=None,
                   help='Override n_mcmc_steps (default 10).')
    p.add_argument('--tag', type=str, default=None,
                   help='Optional output filename suffix tag.')
    args = p.parse_args()

    _FORCE_ADAPTIVE_STEP = bool(args.adaptive_step)
    _FORCE_STEP_SIZE = args.mcmc_step_size
    _FORCE_N_MCMC_STEPS = args.n_mcmc_steps

    suffix_parts = []
    if args.adaptive_step:
        suffix_parts.append('adaptive')
    if args.mcmc_step_size is not None:
        suffix_parts.append(f'step{args.mcmc_step_size:g}')
    if args.n_mcmc_steps is not None:
        suffix_parts.append(f'nsteps{args.n_mcmc_steps}')
    if args.tag:
        suffix_parts.append(args.tag)
    suffix = ('_' + '_'.join(suffix_parts)) if suffix_parts else ''

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR /
                    f'amls_trajectory_{args.benchmark}_inst{args.instance_idx}{suffix}.csv')
    print(f'[probe] FORCE_ADAPTIVE_STEP={_FORCE_ADAPTIVE_STEP}  '
          f'FORCE_STEP_SIZE={_FORCE_STEP_SIZE}  '
          f'FORCE_N_MCMC_STEPS={_FORCE_N_MCMC_STEPS}', flush=True)
    run_probe(args.benchmark, args.instance_idx, out_csv)


if __name__ == '__main__':
    main()
