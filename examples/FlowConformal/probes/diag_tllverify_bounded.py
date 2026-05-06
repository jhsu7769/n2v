"""Smoke test for bounded-AMLS on the tllverify instance that
unbounded AMLS over-rejects. Compares scenario, AMLS unbounded, and
AMLS bounded (both with and without adaptive step) head-to-head on
the same trained flow + calibrated q.

Expected outcome: scenario and amls_bounded both UNSAT in O(1)s; amls
unbounded "detects" with pi_hat ~ 1e-12 and is treated as UNKNOWN.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ_ROOT))

from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (  # noqa: E402
    load_instance,
)
from n2v.probabilistic.verify_flow import _train_flow  # noqa: E402
from n2v.probabilistic.flow.scores import FlowScore  # noqa: E402
from n2v.probabilistic.flow.calibrate import calibrate  # noqa: E402
from n2v.probabilistic.flow.scenario_verify import (  # noqa: E402
    certify_halfspace_disjoint,
)
from n2v.probabilistic.flow.amls import amls_certify_spec  # noqa: E402
from n2v.probabilistic.flow.amls_bounded import (  # noqa: E402
    amls_bounded_certify_spec,
)


def main():
    bench_root = Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/tllverifybench_2023'
    ))
    onnx_rel = 'onnx/tllBench_n=2_N=M=8_m=1_instance_0_1.onnx'
    vnn_rel = 'vnnlib/property_N=8_1.vnnlib'

    net, boxes, spec = load_instance(bench_root, onnx_rel, vnn_rel)
    lb, ub = boxes[0]
    lb_t = torch.tensor(np.asarray(lb).flatten(), dtype=torch.float32)
    ub_t = torch.tensor(np.asarray(ub).flatten(), dtype=torch.float32)
    seed = 0
    torch.manual_seed(seed)

    n_train = 10000
    m_calib = 8000
    x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
    x_ca = lb_t + torch.rand(m_calib, lb_t.shape[0]) * (ub_t - lb_t)
    with torch.no_grad():
        y_tr = net(x_tr).detach()
        y_ca = net(x_ca).detach()
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std

    print(f'[diag] training flow on {n_train} points (epochs=2000)...')
    t0 = time.time()
    flow, _ = _train_flow(
        y_tr_w, dim=y_tr_w.shape[1], n_epochs=2000, seed=seed,
        return_losses=True,
    )
    flow = flow.to('cpu').eval()
    print(f'[diag]   trained in {time.time()-t0:.1f}s')

    score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4', batch_size=65536)
    with torch.no_grad():
        calib_scores = score_fn(y_ca_w)
    alpha = 0.001
    import math
    ell = max(1, min(m_calib, int(math.ceil((m_calib + 1) * (1 - alpha)))))
    q = calibrate(calib_scores, ell).item()
    print(f'[diag] calibrated q = {q:.4f}')

    if isinstance(spec, list):
        s = spec[0]
    else:
        s = spec
    if isinstance(s, dict):
        hs = s.get('Hg')
        if isinstance(hs, list): hs = hs[0]
        s = hs
    G = np.asarray(s.G, dtype=np.float64)
    gv = np.asarray(s.g, dtype=np.float64).flatten()
    mu = y_mean.numpy()
    sigma = y_std.numpy()
    G_w = (G * sigma).astype(np.float64)
    g_w = (gv - (G @ mu).astype(np.float64)).astype(np.float64)

    from n2v.sets.halfspace import HalfSpace
    hs_w = HalfSpace(G_w.reshape(1, -1), g_w.reshape(-1, 1))
    spec_groups = [[hs_w]]

    print()
    print('=== HEAD-TO-HEAD ON tllverify property_N=8_1 (αβ-CROWN: unsat) ===')

    print()
    print('--- 1) scenario (truncated Gaussian on ||z|| <= q) ---')
    for n_scn in (2000, 8000):
        t0 = time.time()
        r = certify_halfspace_disjoint(
            flow_ode=flow, threshold_q=q, halfspace=hs_w,
            n_samples=n_scn, beta_2=0.001, seed=seed,
        )
        print(f'   N={n_scn:>5d}: disjoint={r.disjoint}  '
              f'max_margin={r.worst_max_margin:+.4f}  wall={time.time()-t0:.2f}s')

    print()
    print('--- 2) amls (unbounded, current production) ---')
    for n_amls in (2000, 8000):
        t0 = time.time()
        r = amls_certify_spec(
            flow_ode=flow, spec_groups=spec_groups,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, beta=0.001, seed=seed,
        )
        first = r.per_hs_results[0][0]
        print(f'   N={n_amls:>5d}: detected={first.detected_unsafe}  '
              f'levels={first.levels_used}  '
              f'final_phi={first.final_phi:+.4f}  '
              f'pi_hat={first.pi_hat:.2e}  '
              f'pi_upper={first.pi_upper:.2e}  '
              f'wall={time.time()-t0:.2f}s')

    print()
    print('--- 3) amls_bounded (constrained to ||z|| <= q, fixed step) ---')
    for n_amls in (2000, 8000):
        t0 = time.time()
        r = amls_bounded_certify_spec(
            flow_ode=flow, spec_groups=spec_groups,
            q=q, eps_2_target=alpha,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, mcmc_step_size=0.3,
            adaptive_step=False,
            beta=0.001, seed=seed,
        )
        first = r.per_hs_results[0][0]
        print(f'   N={n_amls:>5d}: unsat={r.unsat_certified}  '
              f'detected={first.detected_unsafe}  '
              f'levels={first.levels_used}  '
              f'final_phi={first.final_phi:+.4f}  '
              f'pi_hat={first.pi_hat:.2e}  '
              f'pi_upper={first.pi_upper:.2e}  '
              f'wall={time.time()-t0:.2f}s')

    print()
    print('--- 4) amls_bounded (adaptive step = q/sqrt(d)) ---')
    for n_amls in (2000, 8000):
        t0 = time.time()
        r = amls_bounded_certify_spec(
            flow_ode=flow, spec_groups=spec_groups,
            q=q, eps_2_target=alpha,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, mcmc_step_size=0.3,
            adaptive_step=True,
            beta=0.001, seed=seed,
        )
        first = r.per_hs_results[0][0]
        print(f'   N={n_amls:>5d}: unsat={r.unsat_certified}  '
              f'detected={first.detected_unsafe}  '
              f'levels={first.levels_used}  '
              f'final_phi={first.final_phi:+.4f}  '
              f'pi_hat={first.pi_hat:.2e}  '
              f'pi_upper={first.pi_upper:.2e}  '
              f'adaptive_used={first.adaptive_step_used}  '
              f'wall={time.time()-t0:.2f}s')

    print()
    print('=== INTERPRETATION ===')
    print('  scenario:                  expect disjoint=True (UNSAT)')
    print('  amls (unbounded):          expect detected=True (UNKNOWN — wrong)')
    print('  amls_bounded (fixed):      expect unsat=True with pi_upper << alpha (UNSAT — correct)')
    print('  amls_bounded (adaptive):   expect unsat=True (UNSAT — correct, comparable wall)')


if __name__ == '__main__':
    main()
