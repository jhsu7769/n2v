"""Run scenario-disjoint (truncated Gaussian on ||z|| <= q) directly on
the tllverify instance that AMLS over-rejects. If scenario certifies
UNSAT cleanly, the issue is confirmed: AMLS asks the wrong question.
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

    # Whitened spec.
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
    print(f'[diag] raw spec: G={G.flatten()}, g={gv}')

    # Whiten the halfspace: y_w = (y - mu) / sigma. Substitute:
    #   G y <= g  =>  G (sigma * y_w + mu) <= g  =>  (G * sigma) y_w <= g - G mu
    mu = y_mean.numpy()
    sigma = y_std.numpy()
    G_w = (G * sigma).astype(np.float64)
    g_w = (gv - (G @ mu).astype(np.float64)).astype(np.float64)
    print(f'[diag] whitened spec: G_w={G_w.flatten()}, g_w={g_w}')

    from n2v.sets.halfspace import HalfSpace
    hs_w = HalfSpace(G_w.reshape(1, -1), g_w.reshape(-1, 1))

    # 1) Scenario disjoint (truncated Gaussian on ||z|| <= q)
    print()
    print(f'[diag] === SCENARIO-DISJOINT (truncated, ||z|| <= q={q:.3f}) ===')
    for n_scn in (2000, 8000, 32000):
        t0 = time.time()
        r = certify_halfspace_disjoint(
            flow_ode=flow, threshold_q=q, halfspace=hs_w,
            n_samples=n_scn, beta_2=0.001, seed=seed,
        )
        print(f'[diag]   N={n_scn:>5d}: disjoint={r.disjoint}  '
              f'max_margin={r.worst_max_margin:+.4f}  wall={time.time()-t0:.2f}s')
        # margin = max_row(G y - g); positive means safe (sample outside U)

    # 2) AMLS for comparison
    print()
    print(f'[diag] === AMLS (full Gaussian + MCMC level splitting) ===')
    spec_groups = [[hs_w]]
    for n_amls in (2000, 8000):
        t0 = time.time()
        r = amls_certify_spec(
            flow_ode=flow, spec_groups=spec_groups,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, beta=0.001, seed=seed,
        )
        first = r.per_hs_results[0][0]
        print(f'[diag]   N={n_amls:>5d}: detected_unsafe={first.detected_unsafe}  '
              f'levels={first.levels_used}  '
              f'final_phi={first.final_phi:+.4f}  '
              f'pi_hat={first.pi_hat:.2e}  '
              f'wall={time.time()-t0:.2f}s')

    print()
    print(f'[diag] === INTERPRETATION ===')
    print(f'[diag] If scenario.disjoint=True, the (1-α) reach set IS disjoint '
          f'from unsafe — UNSAT is the correct verdict.')
    print(f'[diag] If AMLS.detected_unsafe=True nonetheless, it found a '
          f'sample in unsafe via MCMC walk that has near-zero density — '
          f'AMLS is asking the wrong question for our conformal claim.')


if __name__ == '__main__':
    main()
