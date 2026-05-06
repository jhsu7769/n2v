"""Diagnostic: where does ours' reach set boundary lie on tllverify, and
does it actually extend into the unsafe halfspace?

Loads a tllverify UNKNOWN instance, trains the flow at ours-mega config,
calibrates q, then sweeps y values densely along the spec halfspace
direction to find the boundary of the calibrated reach set
``{y : score(y) <= q}``. Compares to:

  * the data envelope (min/max of y_train) — what Hashemi uses
  * the spec threshold (the unsafe halfspace boundary) — what αβ-CROWN
    proves disjoint

If the reach-set boundary lies BEFORE the spec threshold → the set is
disjoint, AMLS is the over-rejecting culprit.
If the reach-set boundary lies BEYOND the spec threshold → the score
geometry itself is the culprit (flow has tail mass into unsafe).
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


def _forward(net, x_t):
    with torch.no_grad():
        return net(x_t).detach()


def main():
    bench_root = Path(os.path.expanduser(
        '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/tllverifybench_2023'
    ))
    onnx_rel = 'onnx/tllBench_n=2_N=M=8_m=1_instance_0_1.onnx'
    vnn_rel = 'vnnlib/property_N=8_1.vnnlib'

    print(f'[diag] loading {onnx_rel} + {vnn_rel}')
    net, boxes, spec = load_instance(bench_root, onnx_rel, vnn_rel)
    lb, ub = boxes[0]
    lb_t = torch.tensor(np.asarray(lb).flatten(), dtype=torch.float32)
    ub_t = torch.tensor(np.asarray(ub).flatten(), dtype=torch.float32)
    print(f'[diag] input dim={lb_t.shape[0]}, '
          f'lb range=[{lb_t.min():.3f}, {lb_t.max():.3f}], '
          f'ub range=[{ub_t.min():.3f}, {ub_t.max():.3f}]')

    # Sample training, calibration, test data — same procedure as verify_flow
    seed = 0
    n_train = 10000
    m_calib = 8000
    n_eval = 4000

    torch.manual_seed(seed)
    x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
    x_ca = lb_t + torch.rand(m_calib, lb_t.shape[0]) * (ub_t - lb_t)
    x_eval = lb_t + torch.rand(n_eval, lb_t.shape[0]) * (ub_t - lb_t)

    y_tr = _forward(net, x_tr)
    y_ca = _forward(net, x_ca)
    y_eval = _forward(net, x_eval)
    print(f'[diag] output dim={y_tr.shape[1]}')
    print(f'[diag] y_train range: '
          f'min={float(y_tr.min()):.4f} max={float(y_tr.max()):.4f}')
    print(f'[diag] y_train mean={float(y_tr.mean()):.4f} '
          f'std={float(y_tr.std()):.4f}')

    # Check spec
    print(f'[diag] spec type: {type(spec).__name__}')
    if isinstance(spec, list):
        for i, s in enumerate(spec):
            G = s.G if hasattr(s, 'G') else None
            g = s.g if hasattr(s, 'g') else None
            print(f'[diag] spec disjunct {i}: G={np.asarray(G).flatten()} '
                  f'g={np.asarray(g).flatten()}')
    elif hasattr(spec, 'G'):
        G = spec.G
        g = spec.g
        print(f'[diag] spec single: G={np.asarray(G).flatten()} '
              f'g={np.asarray(g).flatten()}')

    # Whiten
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std
    y_eval_w = (y_eval - y_mean) / y_std

    # Train flow
    print(f'[diag] training flow (n_train={n_train}, epochs=2000)...')
    t0 = time.time()
    flow, losses = _train_flow(
        y_tr_w, dim=y_tr_w.shape[1], n_epochs=2000, seed=seed,
        return_losses=True,
    )
    flow = flow.to('cpu').eval()
    print(f'[diag]   trained in {time.time()-t0:.1f}s; '
          f'loss {losses[0]:.4f} -> {losses[-1]:.6f}')

    # Calibrate
    score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4',
                         batch_size=65536)
    with torch.no_grad():
        calib_scores = score_fn(y_ca_w)
    alpha = 0.001
    import math
    ell = max(1, min(m_calib, int(math.ceil((m_calib + 1) * (1 - alpha)))))
    q = calibrate(calib_scores, ell).item()
    print(f'[diag] calibrated q (1-α={1-alpha}, m={m_calib}, ell={ell}): '
          f'{q:.4f}')
    print(f'[diag] calib_scores: '
          f'min={float(calib_scores.min()):.4f} '
          f'mean={float(calib_scores.mean()):.4f} '
          f'max={float(calib_scores.max()):.4f}')

    # Empirical coverage
    with torch.no_grad():
        eval_scores = score_fn(y_eval_w)
    coverage = float((eval_scores <= q).float().mean())
    print(f'[diag] empirical coverage on eval: {coverage:.4f}')

    # Now: along the spec direction, find where score(y) = q.
    # tllverify spec is single halfspace G y <= g (UNSAFE region).
    # For a 1-D output, G is a sign and g a scalar.
    # Spec normalisation: tllverify ships single-disjunct AND specs.
    # ``load_instance`` may return either a HalfSpace or a list of dicts
    # with ``Hg`` keys.
    if isinstance(spec, list):
        s = spec[0]
    else:
        s = spec
    if isinstance(s, dict):
        hs = s.get('Hg')
        if isinstance(hs, list):
            hs = hs[0]
        s = hs
    G = np.asarray(s.G, dtype=np.float64).flatten()
    gv = np.asarray(s.g, dtype=np.float64).flatten()
    print(f'[diag] spec G (raw): {G}, g (raw): {gv}')

    # Sweep along the direction G, in original (un-whitened) y space.
    # Span y in [y_tr.min() - 5*std, y_tr.max() + 5*std] along the spec
    # direction.
    y_min = float(y_tr.min())
    y_max = float(y_tr.max())
    yt_std = float(y_tr.std())
    # For 1D output, just sweep y values directly.
    sweep = np.linspace(y_min - 5 * yt_std, y_max + 5 * yt_std, 2000)
    sweep_t = torch.tensor(sweep.reshape(-1, 1), dtype=torch.float32)
    sweep_w = (sweep_t - y_mean) / y_std
    with torch.no_grad():
        sweep_scores = score_fn(sweep_w).cpu().numpy()

    # Find boundary: smallest y > y_max where score crosses q.
    # And largest y < y_min where score crosses q.
    in_set = sweep_scores <= q
    print(f'[diag] sweep range: '
          f'y in [{sweep[0]:.4f}, {sweep[-1]:.4f}]')
    # Continuous segments where in_set is True
    transitions = np.where(np.diff(in_set.astype(int)))[0]
    print(f'[diag] transitions of (score <= q) at y-values: '
          f'{[float(sweep[i]) for i in transitions]}')

    # The reach-set boundary on the right (high y side):
    in_indices = np.where(in_set)[0]
    if len(in_indices) > 0:
        reach_lo = float(sweep[in_indices.min()])
        reach_hi = float(sweep[in_indices.max()])
        print(f'[diag] reach-set boundary (where score <= q): '
              f'y in [{reach_lo:.4f}, {reach_hi:.4f}]')
    else:
        print(f'[diag] no y in sweep has score <= q (degenerate)')

    # Spec boundary in y-coords: G y <= g defines unsafe.
    # If G is 1, unsafe is y <= g. If G is -1, unsafe is y >= -g/G ... wait,
    # G y <= g with G=-1 means -y <= g, i.e., y >= -g.
    # We need the unsafe halfspace boundary as a y value.
    if G.size == 1:
        if G[0] > 0:
            unsafe_thresh = float(gv[0] / G[0])
            unsafe_dir = 'y <= '
        else:
            unsafe_thresh = float(-gv[0] / abs(G[0])) if G[0] < 0 else float('inf')
            unsafe_dir = 'y >= '
        print(f'[diag] unsafe halfspace: {unsafe_dir}{unsafe_thresh:.4f}')

    # Compare to data envelope and reach-set extent
    print()
    print(f'[diag] === SUMMARY ===')
    print(f'[diag] data envelope:    y in [{y_min:.4f}, {y_max:.4f}]')
    if 'reach_lo' in locals():
        print(f'[diag] reach-set:        y in [{reach_lo:.4f}, {reach_hi:.4f}]')
    if G.size == 1:
        print(f'[diag] unsafe halfspace: {unsafe_dir}{unsafe_thresh:.4f}')
        if 'reach_lo' in locals():
            if G[0] > 0 and reach_lo < unsafe_thresh:
                print(f'[diag] !! reach-set EXTENDS into unsafe '
                      f'(reach_lo={reach_lo:.4f} < unsafe_thresh={unsafe_thresh:.4f})')
            elif G[0] < 0 and reach_hi > unsafe_thresh:
                print(f'[diag] !! reach-set EXTENDS into unsafe '
                      f'(reach_hi={reach_hi:.4f} > unsafe_thresh={unsafe_thresh:.4f})')
            else:
                print(f'[diag] ✓ reach-set is DISJOINT from unsafe halfspace')

    # Margin: how much "headroom" between reach set and unsafe?
    if G.size == 1 and 'reach_lo' in locals():
        if G[0] > 0:  # unsafe is y <= unsafe_thresh
            margin = reach_lo - unsafe_thresh
            print(f'[diag] margin (reach_lo - unsafe_thresh): {margin:.4f}')
        else:
            margin = unsafe_thresh - reach_hi
            print(f'[diag] margin (unsafe_thresh - reach_hi): {margin:.4f}')


if __name__ == '__main__':
    main()
