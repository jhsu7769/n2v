"""Sample from the trained flow and directly count tail mass in
unsafe region. Compares to the calibration-based UNSAT statement.

If `Pr[y in unsafe | y ~ flow] >> 0` despite the reach-set being
disjoint, the pipeline is using a verification primitive (AMLS rare-
event probability) that is provably weaker than direct level-set
disjointness checking.
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
    x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
    with torch.no_grad():
        y_tr = net(x_tr).detach()
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std

    print(f'[diag] training flow on {n_train} points...')
    t0 = time.time()
    flow, _ = _train_flow(
        y_tr_w, dim=y_tr_w.shape[1], n_epochs=2000, seed=seed,
        return_losses=True,
    )
    flow = flow.to('cpu').eval()
    print(f'[diag]   trained in {time.time()-t0:.1f}s')

    # Sample from the flow's distribution by integrating from latent to data.
    # The flow's prior is standard Gaussian in y-whitened space.
    print(f'[diag] sampling from the flow...')
    M_total = 200_000
    chunk = 20_000
    samples_y = []
    with torch.no_grad():
        for k in range(M_total // chunk):
            z = torch.randn(chunk, y_tr_w.shape[1])
            y_w = flow.inverse(z, t=1.0, n_steps=30, method='rk4')
            y_orig = y_w * y_std + y_mean
            samples_y.append(y_orig.cpu().numpy())
    samples_y = np.concatenate(samples_y, axis=0).flatten()
    print(f'[diag]   {M_total} flow samples; '
          f'min={samples_y.min():.4f} max={samples_y.max():.4f} '
          f'mean={samples_y.mean():.4f} std={samples_y.std():.4f}')

    # spec disjunct: G y <= g unsafe. Extract.
    if isinstance(spec, list):
        s = spec[0]
    else:
        s = spec
    if isinstance(s, dict):
        hs = s.get('Hg')
        if isinstance(hs, list): hs = hs[0]
        s = hs
    G = np.asarray(s.G, dtype=np.float64).flatten()
    gv = np.asarray(s.g, dtype=np.float64).flatten()
    # Convert G y <= g to a y-threshold for 1D output.
    if G[0] > 0:
        unsafe_thresh = gv[0] / G[0]
        unsafe_mask = samples_y <= unsafe_thresh
        unsafe_dir = f'y <= {unsafe_thresh:.4f}'
    else:
        unsafe_thresh = -gv[0] / abs(G[0])
        unsafe_mask = samples_y >= unsafe_thresh
        unsafe_dir = f'y >= {unsafe_thresh:.4f}'

    p_unsafe = float(unsafe_mask.mean())
    n_unsafe = int(unsafe_mask.sum())
    print()
    print(f'[diag] unsafe halfspace: {unsafe_dir}')
    print(f'[diag] flow samples in unsafe region: {n_unsafe} / {M_total} '
          f'(p={p_unsafe:.6f})')
    print()
    if p_unsafe > 0.001:
        print(f'[diag] !! flow has nonzero mass in unsafe ({p_unsafe:.6f} > '
              f'eps=0.001) — AMLS is correct to detect this; the reach-set '
              f'disjointness statement is misleading')
    elif p_unsafe == 0:
        print(f'[diag] ✓ flow has 0 samples in unsafe; AMLS should be able to '
              f'certify UNSAT')
    else:
        print(f'[diag] ~ flow has small mass in unsafe ({p_unsafe:.6f} <= '
              f'eps=0.001); AMLS could in principle certify UNSAT '
              f'with sufficient samples')

    # What does AMLS at default config see?
    print()
    print(f'[diag] for context: at scenario_n_samples=2000, with rare-event '
          f'probability {p_unsafe:.6f}, the expected count is '
          f'{p_unsafe * 2000:.2f} samples. AMLS uses level splitting so it '
          f'will find them even at low probability.')


if __name__ == '__main__':
    main()
