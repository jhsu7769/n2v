"""Time AMLS bounded specifically on the NO_FIT benchmarks at micro
config to see why the lock probe TIMEOUTed there. Suspicion: AMLS
takes 90+ seconds despite cheap flow + cal phases — likely because
network-forward inside flow integration is slower than expected on
high-dim networks.
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
from n2v.probabilistic.flow.amls_bounded import (  # noqa: E402
    amls_bounded_certify_spec,
)


VNN_COMP_BASE = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks',
))

TARGETS = [
    ('cora_2024', 'cora_2024',
     'onnx/mnist-point.onnx', 'vnnlib/mnist-img0.vnnlib', 30),
    ('safenlp_2024', 'safenlp_2024',
     'onnx/medical/perturbations_0.onnx',
     'vnnlib/medical/hyperrectangle_418.vnnlib', 20),
    ('vit_2023', 'vit_2023',
     'onnx/pgd_2_3_16.onnx', 'vnnlib/pgd_2_3_16_2446.vnnlib', 100),
    ('yolo_2023', 'cctsdb_yolo_2023',
     'onnx/patch-1.onnx',
     'vnnlib/spec_onnx_patch-1_idx_00559_0.vnnlib', 350),
]


def whiten_halfspace(hs, mu, sigma):
    from n2v.sets.halfspace import HalfSpace
    G = np.asarray(hs.G, dtype=np.float64)
    g = np.asarray(hs.g, dtype=np.float64).flatten()
    G_w = (G * sigma).astype(np.float64)
    g_w = (g - (G @ mu).astype(np.float64)).astype(np.float64)
    return HalfSpace(G_w, g_w.reshape(-1, 1))


def normalize_spec(spec, mu, sigma):
    from n2v.sets.halfspace import HalfSpace
    if isinstance(spec, list):
        groups = []
        for g in spec:
            if isinstance(g, dict):
                hs = g.get('Hg')
                if isinstance(hs, list):
                    groups.append([whiten_halfspace(h, mu, sigma) for h in hs])
                else:
                    groups.append([whiten_halfspace(hs, mu, sigma)])
            elif isinstance(g, HalfSpace):
                groups.append([whiten_halfspace(g, mu, sigma)])
        return groups
    if isinstance(spec, HalfSpace):
        return [[whiten_halfspace(spec, mu, sigma)]]
    raise ValueError(f'unsupported spec type: {type(spec)}')


def main():
    print(f'{"benchmark":18s} {"output_dim":>10s} '
          f'{"n_disjuncts":>11s} {"amls_wall":>10s} '
          f'{"per_hs_walls":>15s} {"levels":>15s} {"verdict":>15s}')
    print('-' * 120)

    n_train = 200      # micro
    flow_epochs = 300  # micro
    scenario_n = 100   # micro
    m = 2000           # less than verify_flow default 8000

    for label, dirname, onnx_rel, vnn_rel, budget in TARGETS:
        bench_root = VNN_COMP_BASE / dirname
        try:
            net, boxes, spec = load_instance(bench_root, onnx_rel, vnn_rel)
        except Exception as e:
            print(f'{label:18s} LOAD FAIL: {e}')
            continue
        seed = 0
        torch.manual_seed(seed); np.random.seed(seed)
        lb_t = torch.tensor(np.asarray(boxes[0][0]).flatten(),
                            dtype=torch.float32)
        ub_t = torch.tensor(np.asarray(boxes[0][1]).flatten(),
                            dtype=torch.float32)
        x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
        x_ca = lb_t + torch.rand(m, lb_t.shape[0]) * (ub_t - lb_t)
        with torch.no_grad():
            y_tr = net(x_tr).detach()
            y_ca = net(x_ca).detach()
        y_mean = y_tr.mean(dim=0)
        y_std = y_tr.std(dim=0).clamp_min(1e-8)
        y_tr_w = (y_tr - y_mean) / y_std
        y_ca_w = (y_ca - y_mean) / y_std
        out_dim = y_tr_w.shape[1]

        flow, _ = _train_flow(
            y_tr_w, dim=out_dim, n_epochs=flow_epochs, seed=seed,
            return_losses=True,
        )
        flow = flow.to('cpu').eval()
        score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4',
                             batch_size=65536)
        with torch.no_grad():
            calib_scores = score_fn(y_ca_w)
        import math
        ell = max(1, min(m, int(math.ceil((m + 1) * (1 - 0.001)))))
        q = calibrate(calib_scores, ell).item()

        whitened_groups = normalize_spec(spec, y_mean.numpy(),
                                          y_std.numpy())
        n_hs_total = sum(len(g) for g in whitened_groups)

        # Time AMLS bounded.
        t0 = time.time()
        r = amls_bounded_certify_spec(
            flow_ode=flow, spec_groups=whitened_groups, q=q,
            eps_2_target=0.001,
            n_samples_per_level=scenario_n,
            quantile=0.1, max_levels=30, n_mcmc_steps=10,
            mcmc_step_size=0.3, adaptive_step=False, beta=0.001,
            seed=seed,
        )
        amls_wall = time.time() - t0

        per_hs_walls = []
        per_hs_levels = []
        for grp in r.per_hs_results:
            for hs_r in grp:
                per_hs_levels.append(hs_r.levels_used)
        verdict = 'UNSAT' if r.unsat_certified else 'UNKNOWN'

        print(f'{label:18s} {out_dim:>10d} {n_hs_total:>11d} '
              f'{amls_wall:>9.1f}s {"-":>15s} {str(per_hs_levels):>15s} '
              f'{verdict:>15s}  budget={budget}s')


if __name__ == '__main__':
    main()
