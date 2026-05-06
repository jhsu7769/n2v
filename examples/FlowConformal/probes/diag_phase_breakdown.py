"""Per-phase wall breakdown of run_verification_pipeline on the 4
NO_FIT benchmarks at nano config. Identifies which step is the
bottleneck so we know whether reducing `m` (calibration set size)
helps fit them in budget.

Phases timed:
    1. Network forward × n_train  (training-data prep)
    2. Network forward × m         (calibration-data prep)
    3. Flow training              (n_train × flow_epochs)
    4. Calibration scoring on m   (m × FlowScore)
    5. AMLS bounded verification  (per-halfspace × levels × MCMC)
    6. Coverage testing
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

VNN_COMP_BASE = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks',
))

# Use nano config to make the comparison fair across benchmarks.
NANO = dict(n_train=100, flow_epochs=200, scenario_n=50)

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


def main():
    alpha = 0.001
    print(f'{"benchmark":18s} {"m":>6s} {"net_fwd_train":>14s} '
          f'{"net_fwd_m":>10s} {"flow_train":>11s} {"cal_score":>10s} '
          f'{"total_so_far":>13s} {"budget":>8s}')
    print('-' * 110)

    for label, dirname, onnx_rel, vnn_rel, budget in TARGETS:
        bench_root = VNN_COMP_BASE / dirname
        try:
            net, boxes, spec = load_instance(bench_root, onnx_rel, vnn_rel)
        except Exception as e:
            print(f'{label:18s} LOAD FAIL: {e}')
            continue
        lb_t = torch.tensor(np.asarray(boxes[0][0]).flatten(),
                            dtype=torch.float32)
        ub_t = torch.tensor(np.asarray(boxes[0][1]).flatten(),
                            dtype=torch.float32)
        n_train = NANO['n_train']

        for m in [8000, 2000, 500, 100]:
            seed = 0
            torch.manual_seed(seed)
            x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
            x_ca = lb_t + torch.rand(m, lb_t.shape[0]) * (ub_t - lb_t)

            # Phase 1: training-data forward
            t0 = time.time()
            with torch.no_grad():
                y_tr = net(x_tr).detach()
            t_fwd_train = time.time() - t0

            # Phase 2: calibration-data forward
            t0 = time.time()
            with torch.no_grad():
                y_ca = net(x_ca).detach()
            t_fwd_m = time.time() - t0

            # Whitening
            y_mean = y_tr.mean(dim=0)
            y_std = y_tr.std(dim=0).clamp_min(1e-8)
            y_tr_w = (y_tr - y_mean) / y_std
            y_ca_w = (y_ca - y_mean) / y_std

            # Phase 3: flow training (nano)
            t0 = time.time()
            try:
                flow, _ = _train_flow(
                    y_tr_w, dim=y_tr_w.shape[1],
                    n_epochs=NANO['flow_epochs'], seed=seed,
                    return_losses=True,
                )
                flow = flow.to('cpu').eval()
            except Exception as e:
                print(f'{label:18s} m={m:>5d} flow_train FAIL: {e}')
                continue
            t_flow = time.time() - t0

            # Phase 4: calibration scoring (m × FlowScore)
            score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4',
                                 batch_size=65536)
            t0 = time.time()
            with torch.no_grad():
                calib_scores = score_fn(y_ca_w)
            t_cal_score = time.time() - t0

            total = t_fwd_train + t_fwd_m + t_flow + t_cal_score
            print(f'{label:18s} {m:>6d} {t_fwd_train:>12.2f}s '
                  f'{t_fwd_m:>9.2f}s {t_flow:>10.2f}s '
                  f'{t_cal_score:>9.2f}s {total:>11.2f}s '
                  f'{budget:>6d}s')


if __name__ == '__main__':
    main()
