"""Loader audit for the 5 NO_FIT / problem benchmarks.

For each (benchmark, sample instance):
    * input_dim, output_dim
    * spec structure (single halfspace? OR groups? AND-of-rows?)
    * per-sample network forward time (CPU + GPU if available)
    * memory per sample, flow-training data prep cost projection
    * any loader warnings / fall-throughs (batch_loop_unbatched flag, etc.)

Goal: distinguish loader bugs (fixable) from intrinsic compute issues
(network too big for the per-instance budget).
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


# Benchmarks to audit + a representative instance for each.
# (benchmark, onnx_rel, vnnlib_rel, vnncomp_budget_s, expected_input_dim, expected_output_dim)
AUDIT_TARGETS = [
    ('acasxu_2023',
     'onnx/ACASXU_run2a_1_1_batch_2000.onnx',
     'vnnlib/prop_1.vnnlib', 116, 5, 5),
    ('cora_2024',
     'onnx/mnist-point.onnx',
     'vnnlib/mnist-img0.vnnlib', 30, None, None),
    ('safenlp_2024',
     'onnx/medical/perturbations_0.onnx',
     'vnnlib/medical/hyperrectangle_418.vnnlib', 20, None, None),
    ('vit_2023',
     'onnx/pgd_2_3_16.onnx',
     'vnnlib/pgd_2_3_16_2446.vnnlib', 100, None, None),
    ('yolo_2023',
     'onnx/patch-1.onnx',
     'vnnlib/spec_onnx_patch-1_idx_00559_0.vnnlib', 350, None, None),
]


# ---------------------------------------------------------------------------
# Per-benchmark roots (some need a non-default subdir name).
# ---------------------------------------------------------------------------

VNN_COMP_BASE = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks',
))

BENCH_DIR = {
    'acasxu_2023': 'acasxu_2023',
    'cora_2024': 'cora_2024',
    'safenlp_2024': 'safenlp_2024',
    'vit_2023': 'vit_2023',
    'yolo_2023': 'cctsdb_yolo_2023',
}


def time_forward(net, x_batch, n_repeats: int = 5):
    """Time per-sample forward over n_repeats. Returns ms/sample on the
    network's resident device.
    """
    with torch.no_grad():
        for _ in range(2):
            _ = net(x_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_repeats):
            _ = net(x_batch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    per_call = elapsed / n_repeats
    per_sample = per_call / max(1, x_batch.shape[0])
    return per_sample * 1000.0


def describe_spec(spec) -> str:
    if spec is None:
        return 'None'
    if isinstance(spec, list):
        if len(spec) == 0:
            return 'empty list'
        first = spec[0]
        if hasattr(first, 'G'):
            G_shape = np.asarray(first.G).shape
            return f'list[HalfSpace] (n_disjuncts={len(spec)}, '\
                   f'first.G.shape={G_shape})'
        if isinstance(first, dict):
            keys = list(first.keys())
            inner = first.get('Hg')
            if isinstance(inner, list):
                inner_shape = np.asarray(inner[0].G).shape if hasattr(
                    inner[0], 'G') else 'unknown'
                return (f'list[dict-with-Hg-list] '
                        f'(n_groups={len(spec)}, group0_size={len(inner)}, '
                        f'first_disjunct_G.shape={inner_shape})')
            if hasattr(inner, 'G'):
                return (f'list[dict-with-Hg-HalfSpace] '
                        f'(n_groups={len(spec)}, '
                        f'first_disjunct_G.shape={np.asarray(inner.G).shape})')
            return f'list[dict] (keys={keys}, n_groups={len(spec)})'
        return f'list (len={len(spec)}, first={type(first).__name__})'
    if hasattr(spec, 'G'):
        return f'HalfSpace (G.shape={np.asarray(spec.G).shape})'
    return f'<{type(spec).__name__}>'


def main():
    print(f'{"benchmark":18s} {"in_dim":>8s} {"out_dim":>8s} {"n_boxes":>8s} '
          f'{"per_sample_fwd":>17s} {"wrapper_mode":>20s}')
    print('-' * 100)
    for bench, onnx_rel, vnn_rel, budget, exp_in, exp_out in AUDIT_TARGETS:
        bench_root = VNN_COMP_BASE / BENCH_DIR[bench]
        try:
            net, boxes, spec = load_instance(bench_root, onnx_rel, vnn_rel)
        except Exception as e:
            print(f'{bench:18s} LOAD FAILED: {type(e).__name__}: {e}')
            continue
        lb, ub = boxes[0]
        lb_t = torch.tensor(np.asarray(lb).flatten(), dtype=torch.float32)
        ub_t = torch.tensor(np.asarray(ub).flatten(), dtype=torch.float32)
        in_dim = int(lb_t.numel())

        # Forward pass once to learn output dim.
        x = (lb_t + (ub_t - lb_t) * 0.5).unsqueeze(0)
        try:
            with torch.no_grad():
                y = net(x)
            out_dim = int(np.prod(y.shape[1:])) if y.dim() > 1 else int(y.shape[0])
        except Exception as e:
            print(f'{bench:18s} forward failed (batch=1): '
                  f'{type(e).__name__}: {str(e)[:90]}')
            continue

        # Wrapper mode (the _GenericONNXWrapper records this).
        batch_loop = bool(getattr(net, 'batch_loop', False))
        batch_loop_unbatched = bool(getattr(net, 'batch_loop_unbatched', False))
        if batch_loop_unbatched:
            mode = 'batch_loop_unbatched'
        elif batch_loop:
            mode = 'batch_loop_b1'
        else:
            mode = 'native_batched'

        # Per-sample CPU forward time (n_repeats=5, batch=1).
        per_sample_ms = time_forward(net, x, n_repeats=5)

        # Number of input boxes (matters for OR-of-input specs like prop_6).
        n_boxes = len(boxes)

        print(f'{bench:18s} {in_dim:>8d} {out_dim:>8d} {n_boxes:>8d} '
              f'{per_sample_ms:>14.3f} ms  {mode:>20s}')

        spec_desc = describe_spec(spec)
        print(f'{"":18s}   spec: {spec_desc}')
        print(f'{"":18s}   y range on midpoint: '
              f'[{float(y.min()):.4f}, {float(y.max()):.4f}]')
        print(f'{"":18s}   budget: {budget}s | '
              f'projected n_train=500 fwd cost: '
              f'{per_sample_ms * 500 / 1000:.1f}s')
        print()


if __name__ == '__main__':
    main()
