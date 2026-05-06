"""Compare our network wrapper outputs to onnxruntime (canonical) on
the same inputs. If they differ, our loader has a bug; if they match,
the NO_FIT / FALSE_UNSAT issue is calibration / compute, not loader.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ_ROOT))

from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (  # noqa: E402
    load_instance,
)


VNN_COMP_BASE = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks',
))

TARGETS = [
    ('acasxu_2023', 'acasxu_2023',
     'onnx/ACASXU_run2a_1_1_batch_2000.onnx',
     'vnnlib/prop_1.vnnlib'),
    ('cora_2024', 'cora_2024',
     'onnx/mnist-point.onnx',
     'vnnlib/mnist-img0.vnnlib'),
    ('safenlp_2024', 'safenlp_2024',
     'onnx/medical/perturbations_0.onnx',
     'vnnlib/medical/hyperrectangle_418.vnnlib'),
    ('vit_2023', 'vit_2023',
     'onnx/pgd_2_3_16.onnx',
     'vnnlib/pgd_2_3_16_2446.vnnlib'),
    ('yolo_2023', 'cctsdb_yolo_2023',
     'onnx/patch-1.onnx',
     'vnnlib/spec_onnx_patch-1_idx_00559_0.vnnlib'),
]


def main():
    try:
        import onnxruntime as ort
    except ImportError:
        print('onnxruntime not installed; pip install onnxruntime to run this')
        return

    for label, dirname, onnx_rel, vnn_rel in TARGETS:
        bench_root = VNN_COMP_BASE / dirname
        print(f'=== {label} ({onnx_rel}) ===')

        try:
            net, boxes, _spec = load_instance(bench_root, onnx_rel, vnn_rel)
        except Exception as e:
            print(f'  load failed: {type(e).__name__}: {e}')
            print()
            continue

        lb_np = np.asarray(boxes[0][0], dtype=np.float32).flatten()
        ub_np = np.asarray(boxes[0][1], dtype=np.float32).flatten()
        in_dim = lb_np.size

        # Sample 5 random points within the input box.
        rng = np.random.default_rng(0)
        xs = rng.uniform(lb_np, ub_np, size=(5, in_dim)).astype(np.float32)

        # Wrapper outputs.
        with torch.no_grad():
            xs_t = torch.tensor(xs)
            y_wrap = net(xs_t).cpu().numpy()
        if y_wrap.ndim > 2:
            y_wrap = y_wrap.reshape(y_wrap.shape[0], -1)

        # ONNX Runtime outputs.
        sess = ort.InferenceSession(
            str(bench_root / onnx_rel), providers=['CPUExecutionProvider']
        )
        inp_name = sess.get_inputs()[0].name
        inp_shape = sess.get_inputs()[0].shape
        out_names = [o.name for o in sess.get_outputs()]
        # Reshape per the input metadata; default to (5, in_dim).
        ort_in = xs
        # ORT wants the shape declared by the model. Some models declare
        # static shapes incompatible with a batch of 5 — try a few
        # candidates.
        rt_outs = None
        for shape_try in [
                (5, in_dim),
                (5,) + tuple(d for d in inp_shape[1:] if isinstance(d, int)),
                tuple(d if isinstance(d, int) else 1 for d in inp_shape),
                (in_dim,),
        ]:
            try:
                if shape_try == (in_dim,):
                    # Single-sample call when the model only takes 1-D.
                    per_sample_outs = []
                    for k in range(5):
                        out_k = sess.run(out_names,
                                         {inp_name: xs[k].reshape(shape_try)})
                        per_sample_outs.append(out_k[0])
                    y_ort = np.stack([o.reshape(-1) for o in per_sample_outs],
                                     axis=0)
                else:
                    if shape_try[0] != 5:
                        continue
                    out = sess.run(out_names, {inp_name: xs.reshape(shape_try)})
                    y_ort = out[0]
                    if y_ort.ndim > 2:
                        y_ort = y_ort.reshape(y_ort.shape[0], -1)
                rt_outs = y_ort
                break
            except Exception:
                continue
        if rt_outs is None:
            print(f'  ORT could not be invoked with any candidate shape; '
                  f'declared input shape: {inp_shape}')
            print()
            continue

        # Diff
        if y_wrap.shape != rt_outs.shape:
            print(f'  ⚠ SHAPE MISMATCH: wrapper {y_wrap.shape} vs ORT '
                  f'{rt_outs.shape}')
            # Try to broadcast/squeeze for comparison.
            try:
                w = y_wrap.reshape(rt_outs.shape)
            except Exception:
                w = None
            if w is not None:
                y_wrap = w
            else:
                print()
                continue

        max_abs = float(np.max(np.abs(y_wrap - rt_outs)))
        rel = float(np.max(
            np.abs(y_wrap - rt_outs) / (np.maximum(np.abs(rt_outs), 1e-9))))
        ok = max_abs < 1e-3
        print(f'  shapes: wrapper={y_wrap.shape}, ORT={rt_outs.shape}')
        print(f'  max |Δ|: {max_abs:.6e}    max rel Δ: {rel:.6e}    '
              f'{"✓ MATCH" if ok else "✗ MISMATCH"}')
        if not ok:
            for k in range(min(3, len(xs))):
                print(f'    sample {k}: wrapper={y_wrap[k][:5]} ORT={rt_outs[k][:5]}')
        print()


if __name__ == '__main__':
    main()
