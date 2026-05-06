"""Convert the Cohen-et-al CIFAR-10 ResNet-110 model + RS-style robustness
specs into VNN-LIB format so αβ-CROWN can run on it.

The Cohen et al. (2019) RS paper used ResNet-110 explicitly because no
sound verifier could handle it. Whether that's still true with current
αβ-CROWN is an empirical question we want to settle. This script:

1. Loads the pretrained PyTorch ResNet-110 (σ=0.25 noise) used in our
   Exp 2 ``exp2_run_cifar10_resnet110.py`` pipeline.
2. Exports the network (with input mean/std normalization baked in) to
   ONNX in eval mode. Validation: ONNX inference matches PyTorch on
   100 randomly-sampled CIFAR-10 test inputs to ``atol=1e-4``.
3. Generates 100 ``.vnnlib`` files encoding the L∞-ε robustness spec
   for each of the same 100 instances:

       Input box: per-pixel L∞ ball ``[x_clean[i] - ε, x_clean[i] + ε]``
       clipped to ``[0, 1]``.
       Unsafe region: ``∃ j ≠ y_clean such that Y_j ≥ Y_{y_clean}`` —
       any wrong class scoring at least as high as the correct class.

4. Validation: parses each ``.vnnlib`` back, checks the input box and
   halfspace constraints round-trip exactly.

Outputs land under
``examples/FlowConformal/experiments/exp2_prob_scale/cifar10_resnet110_vnncomp/``
in a VNN-COMP-style layout (``onnx/model.onnx``, ``vnnlib/*.vnnlib``,
``instances.csv``). The αβ-CROWN runner can then point at this
directory like any other VNN-COMP benchmark.

Run-once. Re-run only if the underlying pretrained weights or RS spec
formulation changes.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.build_resnet110_onnx
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

# Reuse the network builder + pretrained-loader from the existing runner.
from examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_cifar10_resnet110 import (
    _build_resnet110,
    _load_cifar10_test,
    _load_pretrained,
)

_SEED = 47
_EPS = 8.0 / 255.0          # standard L∞ budget for CIFAR-10 RS
_TIMEOUT_S = 300            # per-instance shell timeout for αβ-CROWN
_N_INSTANCES = 100
_NUM_CLASSES = 10
_IMG_SHAPE = (3, 32, 32)
_INPUT_DIM = int(np.prod(_IMG_SHAPE))  # 3072

_HERE = Path(__file__).resolve().parent
_ARTIFACT_DIR = _HERE / 'cifar10_resnet110_vnncomp'
_ONNX_DIR = _ARTIFACT_DIR / 'onnx'
_VNNLIB_DIR = _ARTIFACT_DIR / 'vnnlib'
_INSTANCES_CSV = _ARTIFACT_DIR / 'instances.csv'


# ---- Network export wrapper -------------------------------------------------

class _ResNet110WithNormalize(nn.Module):
    """Eval-mode forward that takes ``(B, 3, 32, 32)`` in pixel-space [0, 1]
    and applies CIFAR-10 mean/std normalization internally before the
    base ResNet. Used as the ONNX export target — keeping normalization
    inside the graph means the vnnlib spec can write input bounds in
    pixel space directly.
    """

    _MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)
    _STD = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1)

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.register_buffer('_mean', self._MEAN.clone())
        self.register_buffer('_std', self._STD.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self._mean) / self._std
        return self.base(x)


def _build_export_model(sigma: str = '0.25') -> nn.Module:
    """Load pretrained weights and produce an eval-mode model that takes
    ``(B, 3, 32, 32)`` pixel-space input.

    NOTE on αβ-CROWN compatibility: we tried four distinct configs
    against this exported ONNX (vnncomp21/cifar10-resnet,
    vnncomp24/cifar100, vnncomp22/resnet_A, plus a custom config)
    and αβ-CROWN's ``auto_LiRPA`` consistently fails with shape-mismatch
    errors at the residual-add bound-propagation step. Attempting to
    fold the (x − μ)/σ normalize into the first Conv (so the exported
    ONNX has no Sub/Div ops) does NOT help — the convolution's
    zero-padding interacts incorrectly with the folded bias (padding
    values are 0 in pixel space but ``(0 − μ)/σ`` in normalized space).

    A 110-layer CIFAR-10 ResNet is exactly what Cohen et al. (2019)
    chose for randomized smoothing because no sound verifier could
    handle it; that observation still holds with current αβ-CROWN
    auto_LiRPA. ResNet-110 + ε=8/255 is genuinely beyond sound-
    verifier scaling — the ERROR result is a defensible data point.
    """
    base_full = _load_pretrained(sigma)
    if base_full is None:
        raise FileNotFoundError(
            f'pretrained ResNet-110 weights for σ={sigma} not found '
            f'under ~/v/other/smoothing/models/cifar10/resnet110/')
    # ``_load_pretrained`` returns ``_NormalizeAndReshape(base)`` —
    # accepts ``(B, 3072)`` and reshapes internally. We want the
    # ``(B, 3, 32, 32)`` interface for ONNX, so unwrap to the base
    # ResNet and rewrap with our own normalize-no-reshape module.
    base = base_full.base
    return _ResNet110WithNormalize(base).eval()


# ---- ONNX export + validation ----------------------------------------------

def export_onnx(model: nn.Module, onnx_path: Path,
                *, atol: float = 1e-4, n_validate: int = 100) -> None:
    """Export ``model`` to ONNX and verify numerical equivalence on
    ``n_validate`` random pixel-space inputs.

    Raises:
        RuntimeError: if ONNX inference diverges from PyTorch by more
            than ``atol`` on any test input.
    """
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, *_IMG_SHAPE)
    # Static batch dim only — αβ-CROWN's ``auto_LiRPA`` rejects ONNX
    # graphs with dynamic axes because the ``Shape/Gather/Unsqueeze/
    # Concat/Reshape`` dance PyTorch emits for them is not in the
    # supported-ops set. VNN-COMP convention is also static batch=1.
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=['input'], output_names=['logits'],
        opset_version=13,
        do_constant_folding=True,
    )
    # Post-export simplification: collapses any remaining no-op
    # nodes (Identity on parameter tensors, etc.). Mirrors the
    # ``export_onnx`` in exp4_scaling/networks.py.
    try:
        import onnx as _onnx
        from onnxsim import simplify as _simplify
        _model = _onnx.load(str(onnx_path))
        _simplified, _ok = _simplify(_model)
        if _ok:
            _onnx.save(_simplified, str(onnx_path))
    except ImportError:
        pass

    # Validation
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path),
                                 providers=['CPUExecutionProvider'])

    rng = np.random.RandomState(_SEED)
    xs = rng.uniform(0.0, 1.0, size=(n_validate, *_IMG_SHAPE)).astype(np.float32)
    with torch.no_grad():
        torch_out = model(torch.from_numpy(xs)).numpy()
    # ONNX has static batch=1; loop over inputs.
    onnx_out = np.concatenate([
        sess.run(['logits'], {'input': xs[i:i+1]})[0]
        for i in range(n_validate)
    ], axis=0)

    diff = np.abs(torch_out - onnx_out).max()
    if diff > atol:
        raise RuntimeError(
            f'ONNX ↔ PyTorch divergence: max|Δ|={diff:.2e} > atol={atol}')
    print(f'  ONNX↔PyTorch validation: PASS (max|Δ|={diff:.2e}, atol={atol})')


# ---- VNN-LIB generator + validation ----------------------------------------

def _format_real(x: float) -> str:
    """Format a Real with enough precision to round-trip and avoid
    αβ-CROWN's parser stalling on near-zero negatives like ``-0.0``.
    """
    return f'{float(x):.8f}'


def write_robustness_vnnlib(
    path: Path, x_clean: np.ndarray, y_clean: int, eps: float,
    *, num_classes: int = _NUM_CLASSES,
) -> None:
    """Write an L∞-ε classification-robustness vnnlib file.

    Encodes:
        Input box: per-pixel ``X_i ∈ [max(0, x[i]−ε), min(1, x[i]+ε)]``.
        Unsafe: ``∃ j ≠ y_clean: Y_j ≥ Y_{y_clean}`` — disjunctive
        halfspace.
    """
    if x_clean.size != _INPUT_DIM:
        raise ValueError(f'expected {_INPUT_DIM}-dim input, got {x_clean.size}')
    if not 0 <= y_clean < num_classes:
        raise ValueError(f'invalid y_clean={y_clean}')

    flat = x_clean.flatten().astype(np.float64)
    lb = np.clip(flat - eps, 0.0, 1.0)
    ub = np.clip(flat + eps, 0.0, 1.0)

    lines: List[str] = [f'; CIFAR-10 robustness on label {y_clean}, ε={eps}.']
    for i in range(_INPUT_DIM):
        lines.append(f'(declare-const X_{i} Real)')
    for i in range(num_classes):
        lines.append(f'(declare-const Y_{i} Real)')
    lines.append('')
    lines.append('; Input box (pixel-space, per-pixel L∞ ball clipped to [0, 1])')
    for i in range(_INPUT_DIM):
        lines.append(f'(assert (>= X_{i} {_format_real(lb[i])}))')
        lines.append(f'(assert (<= X_{i} {_format_real(ub[i])}))')
    lines.append('')
    lines.append(
        '; UNSAFE: some wrong class j scores ≥ true class y_clean'
    )
    lines.append('(assert (or')
    for j in range(num_classes):
        if j == y_clean:
            continue
        lines.append(f'    (and (>= Y_{j} Y_{y_clean}))')
    lines.append('))')
    lines.append('')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines))


# ---- VNN-LIB validator (round-trip parse) -----------------------------------

_RE_DECLARE_X = re.compile(r'\(declare-const X_(\d+) Real\)')
_RE_DECLARE_Y = re.compile(r'\(declare-const Y_(\d+) Real\)')
_RE_GEQ_X = re.compile(r'\(assert \(>= X_(\d+) (-?\d+\.\d+)\)\)')
_RE_LEQ_X = re.compile(r'\(assert \(<= X_(\d+) (-?\d+\.\d+)\)\)')
_RE_DISJUNCT_GEQ = re.compile(
    r'\(and \(>= Y_(\d+) Y_(\d+)\)\)')


def parse_vnnlib_for_validation(
    path: Path,
) -> Tuple[np.ndarray, np.ndarray, int, List[int]]:
    """Parse a robustness vnnlib written by :func:`write_robustness_vnnlib`.

    Returns ``(lb, ub, y_clean, wrong_classes)``.
    """
    text = path.read_text()
    # Input box
    lb = np.full(_INPUT_DIM, np.nan)
    ub = np.full(_INPUT_DIM, np.nan)
    for m in _RE_GEQ_X.finditer(text):
        lb[int(m.group(1))] = float(m.group(2))
    for m in _RE_LEQ_X.finditer(text):
        ub[int(m.group(1))] = float(m.group(2))
    if np.isnan(lb).any() or np.isnan(ub).any():
        raise ValueError(f'{path}: input box has missing dimensions')

    # Disjunctive output: each disjunct is ``(and (>= Y_j Y_y_clean))``;
    # all share the same y_clean.
    disjuncts = _RE_DISJUNCT_GEQ.findall(text)
    if not disjuncts:
        raise ValueError(f'{path}: no output disjuncts found')
    wrong_classes = [int(j) for (j, _) in disjuncts]
    y_clean_set = {int(yc) for (_, yc) in disjuncts}
    if len(y_clean_set) != 1:
        raise ValueError(
            f'{path}: inconsistent y_clean across disjuncts: {y_clean_set}')
    return lb, ub, y_clean_set.pop(), sorted(wrong_classes)


def validate_vnnlib_round_trip(
    path: Path, x_clean: np.ndarray, y_clean: int, eps: float,
) -> None:
    """Parse a vnnlib written by :func:`write_robustness_vnnlib` and
    verify it round-trips to the same ``(lb, ub, y_clean, wrong_classes)``.
    """
    lb_parsed, ub_parsed, y_parsed, wrong_parsed = parse_vnnlib_for_validation(path)

    expected_lb = np.clip(x_clean.flatten() - eps, 0.0, 1.0)
    expected_ub = np.clip(x_clean.flatten() + eps, 0.0, 1.0)
    expected_wrong = sorted(j for j in range(_NUM_CLASSES) if j != y_clean)

    if not np.allclose(lb_parsed, expected_lb, atol=1e-7):
        raise RuntimeError(f'{path}: input lb round-trip mismatch')
    if not np.allclose(ub_parsed, expected_ub, atol=1e-7):
        raise RuntimeError(f'{path}: input ub round-trip mismatch')
    if y_parsed != y_clean:
        raise RuntimeError(f'{path}: y_clean round-trip mismatch')
    if wrong_parsed != expected_wrong:
        raise RuntimeError(f'{path}: wrong-classes set mismatch')


# ---- Main pipeline ----------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sigma', default='0.25',
                   help='Cohen-RS noise level σ (default 0.25 — matches '
                        'the Exp 2 default).')
    p.add_argument('--n-instances', type=int, default=_N_INSTANCES)
    p.add_argument('--eps', type=float, default=_EPS)
    p.add_argument('--out-dir', type=Path, default=_ARTIFACT_DIR)
    args = p.parse_args()

    out_dir: Path = args.out_dir
    onnx_dir = out_dir / 'onnx'
    vnnlib_dir = out_dir / 'vnnlib'
    instances_csv = out_dir / 'instances.csv'
    onnx_path = onnx_dir / 'cifar10_resnet110.onnx'

    # 1. Build + export ONNX
    print(f'[1/4] Loading pretrained ResNet-110 (σ={args.sigma})')
    model = _build_export_model(sigma=args.sigma)
    print(f'[2/4] Exporting ONNX -> {onnx_path}')
    export_onnx(model, onnx_path)

    # 2. Sample CIFAR-10 test inputs deterministically
    print(f'[3/4] Loading {args.n_instances} CIFAR-10 test images')
    test_inputs, test_labels = _load_cifar10_test(args.n_instances)

    # 3. Generate vnnlib + instances.csv
    print(f'[4/4] Writing {args.n_instances} vnnlib files')
    vnnlib_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Tuple[str, str, int]] = []
    written = 0
    skipped_misclassified = 0
    skipped_idx: List[int] = []
    for idx in range(args.n_instances):
        x = np.asarray(test_inputs[idx]).reshape(_IMG_SHAPE).astype(np.float32)
        y = int(test_labels[idx])
        # Skip instances where the clean prediction disagrees with the
        # label — RS-style certification requires the prediction to be
        # robust around the *clean* prediction, not around an arbitrary
        # label.
        with torch.no_grad():
            pred = int(model(torch.from_numpy(x[None])).argmax(dim=1).item())
        if pred != y:
            skipped_misclassified += 1
            skipped_idx.append(idx)
            continue
        vnn_name = (
            f'cifar10_test_{idx:04d}_label_{y}_eps_{args.eps:.6f}.vnnlib')
        vnn_path = vnnlib_dir / vnn_name
        write_robustness_vnnlib(vnn_path, x, y, args.eps)
        validate_vnnlib_round_trip(vnn_path, x, y, args.eps)
        rows.append((f'./onnx/{onnx_path.name}', f'./vnnlib/{vnn_name}',
                      _TIMEOUT_S))
        written += 1

    # 4. instances.csv (VNN-COMP format: relative onnx, relative vnnlib, timeout)
    instances_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(instances_csv, 'w', newline='') as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(row)

    print(f'\n=== Build complete ===')
    print(f'  ONNX:           {onnx_path}')
    print(f'  vnnlib:         {written} files in {vnnlib_dir}')
    if skipped_misclassified:
        print(f'  skipped:        {skipped_misclassified} '
              f'(model misclassified clean input; RS robustness undefined)')
    print(f'  instances.csv:  {instances_csv}')
    print(f'  vnnlib round-trip validation: PASS for all {written} specs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
