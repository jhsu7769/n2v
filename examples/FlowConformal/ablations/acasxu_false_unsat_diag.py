"""Diagnose Phase 4 false-UNSAT (and UNKNOWN x abc-SAT) cases by
comparing each alpha,beta-CROWN counterexample's flow score
``||phi(y_cex)||`` against the calibrated ``q``.

Output classification:
  - 'calibration miss' (score > q): real cex is OUTSIDE our calibrated
    S_flow. The conformal coverage claim doesn't hold on this
    adversarial input.
  - 'sampling miss'    (score <= q): real cex is INSIDE S_flow but our
    2000 scenario samples missed it. Calibration is correct but
    sampling is too sparse.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.ablations.acasxu_false_unsat_diag

Reads:
  - examples/FlowConformal/ablations/outputs/acasxu_sweep_flow_conformal.csv
  - ~/v/other/VNNCOMP/vnncomp2025_results/alpha_beta_crown/2025_acasxu_2023/results.csv
  - ~/v/other/VNNCOMP/vnncomp2025_results/alpha_beta_crown/2025_acasxu_2023/*.counterexample.gz

Writes:
    examples/FlowConformal/ablations/outputs/acasxu_false_unsat_diag.csv
"""
from __future__ import annotations

import csv
import gzip
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import (
    _train_flow, _train_flow_tight,
)
from examples.FlowConformal.benchmarks.test_acasxu_single import _ACASXuWrapper
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.scores import FlowScore
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ABC_DIR = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_results/alpha_beta_crown/'
    '2025_acasxu_2023'
))
_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_FLOW_CSV = Path(__file__).parent / 'outputs' / 'acasxu_sweep_flow_conformal.csv'
_OUT_CSV = Path(__file__).parent / 'outputs' / 'acasxu_false_unsat_diag.csv'


def _load_abc_results() -> dict:
    """Return dict[(onnx, vnn)] -> verdict from alpha,beta-CROWN's CSV.

    Format (no header): benchmark, onnx_path, vnnlib_path, total_time,
    verdict, solve_time.
    """
    out = {}
    with open(_ABC_DIR / 'results.csv') as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            onnx = Path(row[1]).name
            vnn = Path(row[2]).name
            out[(onnx, vnn)] = row[4].strip().lower()
    return out


def _parse_abc_counterexample(text: str) -> np.ndarray | None:
    """Parse alpha,beta-CROWN's plain-text VNN-LIB-style cex format.

    Looks for ``(X_i  value)`` lines and returns a numpy array of values
    in order. Returns None if no X_i lines were found.
    """
    pat = re.compile(r'\(\s*X_(\d+)\s+(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*\)')
    matches = pat.findall(text)
    if not matches:
        return None
    pairs = [(int(idx), float(val)) for idx, val in matches]
    pairs.sort(key=lambda p: p[0])
    n_dims = pairs[-1][0] + 1
    if len(pairs) != n_dims:
        return None
    return np.array([v for _, v in pairs], dtype=np.float32)


def _load_abc_counterexample(onnx_name: str, vnn_name: str) -> np.ndarray | None:
    """Load alpha,beta-CROWN's counterexample for an instance, or None
    if no file exists or the file is empty / unparseable."""
    stem = onnx_name.replace('.onnx', '') + '_' + vnn_name.replace('.vnnlib', '')
    cex_path = _ABC_DIR / f'{stem}.counterexample.gz'
    if not cex_path.exists():
        return None
    try:
        with gzip.open(cex_path, 'rt') as f:
            text = f.read()
    except Exception:
        return None
    return _parse_abc_counterexample(text)


def _per_instance_seed(onnx_rel: str, vnn_rel: str) -> int:
    """Match the per-instance seed used by acasxu_sweep.py."""
    return hash((onnx_rel, vnn_rel)) & 0x7FFFFFFF


def _diagnose_one(
    onnx_name: str,
    vnn_name: str,
    x_cex: np.ndarray,
    n_train: int = 5000,
    flow_epochs: int = 2000,
    flow_config: str = 'base',
    m: int = 8000,
    ell: int = 7999,
) -> dict:
    """Train the flow with Phase 4 config, calibrate q, then compute the
    score for the alpha,beta-CROWN counterexample's output."""
    seed = _per_instance_seed('onnx/' + onnx_name, 'vnnlib/' + vnn_name)

    onnx_path = _ACASXU_ROOT / 'onnx' / onnx_name
    vnn_path = _ACASXU_ROOT / 'vnnlib' / vnn_name
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))

    # Handle OR-of-input-regions (e.g. prop_6) defensively
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        return {
            'onnx': onnx_name, 'vnn': vnn_name,
            'q_calibrated': float('nan'),
            'score_at_cex': float('nan'),
            'classification': 'OR-of-input-regions',
            'cex_in_input_box': False,
        }
    lb = np.asarray(prop['lb'], dtype=np.float32).flatten()
    ub = np.asarray(prop['ub'], dtype=np.float32).flatten()

    # Sanity check: the alpha-beta-crown cex should lie in the input box
    cex_in_box = bool(np.all(x_cex >= lb - 1e-5) and np.all(x_cex <= ub + 1e-5))

    # 1. Sample training data, calibration data
    rng = np.random.default_rng(seed)
    x_tr = torch.from_numpy(
        rng.uniform(lb, ub, size=(n_train, len(lb))).astype(np.float32)
    )
    rng_ca = np.random.default_rng(seed + 1_000_000)
    x_ca = torch.from_numpy(
        rng_ca.uniform(lb, ub, size=(m, len(lb))).astype(np.float32)
    )
    with torch.no_grad():
        y_tr = network(x_tr)
        y_ca = network(x_ca)

    # 2. Whiten
    y_mean = y_tr.mean(dim=0)
    y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    y_ca_w = (y_ca - y_mean) / y_std

    # 3. Train flow (same config as Phase 4)
    if flow_config == 'base':
        flow = _train_flow(
            y_tr_w, y_tr_w.shape[1], flow_epochs, seed,
            internal_standardize=False,
        )
    else:
        flow = _train_flow_tight(
            y_tr_w, y_tr_w.shape[1], flow_epochs, seed,
            internal_standardize=False,
        )
    flow = flow.to('cpu').eval()

    # 4. Calibrate
    score_fn = FlowScore(
        flow, t=1.0, n_steps=30, method='rk4', batch_size=65536,
    )
    calib_scores = score_fn(y_ca_w)
    q = calibrate(calib_scores, ell).item()

    # 5. Forward the cex through the network and compute the score
    with torch.no_grad():
        y_cex = network(
            torch.as_tensor(x_cex.reshape(1, -1), dtype=torch.float32)
        )
    y_cex_w = (y_cex - y_mean) / y_std
    score_at_cex = score_fn(y_cex_w).item()

    classification = 'calibration miss' if score_at_cex > q else 'sampling miss'

    return {
        'onnx': onnx_name, 'vnn': vnn_name,
        'q_calibrated': q,
        'score_at_cex': score_at_cex,
        'score_over_q_ratio': score_at_cex / q if q > 0 else float('nan'),
        'classification': classification,
        'cex_in_input_box': cex_in_box,
    }


def main():
    abc = _load_abc_results()

    flow_rows = {}
    with open(_FLOW_CSV) as f:
        for r in csv.DictReader(f):
            flow_rows[(r['onnx_file'], r['vnnlib_file'])] = r

    fail_cases = []
    for k, r in flow_rows.items():
        abc_v = abc.get(k, '')
        if r['verdict'] == 'UNSAT' and abc_v == 'sat':
            fail_cases.append((k, 'UNSAT (false)', abc_v))
        elif r['verdict'] == 'UNKNOWN' and abc_v == 'sat':
            fail_cases.append((k, 'UNKNOWN', abc_v))

    print(f'Diagnostic on {len(fail_cases)} failure cases.')
    print()

    out_rows = []
    for (onnx, vnn), flow_v, abc_v in fail_cases:
        x_cex = _load_abc_counterexample(onnx, vnn)
        if x_cex is None:
            print(f'  SKIP {onnx} / {vnn}: no abc counterexample file')
            out_rows.append({
                'onnx': onnx, 'vnn': vnn,
                'flow_verdict': flow_v, 'abc_verdict': abc_v,
                'q_calibrated': '', 'score_at_cex': '',
                'score_over_q_ratio': '',
                'classification': 'no cex file',
                'cex_in_input_box': '',
            })
            continue

        print(f'  diagnosing {onnx} / {vnn} (flow={flow_v}, abc={abc_v}) ...',
              flush=True)
        t0 = time.time()
        try:
            r = _diagnose_one(onnx, vnn, x_cex)
        except Exception as e:
            print(f'    error: {type(e).__name__}: {e}', file=sys.stderr)
            r = {
                'onnx': onnx, 'vnn': vnn,
                'q_calibrated': float('nan'), 'score_at_cex': float('nan'),
                'score_over_q_ratio': float('nan'),
                'classification': f'error: {type(e).__name__}',
                'cex_in_input_box': False,
            }
        r['flow_verdict'] = flow_v
        r['abc_verdict'] = abc_v
        out_rows.append(r)
        print(f'    q={r["q_calibrated"]:.4f} score={r["score_at_cex"]:.4f} '
              f'ratio={r["score_over_q_ratio"]:.4f} -> {r["classification"]} '
              f'({time.time()-t0:.0f}s)')

    _OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not out_rows:
        with open(_OUT_CSV, 'w') as f:
            f.write('no failure cases\n')
        return
    fields = ['onnx', 'vnn', 'flow_verdict', 'abc_verdict',
              'q_calibrated', 'score_at_cex', 'score_over_q_ratio',
              'classification', 'cex_in_input_box']
    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in out_rows:
            writer.writerow({k: r.get(k, '') for k in fields})

    n_calib = sum(1 for r in out_rows if r.get('classification') == 'calibration miss')
    n_samp = sum(1 for r in out_rows if r.get('classification') == 'sampling miss')
    n_other = len(out_rows) - n_calib - n_samp
    print(f'\nSummary: {n_calib} calibration miss / {n_samp} sampling miss / '
          f'{n_other} other')
    print(f'Wrote {_OUT_CSV}')


if __name__ == '__main__':
    main()
