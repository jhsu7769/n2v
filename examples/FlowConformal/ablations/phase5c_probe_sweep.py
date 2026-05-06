"""Probe sweep: 20 representative ACAS Xu instances with custom kwargs.

Used by Phase 5c B-step iterations (B1.1, B1.2, B2.1, B2.2, A3, B3.1/2/3).
Each invocation produces phase5c_probe_<tag>.csv with verdict, q, margin
per instance, for fast feedback after each tuning step.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.ablations.phase5c_probe_sweep \\
        --tag B1.1 --adaptive-threshold 0.3 --adaptive-n-samples 30000
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_OUT_DIR = Path(__file__).parent / 'outputs'

_INSTANCES = [
    # 4 persistent false UNSATs from Phase 5b
    ('onnx/ACASXU_run2a_1_2_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_1_5_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_1_6_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_1_9_batch_2000.onnx', 'vnnlib/prop_7.vnnlib'),
    # 5 cases that changed P4 -> P5b (regression sensitivity)
    ('onnx/ACASXU_run2a_1_3_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_1_4_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_3_2_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_5_3_batch_2000.onnx', 'vnnlib/prop_2.vnnlib'),
    ('onnx/ACASXU_run2a_2_9_batch_2000.onnx', 'vnnlib/prop_8.vnnlib'),
    # 11 typical UNSAT controls
    ('onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
    ('onnx/ACASXU_run2a_2_3_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
    ('onnx/ACASXU_run2a_3_5_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
    ('onnx/ACASXU_run2a_4_7_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
    ('onnx/ACASXU_run2a_5_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib'),
    ('onnx/ACASXU_run2a_2_5_batch_2000.onnx', 'vnnlib/prop_3.vnnlib'),
    ('onnx/ACASXU_run2a_3_7_batch_2000.onnx', 'vnnlib/prop_3.vnnlib'),
    ('onnx/ACASXU_run2a_4_9_batch_2000.onnx', 'vnnlib/prop_3.vnnlib'),
    ('onnx/ACASXU_run2a_2_7_batch_2000.onnx', 'vnnlib/prop_4.vnnlib'),
    ('onnx/ACASXU_run2a_3_9_batch_2000.onnx', 'vnnlib/prop_4.vnnlib'),
    ('onnx/ACASXU_run2a_5_5_batch_2000.onnx', 'vnnlib/prop_4.vnnlib'),
]

_BASE_KWARGS = dict(
    alpha=0.001,
    n_train=5_000, flow_epochs=2_000, flow_config='base',
    scenario_n_samples=2_000, scenario_beta=0.001,
)


def _load_instance(onnx_rel: str, vnn_rel: str):
    onnx_path = _ACASXU_ROOT / onnx_rel.removeprefix('./')
    vnn_path = _ACASXU_ROOT / vnn_rel.removeprefix('./')
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError('OR-of-input-regions not supported')
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    spec = _extract_spec(prop['prop'])
    return network, input_lb, input_ub, spec


def _extract_worst_max_margin(result: dict) -> 'float | None':
    """Pull the worst (min) max-row-margin across all per-group / per-hs
    results in the scenario_result. Positive means certified disjoint;
    negative means a flow sample fell inside the unsafe region.

    Returns None when scenario step did not run (SAT short-circuit) or
    structure is empty.
    """
    sr = result.get('scenario_result')
    if sr is None:
        return None
    per_group = sr.get('per_group_results') or []
    margins = []
    for group_res in per_group:
        per_hs = getattr(group_res, 'per_hs_results', None) or []
        for hs_res in per_hs:
            wm = getattr(hs_res, 'worst_max_margin', None)
            if wm is not None:
                margins.append(float(wm))
    if not margins:
        return None
    return min(margins)


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--tag', required=True,
                   help='label for output filename, e.g. B1.1')
    p.add_argument('--alpha', type=float)
    p.add_argument('--n-calib', type=int)
    p.add_argument('--n-train', type=int)
    p.add_argument('--flow-epochs', type=int)
    p.add_argument('--scenario-n', type=int)
    p.add_argument('--adaptive-threshold', type=float, default=0.5)
    p.add_argument('--adaptive-n-samples', type=int, default=20_000)
    p.add_argument('--flow-ensemble-size', type=int, default=1)
    p.add_argument('--sampling-strategy', type=str, default='uniform')
    p.add_argument('--verification-method', type=str, default='scenario')
    args = p.parse_args()

    kwargs = dict(_BASE_KWARGS)
    if args.alpha is not None:
        kwargs['alpha'] = args.alpha
    if args.n_calib is not None:
        # ell = ceil((m+1)(1-alpha)); cap at m to keep <= m.
        alpha_for_ell = kwargs['alpha']
        kwargs['m'] = args.n_calib
        kwargs['ell'] = min(
            args.n_calib,
            int(math.ceil((args.n_calib + 1) * (1.0 - alpha_for_ell))),
        )
    if args.n_train is not None:
        kwargs['n_train'] = args.n_train
    if args.flow_epochs is not None:
        kwargs['flow_epochs'] = args.flow_epochs
    if args.scenario_n is not None:
        kwargs['scenario_n_samples'] = args.scenario_n
    kwargs['adaptive_threshold'] = args.adaptive_threshold
    kwargs['adaptive_n_samples'] = args.adaptive_n_samples
    if args.flow_ensemble_size > 1:
        kwargs['flow_ensemble_size'] = args.flow_ensemble_size
    if args.sampling_strategy != 'uniform':
        kwargs['sampling_strategy'] = args.sampling_strategy
    if args.verification_method != 'scenario':
        kwargs['verification_method'] = args.verification_method

    out_csv = _OUT_DIR / f'phase5c_probe_{args.tag}.csv'
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = ['onnx_file', 'vnnlib_file', 'verdict', 'q',
              'worst_max_margin', 'wall_s', 'error']

    print(f'Phase 5c probe sweep: tag={args.tag}', flush=True)
    print(f'  kwargs={kwargs}', flush=True)
    print(f'  instances={len(_INSTANCES)}', flush=True)
    print(f'  out={out_csv}', flush=True)

    t0 = time.time()
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
        for k, (onnx_rel, vnn_rel) in enumerate(_INSTANCES, start=1):
            inst_seed = hash((onnx_rel, vnn_rel)) & 0x7FFFFFFF
            print(f'[{k}/{len(_INSTANCES)}] '
                  f'{Path(onnx_rel).name}+{Path(vnn_rel).name}',
                  flush=True)
            t_inst = time.time()
            try:
                network, lb, ub, spec = _load_instance(onnx_rel, vnn_rel)
                res = run_verification_pipeline(
                    network=network, input_lb=lb, input_ub=ub, spec=spec,
                    seed=inst_seed, **kwargs,
                )
                wmm = _extract_worst_max_margin(res)
                row = {
                    'onnx_file': Path(onnx_rel).name,
                    'vnnlib_file': Path(vnn_rel).name,
                    'verdict': res['verdict'],
                    'q': _fmt(res.get('q'), '.6f'),
                    'worst_max_margin': _fmt(wmm, '.6f'),
                    'wall_s': f'{time.time()-t_inst:.1f}',
                    'error': '',
                }
            except Exception as e:
                row = {
                    'onnx_file': Path(onnx_rel).name,
                    'vnnlib_file': Path(vnn_rel).name,
                    'verdict': 'ERROR', 'q': '', 'worst_max_margin': '',
                    'wall_s': f'{time.time()-t_inst:.1f}',
                    'error': f'{type(e).__name__}: {e}',
                }
            w.writerow(row)
            f.flush()
            print(f'    verdict={row["verdict"]}  q={row["q"]}  '
                  f'wmm={row["worst_max_margin"]}  wall={row["wall_s"]}s',
                  flush=True)
    print(f'\nWrote {out_csv}  total wall {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
