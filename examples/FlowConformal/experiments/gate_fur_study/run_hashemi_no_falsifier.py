"""Gate-only Hashemi-clipping runner — disables APGD pre-step on SAT instances.

Counterpart of ``exp{1,2}_run_hashemi_clipping.py`` for the gate-FUR
study. Filters to SAT-ground-truth instances, runs the
m=8000 ``clipping_block`` calibration without the APGD pre-step, and
records the verdict.

Verdict semantics (no falsifier):

* **UNSAT**: ``halfspace_disjoint_from_box`` certified the calibrated
  ``ProbabilisticBox`` is fully outside the unsafe halfspace in every
  group. On a SAT-ground-truth instance this is a *false UNSAT*.
* **UNKNOWN**: at least one group's calibrated box overlaps the unsafe
  halfspace; correct abstention.
* **SAT** is impossible without a falsifier (no random sampling either —
  Hashemi-clipping itself produces no CEX).
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

from examples.FlowConformal.experiments.baselines._common import (
    empirical_coverage_for_box,
    halfspace_disjoint_from_box,
    torch_callable,
)
from examples.FlowConformal.experiments.gate_fur_study._common import (
    GATE_FUR_BENCHMARKS,
    benchmark_dispatch,
    hashemi_config,
    list_sat_instances,
)
from n2v.probabilistic import verify
from n2v.sets import Box

_SEED = 47
_M = 8000
_EPSILON = 0.001
_ELL = _M - 1  # second-largest score, Hashemi default
_N_TEST_COVERAGE = 1000
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'ground_truth', 'verdict',
    'wall_s', 'vnncomp_timeout_s', 'm', 'ell', 'epsilon',
    'pca_components', 'coverage',
    'coverage_empirical', 'coverage_n_test', 'confidence',
    'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _run_one_instance(benchmark: str, onnx_rel: str, vnn_rel: str,
                      *, seed: int) -> Dict[str, Any]:
    _, _, _, load_one_instance = benchmark_dispatch(benchmark)
    try:
        network, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'unsupported_spec {type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'load {type(e).__name__}: {e}'}

    try:
        network = network.cuda()
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'gpu_move {type(e).__name__}: {e}'}

    model_fn = torch_callable(network)
    any_unknown = False
    cov_vals: list = []
    cov_n_total = 0
    last_pbox = None

    # Look up the per-benchmark Hashemi config so the gate-FUR study
    # mirrors the Hashemi configuration the production sweep uses for
    # this benchmark (raw clipping_block vs. PCA-augmented).
    hcfg = hashemi_config(benchmark)
    pca_components = hcfg.get('pca_components')

    for box_idx, (lb, ub) in enumerate(boxes):
        input_set = Box(np.asarray(lb).flatten(),
                        np.asarray(ub).flatten())

        # NO FALSIFIER. Gate-only: skip the Stage-1 APGD pre-step.
        try:
            verify_kwargs = dict(
                model=model_fn,
                input_set=input_set,
                m=_M, ell=_ELL,
                epsilon=_EPSILON,
                surrogate='clipping_block',
                seed=seed,
                verbose=False,
            )
            if pca_components is not None:
                verify_kwargs['pca_components'] = pca_components
            pbox = verify(**verify_kwargs)
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'verify box={box_idx} {type(e).__name__}: {e}'}
        last_pbox = pbox

        try:
            cov, _sigma, n_eff = empirical_coverage_for_box(
                model_fn=model_fn,
                input_lb=input_set.lb, input_ub=input_set.ub,
                box_lb=pbox.lb, box_ub=pbox.ub,
                n_test=_N_TEST_COVERAGE,
                seed=seed,
            )
            if not np.isnan(cov):
                cov_vals.append(cov)
                cov_n_total += n_eff
        except Exception:
            pass

        disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
        if disjoint is True:
            continue
        any_unknown = True

    cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
    verdict = 'UNKNOWN' if any_unknown else 'UNSAT'

    return {
        'verdict': verdict,
        'm': _M, 'ell': _ELL, 'epsilon': _EPSILON,
        'pca_components': (
            pca_components if pca_components is not None else ''),
        'coverage': (
            f'{last_pbox.coverage:.4f}' if last_pbox is not None else ''),
        'coverage_empirical': (
            f'{cov_emp:.4f}' if not np.isnan(cov_emp) else ''),
        'coverage_n_test': cov_n_total,
        'confidence': (
            f'{last_pbox.confidence:.4f}' if last_pbox is not None else ''),
        'error': '',
    }


def _write_timeout_row(out_csv: Path, benchmark: str,
                       onnx_rel: str, vnn_rel: str,
                       vnncomp_t: int) -> None:
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    with open(out_csv, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader()
            f.flush()
        out_row = {_f: '' for _f in _FIELDS}
        out_row.update({
            'benchmark': benchmark,
            'onnx_file': Path(onnx_rel).name,
            'vnnlib_file': Path(vnn_rel).name,
            'ground_truth': 'SAT',
            'verdict': 'TIMEOUT',
            'wall_s': '',
            'vnncomp_timeout_s': vnncomp_t,
            'error': 'shell timeout (run_cell.sh exit 124)',
            'timestamp': _now_iso(),
        })
        writer.writerow(out_row)
        f.flush()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=GATE_FUR_BENCHMARKS)
    p.add_argument('--smoke', action='store_true',
                   help='Run only the first SAT instance (no ground-truth assert).')
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only SAT-instance index (0-based) and append.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <vnncomp_timeout_s>" for each SAT instance.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row at --instance-idx.')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark
    instances = list_sat_instances(benchmark)

    if args.list_instances:
        for idx, (_o, _v, vnncomp_t) in enumerate(instances):
            print(f'{idx} {vnncomp_t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'gate_fur_{benchmark}_hashemi_clipping.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, onnx_rel, vnn_rel, vnncomp_t)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(instances)})', file=sys.stderr)
            sys.exit(2)
        instances = [instances[args.instance_idx]]
        append_mode = True
        print(f'[{benchmark}] running only SAT-instance idx='
              f'{args.instance_idx}; appending to {out_csv}', flush=True)
    elif args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first SAT instance',
              flush=True)

    if not instances:
        print(f'[{benchmark}] no SAT-ground-truth instances; nothing to do',
              flush=True)
        return

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    print(f'[{benchmark}] Loaded {len(instances)} SAT instances; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark}] gate-only Hashemi-clipping cfg: m={_M} ell={_ELL} '
          f'epsilon={_EPSILON} use_falsifier=False SEED={_SEED}', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'SKIPPED': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, (onnx_rel, vnn_rel, vnncomp_t) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            timeout_s = vnncomp_t if vnncomp_t > 0 else 600
            print(f'[{benchmark} {k}/{len(instances)} t={elapsed:.0f}s '
                  f'budget={timeout_s}s] {onnx_rel} + {vnn_rel}',
                  flush=True)
            t0 = time.time()
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                row = _run_one_instance(
                    benchmark, onnx_rel, vnn_rel, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'onnx_file': Path(onnx_rel).name,
                'vnnlib_file': Path(vnn_rel).name,
                'ground_truth': 'SAT',
                'wall_s': f'{wall_s:.1f}',
                'vnncomp_timeout_s': vnncomp_t,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            writer.writerow(out_row)
            f.flush()

            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={wall_s:.1f}s',
                  flush=True)

    print('\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    n_total = sum(counts.values()) - counts.get('SKIPPED', 0) - counts.get('ERROR', 0)
    n_false_unsat = counts.get('UNSAT', 0)
    if n_total > 0:
        fur = n_false_unsat / n_total
        print(f'Gate-only FUR on {benchmark}: {n_false_unsat}/{n_total} = {fur:.4f}')


if __name__ == '__main__':
    main()
