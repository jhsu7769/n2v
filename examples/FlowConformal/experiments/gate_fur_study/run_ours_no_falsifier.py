"""Gate-only ours runner — disables APGD pre-step on SAT-ground-truth instances.

This is the stress-test counterpart of ``exp{1,2}_run_ours.py``. It:

* Filters instances to those whose VNN-COMP 2025 consensus verdict is SAT.
* Forces ``use_falsifier=False`` so the gate is exposed to every instance.
* Otherwise reuses the same per-benchmark cfg, loader, and verification
  pipeline as the production runners.

The expected verdict on every instance is **UNKNOWN** (correct
abstention). A verdict of **UNSAT** is a *false UNSAT* — the gate
silently certified a region the network actually crosses. A verdict of
**SAT** without a falsifier means the AMLS-bounded sampler itself
detected an unsafe sample (``amls_bounded_detected_unsafe=True``); we
treat that as a correct verdict, just by a different mechanism.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.gate_fur_study.run_ours_no_falsifier \\
        --benchmark acasxu_2023 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.gate_fur_study._common import (
    GATE_FUR_BENCHMARKS,
    benchmark_dispatch,
    list_sat_instances,
)
from n2v.probabilistic.verify_flow import run_verification_pipeline

_SEED = 47
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'ground_truth', 'verdict',
    'wall_s', 'train_s', 'verify_s',
    'vnncomp_timeout_s', 'coverage', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_bounded_detected_unsafe',
    'amls_levels_used',
    'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def _run_one_instance(benchmark: str, onnx_rel: str, vnn_rel: str,
                      cfg: dict, *, seed: int) -> dict:
    _, _, _, load_one_instance = benchmark_dispatch(benchmark)
    try:
        network, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'{type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'loadfailed {type(e).__name__}: {e}'}

    if torch.cuda.is_available():
        try:
            network = network.cuda()
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'gpu_move {type(e).__name__}: {e}'}

    # Gate-only override: ignore cfg['use_falsifier'] (always False here).
    pipeline_kwargs = dict(
        flow_config=cfg['flow_config'],
        n_train=cfg['n_train'],
        flow_epochs=cfg['flow_epochs'],
        scenario_n_samples=cfg['scenario_n_samples'],
        scenario_beta=0.001,
        verification_method=cfg['verification_method'],
        amls_max_levels=cfg['amls_max_levels'],
        use_falsifier=False,
    )

    box_results = []
    for box_idx, (lb, ub) in enumerate(boxes):
        try:
            r = run_verification_pipeline(
                network=network,
                input_lb=lb, input_ub=ub, spec=spec,
                alpha=cfg['alpha'],
                seed=seed,
                **pipeline_kwargs,
            )
        except NotImplementedError as e:
            return {'verdict': 'SKIPPED',
                    'error': f'{type(e).__name__}: {e}'}
        except TimeoutError:
            raise
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'runfailed box={box_idx} {type(e).__name__}: {e}'}
        box_results.append(r)
        if r['verdict'] == 'SAT':
            break

    verdicts = [r['verdict'] for r in box_results]
    if 'SAT' in verdicts:
        result = next(r for r in box_results if r['verdict'] == 'SAT')
    elif all(v == 'UNSAT' for v in verdicts):
        result = box_results[0]
        if len(box_results) > 1:
            eps_sum = sum(
                (r.get('epsilon_total') or 0.0) for r in box_results)
            delta_min = min(
                (r.get('delta_total') or 1.0) for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')

    cex_x, cex_y = '', ''
    if result.get('counterexample') is not None:
        ce = result['counterexample']
        cex_x = json.dumps(np.asarray(ce['x']).tolist())
        cex_y = json.dumps(np.asarray(ce['y']).tolist())
    amls_lvls = result.get('amls_levels_used')
    return {
        'verdict': result['verdict'],
        'wall_s': _fmt(result.get('total_time_s'), '.1f'),
        'train_s': _fmt(result.get('flow_train_time_s'), '.1f'),
        'verify_s': _fmt(result.get('verification_time_s'), '.1f'),
        'coverage': _fmt(result.get('coverage_empirical'), '.4f'),
        'q': _fmt(result.get('q'), '.4f'),
        'epsilon_total': _fmt(result.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(result.get('delta_total'), '.4f'),
        'amls_bounded_eps_2_upper': _fmt(
            result.get('amls_bounded_eps_2_upper'), '.4e'),
        'amls_bounded_detected_unsafe': str(
            result.get('amls_bounded_detected_unsafe', '')),
        'amls_levels_used': str(amls_lvls) if amls_lvls is not None else '',
        'cex_x': cex_x,
        'cex_y': cex_y,
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
            'vnncomp_timeout_s': vnncomp_t,
            'error': 'shell timeout (run_cell.sh exit 124)',
            'timestamp': _now_iso(),
        })
        writer.writerow(out_row)
        f.flush()


def _run_and_write(out_csv: Path, benchmark: str, cfg: dict,
                   instances: list, *, append: bool) -> dict:
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'SKIPPED': 0, 'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    mode = 'a' if append and file_exists else 'w'
    with open(out_csv, mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or mode == 'w':
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
                    benchmark, onnx_rel, vnn_rel, cfg, seed=_SEED)
            except Exception as e:
                row = {'verdict': 'ERROR',
                       'error': f'{type(e).__name__}: {e}'}

            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'onnx_file': Path(onnx_rel).name,
                'vnnlib_file': Path(vnn_rel).name,
                'ground_truth': 'SAT',
                'vnncomp_timeout_s': vnncomp_t,
                'timestamp': _now_iso(),
            })
            out_row.update(row)
            writer.writerow(out_row)
            f.flush()

            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  '
                  f'wall={time.time()-t0:.1f}s', flush=True)
    return counts


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
    _, cfg, _, _ = benchmark_dispatch(benchmark)
    instances = list_sat_instances(benchmark)

    if args.list_instances:
        for idx, (_o, _v, vnncomp_t) in enumerate(instances):
            print(f'{idx} {vnncomp_t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'gate_fur_{benchmark}_ours.csv')
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

    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        if not (0 <= args.instance_idx < len(instances)):
            print(f'--instance-idx {args.instance_idx} out of range '
                  f'[0, {len(instances)})', file=sys.stderr)
            sys.exit(2)
        only = [instances[args.instance_idx]]
        print(f'[{benchmark}] running only SAT-instance idx='
              f'{args.instance_idx}; appending to {out_csv}', flush=True)
        _run_and_write(out_csv, benchmark, cfg, only, append=True)
        return

    if args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first SAT instance',
              flush=True)
    if not instances:
        print(f'[{benchmark}] no SAT-ground-truth instances; nothing to do',
              flush=True)
        return
    print(f'[{benchmark}] Loaded {len(instances)} SAT instances; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark}] gate-only cfg: flow_config={cfg["flow_config"]} '
          f'n_train={cfg["n_train"]} flow_epochs={cfg["flow_epochs"]} '
          f'alpha={cfg["alpha"]} method={cfg["verification_method"]} '
          f'use_falsifier=False (gate-only) SEED={_SEED}', flush=True)

    t_start = time.time()
    counts = _run_and_write(out_csv, benchmark, cfg, instances, append=False)

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
