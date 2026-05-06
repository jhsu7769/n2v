"""Full flow-conformal verification sweep on VNN-COMP ACAS Xu 2023.

Doubles as the canonical Exp 1 ACAS Xu (ours) runner under the
FlowConformal seeding convention (SEED=47 reset per-instance) and
VNN-COMP per-row timeouts. See
``examples/FlowConformal/experiments/README.md``.

Iterates the 186-instance instance list, calls run_verification_pipeline
with the scaling-study-locked base config, and writes a CSV that joins
1:1 with acasxu_sweep_deterministic.csv (by (onnx_file, vnnlib_file)).

OR-of-ANDs specs (prop_5-10 in ACAS Xu, 6 instances total) are marked
SKIPPED; the remaining 180 instances use Phase 2's single-HalfSpace
dispatcher.

Usage:
    cd /home/sasakis/v/tools/n2v

    # Smoke: run only instance 1 and assert UNSAT.
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.ablations.acasxu_sweep \\
        --verification-method amls_bounded --vnncomp-timeouts --smoke

    # Canonical Exp 1 sweep (~3 hr at base config + 116s VNN-COMP timeout):
    nohup /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.ablations.acasxu_sweep \\
        --verification-method amls_bounded --vnncomp-timeouts \\
        --output-csv examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_acasxu_2023_ours.csv \\
        > /tmp/exp1_acasxu_ours.log 2>&1 &
    disown

Output CSV columns:
    onnx_file, vnnlib_file, verdict, coverage, q,
    epsilon_total, delta_total, train_s, verify_s, total_s,
    amls_levels_used, cex_x, cex_y, error

For SKIPPED instances: verdict='SKIPPED', error contains the reason.
"""
from __future__ import annotations

import csv
import json
import os
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_VNNCOMP_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
))
_INSTANCES_CSV = _VNNCOMP_ROOT / 'instances.csv'
_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'acasxu_sweep_flow_conformal.csv'
# Default per-instance timeout for legacy/ablation runs. The Exp 1 ACAS Xu
# sweep honours the VNN-COMP per-row timeout from instances.csv column 3
# (116s for ACAS Xu) instead — see ``--vnncomp-timeouts``.
_PER_INSTANCE_TIMEOUT_S = 600

# >>> LOCKED CONFIG from scaling study (`acasxu_scaling_study.csv`) <<<
# base-fast: tightest q across all 6 probes (4.5 to 17.95), fastest
# training (~26s/instance), verdict correctness 6/6. base-med/-long
# were unstable (q occasionally exploded to 1e2-1e4); tight-* were
# stable but consistently looser-q than base-fast. See
# docs/audits/2026-04-24-phase3-acasxu-sweep-results.md §scaling study.
_FLOW_CONFIG = 'base'
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_ALPHA = 0.001
_VERIFICATION_METHOD = 'amls'  # Phase 5d: AMLS replaces uniform random scenario sampling
_SEED = 47  # FlowConformal canonical seed (see examples/FlowConformal/experiments/README.md §Seeding)


def _raise_timeout(signum, frame):
    raise TimeoutError()


def _load_instance(onnx_rel: str, vnn_rel: str):
    """Load and normalize one instance. Returns (network, boxes, spec) where
    `boxes` is a list of (lb, ub) input-region tuples (length 1 for the
    typical single-box case, length >1 for OR-of-input-regions like prop_6).
    Raises NotImplementedError if the output spec is OR-of-ANDs."""
    onnx_path = _ACASXU_ROOT / onnx_rel.removeprefix('./')
    vnn_path = _ACASXU_ROOT / vnn_rel.removeprefix('./')
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        # OR-of-input-regions (e.g. prop_6): the input region is a union of
        # disjoint boxes. Verify each box independently and Bonferroni-union.
        lbs, ubs = prop['lb'], prop['ub']
        boxes = [(np.asarray(lb).flatten(), np.asarray(ub).flatten())
                 for lb, ub in zip(lbs, ubs)]
    else:
        boxes = [(np.asarray(prop['lb']).flatten(),
                  np.asarray(prop['ub']).flatten())]
    spec = _extract_spec(prop['prop'])  # raises NotImplementedError on OR-of-ANDs
    return network, boxes, spec


def _run_one_instance(onnx_rel: str, vnn_rel: str, *, seed: int) -> dict:
    """Run the full verification pipeline on one instance. Returns a
    row dict; always returns (never raises) so the outer loop can log
    errors to CSV instead of aborting.

    ``seed`` is the FlowConformal canonical seed (`SEED=47`). The outer
    loop also resets ``torch.manual_seed`` and ``np.random.seed`` to
    ``seed`` before calling this, so each instance starts from the
    same RNG state regardless of execution order.
    """
    try:
        network, boxes, spec = _load_instance(onnx_rel, vnn_rel)
    except NotImplementedError as e:
        return {
            'verdict': 'SKIPPED',
            'error': f'{type(e).__name__}: {e}',
        }
    except Exception as e:
        return {
            'verdict': 'ERROR',
            'error': f'loadfailed {type(e).__name__}: {e}',
        }

    # Run pipeline once per input box. For OR-of-input-regions (e.g. prop_6),
    # boxes contains > 1 tuple and we Bonferroni-aggregate verdicts: any SAT
    # → SAT (cex on the union exists); all UNSAT → UNSAT; else UNKNOWN.
    box_results = []
    for box_idx, (lb, ub) in enumerate(boxes):
        try:
            r = run_verification_pipeline(
                network=network,
                input_lb=lb, input_ub=ub, spec=spec,
                alpha=_ALPHA,
                n_train=_N_TRAIN, flow_epochs=_FLOW_EPOCHS,
                flow_config=_FLOW_CONFIG,
                scenario_n_samples=_SCENARIO_N, scenario_beta=0.001,
                verification_method=_VERIFICATION_METHOD,
                seed=seed,  # SEED=47 globally — no sub-seeding
            )
        except NotImplementedError as e:
            return {
                'verdict': 'SKIPPED',
                'error': f'{type(e).__name__}: {e}',
            }
        except Exception as e:
            return {
                'verdict': 'ERROR',
                'error': f'runfailed box={box_idx} {type(e).__name__}: {e}',
            }
        box_results.append(r)
        # Short-circuit on SAT (cex found on this box → cex on the union)
        if r['verdict'] == 'SAT':
            break

    # Aggregate verdicts across boxes
    verdicts = [r['verdict'] for r in box_results]
    if 'SAT' in verdicts:
        result = next(r for r in box_results if r['verdict'] == 'SAT')
    elif all(v == 'UNSAT' for v in verdicts):
        # All boxes certified UNSAT independently. The total certificate uses
        # Bonferroni: epsilon_total_union = sum(epsilon_total_i), delta_total
        # likewise unions over confidence-failure events.
        result = box_results[0]  # for simplicity, report the first; aggregate
        # epsilon/delta if multiple boxes
        if len(box_results) > 1:
            eps_sum = sum((r.get('epsilon_total') or 0.0) for r in box_results)
            delta_min = min((r.get('delta_total') or 1.0) for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        # Any UNKNOWN means the union verdict is UNKNOWN
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')

    # Format helper: print '' for None, else format. Phase 5's falsify-
    # first pipeline returns None for flow / certificate fields on SAT
    # verdicts (where flow training never ran). Bare f-strings on None
    # raise TypeError, so every SAT row in the CSV needs this.
    def _fmt(v, spec):
        return f'{v:{spec}}' if v is not None else ''

    cex_x, cex_y = '', ''
    if result['counterexample'] is not None:
        ce = result['counterexample']
        cex_x = json.dumps(ce['x'].tolist())
        cex_y = json.dumps(ce['y'].tolist())
    amls_lvls = result.get('amls_levels_used')
    return {
        'verdict': result['verdict'],
        'coverage': _fmt(result.get('coverage_empirical'), '.4f'),
        'q': _fmt(result.get('q'), '.4f'),
        'epsilon_total': _fmt(result.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(result.get('delta_total'), '.4f'),
        'train_s': _fmt(result.get('flow_train_time_s'), '.1f'),
        'verify_s': _fmt(result.get('verification_time_s'), '.1f'),
        'total_s': _fmt(result.get('total_time_s'), '.1f'),
        'amls_levels_used': str(amls_lvls) if amls_lvls is not None else '',
        'cex_x': cex_x, 'cex_y': cex_y,
        'error': '',
    }


def main():
    # CLI overrides for verification method, output path, timeout policy,
    # and smoke mode. Defaults match Phase 5d behavior; the canonical
    # Exp 1 ACAS Xu sweep uses ``--verification-method amls_bounded
    # --vnncomp-timeouts``.
    global _VERIFICATION_METHOD, _OUT_CSV  # noqa: PLW0603
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--verification-method', dest='verification_method',
                   default=_VERIFICATION_METHOD,
                   choices=('scenario', 'scenario_v2', 'amls',
                            'amls_bounded', 'amls_bounded_union',
                            'is_tilted', 'derived'))
    p.add_argument('--output-csv', dest='output_csv', type=Path,
                   default=_OUT_CSV)
    p.add_argument('--vnncomp-timeouts', dest='vnncomp_timeouts',
                   action='store_true',
                   help='Use the VNN-COMP per-row timeout from '
                        'instances.csv column 3 (116s for ACAS Xu) '
                        f'instead of the fixed {_PER_INSTANCE_TIMEOUT_S}s '
                        'budget. Required for the canonical Exp 1 sweep.')
    p.add_argument('--smoke', dest='smoke', action='store_true',
                   help='Run only the first instance and assert a '
                        'hand-checked expected verdict, then exit. Used '
                        'in pre-flight before launching the full sweep.')
    args = p.parse_args()
    _VERIFICATION_METHOD = args.verification_method
    _OUT_CSV = args.output_csv

    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGALRM, _raise_timeout)
    if not _INSTANCES_CSV.exists():
        print(f'instances.csv not found at {_INSTANCES_CSV}', file=sys.stderr)
        sys.exit(2)

    # Read the VNN-COMP per-row timeout (column 3) along with the path
    # pair so we can honour it under ``--vnncomp-timeouts``.
    instances = []
    with open(_INSTANCES_CSV, newline='') as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                vnncomp_to = int(row[2])
            except ValueError:
                continue  # skip header or malformed
            instances.append((row[0].strip(), row[1].strip(), vnncomp_to))
    if args.smoke:
        instances = instances[:1]
        print(f'[smoke] Running only the first instance', flush=True)
    print(f'Loaded {len(instances)} instances', flush=True)
    print(f'Config: flow_config={_FLOW_CONFIG}  n_train={_N_TRAIN}  '
          f'flow_epochs={_FLOW_EPOCHS}  alpha={_ALPHA}  SEED={_SEED}',
          flush=True)
    print(f'Verification method: {_VERIFICATION_METHOD}', flush=True)
    print(f'Timeout policy: '
          f'{"VNN-COMP per-row" if args.vnncomp_timeouts else f"fixed {_PER_INSTANCE_TIMEOUT_S}s"}',
          flush=True)

    fields = ['onnx_file', 'vnnlib_file', 'verdict', 'coverage', 'q',
              'epsilon_total', 'delta_total', 'train_s', 'verify_s',
              'total_s', 'amls_levels_used', 'cex_x', 'cex_y', 'error']
    t_start = time.time()
    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0, 'SKIPPED': 0,
              'ERROR': 0, 'TIMEOUT': 0}

    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()

        for k, (onnx_rel, vnn_rel, vnncomp_to) in enumerate(instances, start=1):
            elapsed = time.time() - t_start
            timeout_s = vnncomp_to if args.vnncomp_timeouts else _PER_INSTANCE_TIMEOUT_S
            print(f'[{k}/{len(instances)}  t={elapsed:.0f}s  budget={timeout_s}s] '
                  f'{onnx_rel} + {vnn_rel}', flush=True)
            t0 = time.time()
            # Per-instance RNG reset: every instance starts from the same
            # state (SEED=47) regardless of execution order, so reordering
            # the sweep doesn't change individual rows.
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            try:
                # SIGALRM fires at Python bytecode boundaries — long C-extension
                # calls (CUDA, LP solvers) may delay the TimeoutError until
                # control returns to Python. The budget is a soft bound, not a hard one.
                signal.alarm(timeout_s)
                row = _run_one_instance(onnx_rel, vnn_rel, seed=_SEED)
            except TimeoutError:
                row = {'verdict': 'TIMEOUT',
                       'error': f'per-instance timeout {timeout_s}s'}
            finally:
                signal.alarm(0)

            out_row = {_f: '' for _f in fields}
            out_row['onnx_file'] = Path(onnx_rel).name
            out_row['vnnlib_file'] = Path(vnn_rel).name
            for k2, v in row.items():
                out_row[k2] = v
            writer.writerow(out_row)
            f.flush()

            counts[row['verdict']] = counts.get(row['verdict'], 0) + 1
            print(f'    verdict={row["verdict"]}  wall={time.time()-t0:.1f}s',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {_OUT_CSV}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    # Smoke assertion: ACAS Xu instance 1 is (1_1, prop_1) — αβ-CROWN
    # (VNN-COMP 2025) returns UNSAT in 8.45s, and Phase 5e of ours
    # returned UNSAT in 61.7s wall under the same base config.
    # If smoke is on and the verdict isn't UNSAT, exit non-zero so
    # CI / pre-flight can catch it.
    if args.smoke:
        first_verdict = counts.get('UNSAT', 0)
        if first_verdict != 1:
            print(
                f'[smoke] FAIL: expected UNSAT on first instance, got '
                f'counts={counts}', file=sys.stderr,
            )
            sys.exit(1)
        print('[smoke] PASS: first instance UNSAT as expected.')


if __name__ == '__main__':
    main()
