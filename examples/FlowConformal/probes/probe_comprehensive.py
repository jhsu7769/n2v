"""Comprehensive Tier-1 hyperparameter probe.

Goal: pick the right hyperparameters for the production sweeps with
high confidence. For each (benchmark, method) cell we run multiple
hyperparameter setpoints across multiple instances and record verdicts
+ wall-clocks against the per-row VNN-COMP timeout.

Sweeps (per benchmark × instance):

  Hashemi-clipping
    m in {500, 2000, 8000}
      effect: wall ~ linear in m; tightness improves with m. Goal is
              the largest m that fits the per-benchmark VNN-COMP budget.

  Ours (flow-conformal + AMLS)
    (n_train, flow_epochs, scenario_n_samples) in
        small  = (1000, 1000,  500)
        medium = (2000, 2000, 1000)
        full   = (5000, 2000, 2000)   # Phase 5d locked
      effect: each axis trades wall for tightness; goal is the largest
              combo that fits.

Falsifier is OFF for the probe so we measure the true verification
pipeline cost without random-sampling shortcuts. The production sweeps
(Exp 1 with falsifier ON, Exp 2 with falsifier OFF per design) will be
at most as slow as this probe, so any cell that fits here will fit the
production budget.

Per-cell soft timeout = per-instance VNN-COMP budget (column 3 of
each ``instances.csv``). Smart skip: if a cell's first instance hits
TIMEOUT, the remaining instances for that (benchmark, method, hparam)
cell are recorded as SKIPPED instead of being re-run.

Output CSV: one row per (benchmark, instance, method, hparam) attempt.
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_PROJ_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))

# ---------------------------------------------------------------------------
# Sweep matrix
# ---------------------------------------------------------------------------

HASHEMI_M_VALUES = [500, 2000, 8000]

OURS_HPARAM_COMBOS = [
    ('small',  dict(n_train=1000, flow_epochs=1000, scenario_n_samples=500)),
    ('medium', dict(n_train=2000, flow_epochs=2000, scenario_n_samples=1000)),
    ('full',   dict(n_train=5000, flow_epochs=2000, scenario_n_samples=2000)),
]

ALL_BENCHMARKS = [
    # Exp 1 (ACAS Xu already covered by Phase 5d sweep — exclude here)
    'collins_rul_cnn_2022',
    'cora_2024',
    'dist_shift_2023',
    'linearizenn_2024',
    'safenlp_2024',
    'tllverify_2023',
    # Exp 2
    'vit_2023',
    'yolo_2023',
    'cifar10_resnet110',
    'vit_small_cifar10',
]

# Phase 5d locked conformal params (held constant across the probe).
_ALPHA = 0.001
_EPSILON = 0.001
_SCENARIO_BETA = 0.001
_FLOW_CONFIG = 'base'
_VERIFICATION_METHOD = 'amls'
_N_TEST_COVERAGE = 1000

CSV_FIELDS = [
    'benchmark', 'instance', 'method', 'hp_name',
    'm', 'n_train', 'flow_epochs', 'scenario_n_samples',
    'verdict', 'wall_s', 'vnncomp_timeout_s',
    'coverage_empirical', 'coverage_n_test',
    'q', 'epsilon_total', 'delta_total',
    'error', 'timestamp',
]


# ---------------------------------------------------------------------------
# Per-row VNN-COMP timeouts
# ---------------------------------------------------------------------------

def _read_vnncomp_timeouts(benchmark: str) -> dict[str, int]:
    """Return ``{instance_name: timeout_seconds}`` for every row of
    ``benchmark``'s ``instances.csv``. ``instance_name`` matches the
    naming convention used by ``load_benchmark_instances`` —
    ``"<onnx_basename>+<vnnlib_basename>"``.
    """
    from examples.FlowConformal.experiments.baselines._common import (
        VNNCOMP_BENCHMARK_ROOTS,
    )
    root_str = VNNCOMP_BENCHMARK_ROOTS.get(benchmark)
    if root_str is None:
        return {}
    csv_path = Path(os.path.expanduser(root_str)) / 'instances.csv'
    if not csv_path.exists():
        return {}
    out: dict[str, int] = {}
    with open(csv_path) as f:
        for row in csv.reader(f):
            if len(row) < 3:
                continue
            try:
                t = int(row[2])
            except ValueError:
                continue
            onnx_basename = Path(row[0]).name
            vnn_basename = Path(row[1]).name
            out[f'{onnx_basename}+{vnn_basename}'] = t
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_timeout(signum, frame):
    raise TimeoutError()


def _run_with_timeout(fn, timeout_s, on_timeout):
    """Run ``fn()`` with a SIGALRM soft timeout. Returns
    ``(result, wall_s)`` where ``result`` is ``on_timeout`` if the
    timeout fired.
    """
    signal.signal(signal.SIGALRM, _raise_timeout)
    signal.alarm(int(timeout_s))
    t0 = time.time()
    try:
        return fn(), time.time() - t0
    except TimeoutError:
        return on_timeout, time.time() - t0
    finally:
        signal.alarm(0)


def _fmt(v):
    if v is None or v == '':
        return ''
    if isinstance(v, float):
        if np.isnan(v):
            return ''
        return f'{v:.4f}'
    return str(v)


def _empty_row(benchmark, instance, method, hp_name, vnncomp_timeout_s,
               *, m=None, n_train=None, flow_epochs=None,
               scenario_n_samples=None):
    return {
        'benchmark': benchmark,
        'instance': instance,
        'method': method,
        'hp_name': hp_name,
        'm': m if m is not None else '',
        'n_train': n_train if n_train is not None else '',
        'flow_epochs': flow_epochs if flow_epochs is not None else '',
        'scenario_n_samples': (scenario_n_samples
                                if scenario_n_samples is not None else ''),
        'verdict': '', 'wall_s': '',
        'vnncomp_timeout_s': vnncomp_timeout_s,
        'coverage_empirical': '', 'coverage_n_test': '',
        'q': '', 'epsilon_total': '', 'delta_total': '',
        'error': '',
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


# ---------------------------------------------------------------------------
# Method: Hashemi-clipping
# ---------------------------------------------------------------------------

def run_hashemi_clipping(loader, *, m: int, seed: int) -> dict:
    """Run Hashemi-clipping on one (network, boxes, spec) tuple.

    The sample-based SAT path (random sampling against the unsafe
    halfspace) is part of Hashemi's standard verification flow and is
    kept on. It costs O(min(2048, m/4)) extra forward passes.
    """
    from examples.FlowConformal.experiments.baselines._common import (
        empirical_coverage_for_box, halfspace_disjoint_from_box,
        halfspace_witness_from_samples, torch_callable,
    )
    from n2v.probabilistic import verify
    from n2v.sets import Box

    out = loader()
    if out is None:
        return {'verdict': 'SKIPPED', 'error': 'loader returned None'}
    net, boxes, spec, _ = out

    ell = m - 1
    cov_vals: list[float] = []
    cov_n_total = 0
    any_unknown = False
    pbox = None

    for (lb, ub) in boxes:
        input_set = Box(np.asarray(lb).flatten(),
                        np.asarray(ub).flatten())
        model_fn = torch_callable(net)
        try:
            pbox = verify(
                model=model_fn,
                input_set=input_set,
                m=m, ell=ell, epsilon=_EPSILON,
                surrogate='clipping_block',
                seed=seed, verbose=False,
            )
        except TimeoutError:
            raise
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'verify {type(e).__name__}: {e}'}

        # Empirical coverage on held-out samples drawn from this input box.
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

        # SAT via random sampling — part of the method, not the falsifier
        # ensemble (which is an n2v-only addition).
        try:
            lb_samp = input_set.lb.flatten()
            ub_samp = input_set.ub.flatten()
            rng = np.random.default_rng(seed)
            xs = rng.uniform(lb_samp, ub_samp,
                             size=(min(2048, max(1, m // 4)),
                                   lb_samp.size)).astype(np.float32)
            ys = model_fn(xs)
            cex_idx = halfspace_witness_from_samples(spec, ys)
            if cex_idx is not None:
                cov_emp = (float(np.mean(cov_vals))
                           if cov_vals else float('nan'))
                return {
                    'verdict': 'SAT',
                    'coverage_empirical': cov_emp,
                    'coverage_n_test': cov_n_total,
                    'q': '', 'epsilon_total': _EPSILON, 'delta_total': '',
                    'error': '',
                }
        except Exception:
            pass

        disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
        if disjoint is True:
            continue
        any_unknown = True

    cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
    return {
        'verdict': 'UNKNOWN' if any_unknown else 'UNSAT',
        'coverage_empirical': cov_emp,
        'coverage_n_test': cov_n_total,
        'q': '', 'epsilon_total': _EPSILON, 'delta_total': '',
        'error': '',
    }


# ---------------------------------------------------------------------------
# Method: ours (flow-conformal + AMLS)
# ---------------------------------------------------------------------------

def run_ours(loader, *, hparams: dict, seed: int) -> dict:
    """Run ours on one (network, boxes, spec) tuple. Falsifier OFF so the
    wall-time reflects the verification pipeline, not random sampling.
    """
    from examples.FlowConformal.benchmarks._common import (
        run_verification_pipeline,
    )

    out = loader()
    if out is None:
        return {'verdict': 'SKIPPED', 'error': 'loader returned None'}
    net, boxes, spec, _ = out

    box_results = []
    for box_idx, (lb, ub) in enumerate(boxes):
        try:
            r = run_verification_pipeline(
                network=net,
                input_lb=np.asarray(lb).flatten(),
                input_ub=np.asarray(ub).flatten(),
                spec=spec,
                alpha=_ALPHA,
                scenario_beta=_SCENARIO_BETA,
                flow_config=_FLOW_CONFIG,
                verification_method=_VERIFICATION_METHOD,
                use_falsifier=False,
                seed=seed + 7919 * box_idx,
                **hparams,
            )
        except TimeoutError:
            raise
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': (f'runfailed box={box_idx} '
                              f'{type(e).__name__}: {e}')}
        box_results.append(r)
        if r['verdict'] == 'SAT':
            break

    verdicts = [r['verdict'] for r in box_results]
    if 'SAT' in verdicts:
        result = next(r for r in box_results if r['verdict'] == 'SAT')
    elif all(v == 'UNSAT' for v in verdicts):
        result = box_results[0]
        if len(box_results) > 1:
            eps_sum = sum((r.get('epsilon_total') or 0.0) for r in box_results)
            delta_min = min((r.get('delta_total') or 1.0) for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')

    return {
        'verdict': result['verdict'],
        'coverage_empirical': result.get('coverage_empirical', ''),
        'coverage_n_test': '',
        'q': result.get('q', ''),
        'epsilon_total': result.get('epsilon_total', ''),
        'delta_total': result.get('delta_total', ''),
        'error': '',
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _maybe_round_floats(row: dict) -> dict:
    for k in ('coverage_empirical', 'q', 'epsilon_total', 'delta_total'):
        if k in row and row[k] != '' and row[k] is not None:
            row[k] = _fmt(row[k])
    return row


def _process_cell(*, writer, f_out, cell_label, instances, vnncomp_t_map,
                   default_timeout_s, run_one):
    """Run ``run_one(loader, seed)`` over each instance of one cell, with
    per-cell smart skip on first-instance TIMEOUT. ``run_one`` must
    return a result dict; row scaffolding is added here.
    """
    cell_skipped = False
    for inst_idx, item in enumerate(instances):
        # ``load_benchmark_instances`` returns either ``(name, loader)``
        # or ``(name, loader, vnncomp_timeout_s)`` — accept both.
        instance_name = item[0]
        loader = item[1]
        loader_t = item[2] if len(item) > 2 else 0
        vnncomp_t = (vnncomp_t_map.get(instance_name, 0)
                     or loader_t or default_timeout_s)
        if cell_skipped:
            row = run_one.row_scaffold(instance_name, vnncomp_t)
            row['verdict'] = 'SKIPPED'
            row['error'] = 'skipped after instance 1 TIMEOUT'
            writer.writerow(row)
            f_out.flush()
            print(f'[probe] {cell_label} inst {inst_idx + 1} {instance_name}'
                  f'  -> SKIPPED', flush=True)
            continue

        seed = (hash((cell_label, instance_name)) & 0x7FFFFFFF)
        print(f'[probe] {cell_label} inst {inst_idx + 1}/{len(instances)} '
              f'{instance_name} budget={vnncomp_t}s', flush=True)

        def call(loader=loader, seed=seed):
            return run_one.invoke(loader, seed)

        on_timeout = {'verdict': 'TIMEOUT',
                       'error': f'timeout {vnncomp_t}s'}
        res, wall = _run_with_timeout(call, vnncomp_t, on_timeout)

        row = run_one.row_scaffold(instance_name, vnncomp_t)
        row.update(res)
        row['wall_s'] = f'{wall:.1f}'
        row = _maybe_round_floats(row)
        writer.writerow(row)
        f_out.flush()

        print(f'[probe]    verdict={row["verdict"]} wall={wall:.1f}s '
              f'cov={row.get("coverage_empirical", "")}', flush=True)

        if inst_idx == 0 and row['verdict'] == 'TIMEOUT':
            cell_skipped = True


class _HashemiCell:
    def __init__(self, benchmark, m):
        self.benchmark = benchmark
        self.m = m
        self.hp_name = f'm={m}'

    def row_scaffold(self, instance_name, vnncomp_t):
        return _empty_row(self.benchmark, instance_name, 'hashemi_clipping',
                          self.hp_name, vnncomp_t, m=self.m)

    def invoke(self, loader, seed):
        return run_hashemi_clipping(loader, m=self.m, seed=seed)


class _OursCell:
    def __init__(self, benchmark, hp_name, hp_dict):
        self.benchmark = benchmark
        self.hp_name = hp_name
        self.hp_dict = hp_dict

    def row_scaffold(self, instance_name, vnncomp_t):
        return _empty_row(self.benchmark, instance_name, 'ours',
                          self.hp_name, vnncomp_t,
                          n_train=self.hp_dict['n_train'],
                          flow_epochs=self.hp_dict['flow_epochs'],
                          scenario_n_samples=self.hp_dict['scenario_n_samples'])

    def invoke(self, loader, seed):
        return run_ours(loader, hparams=self.hp_dict, seed=seed)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-csv', type=Path, required=True,
                        help='Output CSV path (one row per attempt).')
    parser.add_argument('--benchmarks', type=str,
                        default=','.join(ALL_BENCHMARKS),
                        help='Comma-separated benchmark list.')
    parser.add_argument('--methods', type=str,
                        default='hashemi_clipping,ours',
                        help='Comma-separated method list.')
    parser.add_argument('--instances-per-cell', type=int, default=3,
                        help='Number of instances per (benchmark × hp) cell.')
    parser.add_argument('--default-timeout-s', type=int, default=600,
                        help='Fallback per-instance timeout when the '
                             'benchmark instances.csv is missing.')
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',') if b.strip()]
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(args.output_csv, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    f_out.flush()

    from examples.FlowConformal.experiments.baselines._common import (
        load_benchmark_instances,
    )

    print(f'[probe] benchmarks: {benchmarks}', flush=True)
    print(f'[probe] methods:    {methods}', flush=True)
    print(f'[probe] N inst/cell: {args.instances_per_cell}', flush=True)
    print(f'[probe] output CSV:  {args.output_csv}', flush=True)
    print(f'[probe] sweep matrix:', flush=True)
    if 'hashemi_clipping' in methods:
        print(f'[probe]   hashemi_clipping m: {HASHEMI_M_VALUES}', flush=True)
    if 'ours' in methods:
        for nm, d in OURS_HPARAM_COMBOS:
            print(f'[probe]   ours {nm:7s}: {d}', flush=True)

    t_start = time.time()
    counts: dict[str, int] = {}

    for bench in benchmarks:
        print(f'\n[probe] === benchmark: {bench} === '
              f'(elapsed={(time.time()-t_start)/60:.1f}min)', flush=True)
        try:
            instances = load_benchmark_instances(
                bench, args.instances_per_cell,
            )
        except Exception as e:
            print(f'[probe]   load failed: {type(e).__name__}: {e}',
                  flush=True)
            continue
        if not instances:
            print(f'[probe]   no instances; skipping', flush=True)
            continue

        vnncomp_t_map = _read_vnncomp_timeouts(bench)
        print(f'[probe]   {len(instances)} instances; budgets: '
              f'{[vnncomp_t_map.get(item[0], 0) or (item[2] if len(item) > 2 else 0) for item in instances]}',
              flush=True)

        if 'hashemi_clipping' in methods:
            for m in HASHEMI_M_VALUES:
                cell = _HashemiCell(bench, m)
                cell_label = f'[{bench} hashemi {cell.hp_name}]'
                _process_cell(writer=writer, f_out=f_out,
                              cell_label=cell_label,
                              instances=instances,
                              vnncomp_t_map=vnncomp_t_map,
                              default_timeout_s=args.default_timeout_s,
                              run_one=cell)

        if 'ours' in methods:
            for hp_name, hp_dict in OURS_HPARAM_COMBOS:
                cell = _OursCell(bench, hp_name, hp_dict)
                cell_label = f'[{bench} ours {hp_name}]'
                _process_cell(writer=writer, f_out=f_out,
                              cell_label=cell_label,
                              instances=instances,
                              vnncomp_t_map=vnncomp_t_map,
                              default_timeout_s=args.default_timeout_s,
                              run_one=cell)

    f_out.close()
    elapsed_min = (time.time() - t_start) / 60
    print(f'\n[probe] === probe complete in {elapsed_min:.1f} min ===',
          flush=True)
    print(f'[probe] wrote {args.output_csv}', flush=True)


if __name__ == '__main__':
    main()
