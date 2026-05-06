"""Comprehensive Tier-1 hyperparameter probe v2.

What's different from v1:

  * Falsifier / random-sample SAT path is OFF for both methods. v1 had
    Hashemi's random-sample SAT path on (it's hard-coded inside the
    runner). The result was misleading: 3 of Hashemi's "wins" came from
    random sampling, not the reach-set machinery. v2 turns it off so the
    UNSAT comparison is purely reach-set-vs-reach-set.

  * Wider hparam grid:
      Hashemi-clipping  m in {500, 1000, 2000, 4000, 8000}
      ours              tiny < small < medium < large < full
        tiny   = (n_train= 500, flow_epochs= 500, scenario_n= 250)
        small  = (n_train=1000, flow_epochs=1000, scenario_n= 500)
        medium = (n_train=2000, flow_epochs=2000, scenario_n=1000)
        large  = (n_train=3000, flow_epochs=1500, scenario_n=1500)
        full   = (n_train=5000, flow_epochs=2000, scenario_n=2000)

  * Adds ACAS Xu so every Tier-1 benchmark is probed.

  * Surfaces the flow training loss curve (full per-epoch list) in the
    CSV so we can fit knee-of-curve points and pick smaller flow_epochs
    where convergence is fast.

  * Cross-references each (instance, verdict) against αβ-CROWN +
    NeuralSAT + PyRAT + NNEnum ground truth. A verdict is flagged
    ``FALSE_UNSAT`` only when *some* sound verifier found a true ``sat``
    counterexample for that instance — i.e. only when the reach-set
    method's UNSAT contradicts a verified-by-another-tool ``sat``.

Per-cell soft timeout = per-row VNN-COMP budget (column 3 of each
``instances.csv``); image benchmarks (cifar10_resnet110, vit_small)
default to 600s. Smart skip: TIMEOUT on instance 1 of a cell skips the
remaining instances of that cell.

Output CSV columns: benchmark, instance, method, hp_name, m, n_train,
flow_epochs, scenario_n_samples, verdict, wall_s, vnncomp_timeout_s,
coverage_empirical, coverage_n_test, q, epsilon_total, delta_total,
flow_loss_curve, ground_truth, ground_truth_source, soundness_flag,
error, timestamp.

Estimated wall: ~5–6 hr. With smart-skip, typically 4–5 hr in practice.
"""
from __future__ import annotations

import argparse
import csv
import json
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

HASHEMI_M_VALUES = [500, 1000, 2000, 4000, 8000]

OURS_HPARAM_COMBOS = [
    ('tiny',   dict(n_train=500,  flow_epochs=500,  scenario_n_samples=250)),
    ('small',  dict(n_train=1000, flow_epochs=1000, scenario_n_samples=500)),
    ('medium', dict(n_train=2000, flow_epochs=2000, scenario_n_samples=1000)),
    ('large',  dict(n_train=3000, flow_epochs=1500, scenario_n_samples=1500)),
    ('full',   dict(n_train=5000, flow_epochs=2000, scenario_n_samples=2000)),
    # 'mega' uses the n_train headroom confirmed by the GPU-batchsize
    # ablation: at the production batch_size=2048 the wall scales roughly
    # linearly in n_train, so 2x data costs ~2x training wall — within
    # budget for benchmarks where 'full' fits comfortably.
    ('mega',   dict(n_train=10000, flow_epochs=2000, scenario_n_samples=2000)),
]

ALL_BENCHMARKS = [
    # Exp 1
    'acasxu_2023',
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

# Sound-verifier subdir map (under vnncomp2025_results/<tool>/<bench_dir>).
# Used to gather ground-truth verdicts for cross-referencing.
_SOUND_VERIFIERS = ['alpha_beta_crown', 'neuralsat', 'pyrat', 'nnenum', 'nnv']

_VNNCOMP_BENCH_DIR_MAP = {
    'acasxu_2023': '2025_acasxu_2023',
    'collins_rul_cnn_2022': '2025_collins_rul_cnn_2022',
    'cora_2024': '2025_cora_2024',
    'dist_shift_2023': '2025_dist_shift_2023',
    'linearizenn_2024': '2025_linearizenn_2024',
    'malbeware': '2025_malbeware',
    'ml4acopf_2024': '2025_ml4acopf_2024',
    'safenlp_2024': '2025_safenlp_2024',
    'tllverify_2023': '2025_tllverifybench_2023',
    'vit_2023': '2025_vit_2023',
    'tinyimagenet_2024': '2025_tinyimagenet_2024',
    'yolo_2023': '2025_cctsdb_yolo_2023',
}
_VNNCOMP_RESULTS_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_results',
))

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
    'flow_loss_curve',
    'ground_truth', 'ground_truth_source', 'soundness_flag',
    'error', 'timestamp',
]


# ---------------------------------------------------------------------------
# Per-row VNN-COMP timeouts
# ---------------------------------------------------------------------------

def _read_vnncomp_timeouts(benchmark: str) -> dict[str, int]:
    """Return ``{instance_name: timeout_seconds}``. ``instance_name``
    matches ``"<onnx_basename>+<vnnlib_basename>"``.
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
            out[f'{Path(row[0]).name}+{Path(row[1]).name}'] = t
    return out


# ---------------------------------------------------------------------------
# Ground-truth lookup across sound verifiers
# ---------------------------------------------------------------------------

def _load_sound_verifier_results(benchmark: str
                                  ) -> dict[str, dict[str, str]]:
    """Return ``{verifier: {instance_name: verdict}}`` for sound verifiers
    that have a results.csv for this benchmark. ``verdict`` is one of
    ``sat | unsat | timeout | unknown | error``.

    Each VNN-COMP results.csv row format:
        bench, onnx_path, vnnlib_path, prep_time, result, wall
    """
    bench_subdir = _VNNCOMP_BENCH_DIR_MAP.get(benchmark)
    if bench_subdir is None:
        return {}
    out: dict[str, dict[str, str]] = {}
    for verifier in _SOUND_VERIFIERS:
        csv_path = _VNNCOMP_RESULTS_ROOT / verifier / bench_subdir / 'results.csv'
        if not csv_path.exists():
            continue
        verifier_results: dict[str, str] = {}
        with open(csv_path) as f:
            for row in csv.reader(f):
                if len(row) < 5:
                    continue
                onnx_n = Path(row[1]).name
                vnn_n = Path(row[2]).name
                key = f'{onnx_n}+{vnn_n}'
                verifier_results[key] = row[4].lower()
        if verifier_results:
            out[verifier] = verifier_results
    return out


def _ground_truth(verifier_results: dict[str, dict[str, str]],
                   instance: str
                   ) -> tuple[str, str]:
    """Decide ground-truth ``sat | unsat | unknown`` for ``instance``.

    Rule: a verdict is ``ground_truth = sat`` iff *any* sound verifier
    reported ``sat`` for this instance. Symmetric for ``unsat`` — at
    least one verifier reported ``unsat`` and *no* verifier disagreed
    with ``sat``. If two sound verifiers disagree (one ``sat``, one
    ``unsat``) the ground truth is ``conflict``.

    Returns ``(verdict, source)`` where ``source`` lists the verifiers
    that agreed.
    """
    sat_voters = [v for v, res in verifier_results.items()
                  if res.get(instance) == 'sat']
    unsat_voters = [v for v, res in verifier_results.items()
                    if res.get(instance) == 'unsat']
    if sat_voters and unsat_voters:
        return ('conflict', f'sat=[{",".join(sat_voters)}] '
                              f'unsat=[{",".join(unsat_voters)}]')
    if sat_voters:
        return ('sat', ','.join(sat_voters))
    if unsat_voters:
        return ('unsat', ','.join(unsat_voters))
    return ('unknown', '')


def _soundness_flag(probe_verdict: str, ground_truth: str) -> str:
    """Return a flag describing the soundness of ``probe_verdict`` given
    ``ground_truth``. Empty string when no violation is observed.

    * ``FALSE_UNSAT`` — the probe certified UNSAT but at least one sound
      verifier produced a counterexample (``sat``). This is the
      soundness story we care about.
    * ``FALSE_SAT`` — the probe asserted SAT but the ground-truth
      verifiers proved ``unsat``. (Should never happen with the
      reach-set machinery alone; it's only a meaningful flag in v1
      where Hashemi had random-sample SAT enabled.)
    * ``ok`` — the probe verdict matches ground truth.
    * ``no_gt`` — no sound verifier reached a verdict for this instance.
    * ``conflict`` — sound verifiers disagree among themselves.
    """
    if ground_truth == 'unknown':
        return 'no_gt'
    if ground_truth == 'conflict':
        return 'conflict'
    if probe_verdict == 'UNSAT' and ground_truth == 'sat':
        return 'FALSE_UNSAT'
    if probe_verdict == 'SAT' and ground_truth == 'unsat':
        return 'FALSE_SAT'
    if probe_verdict in ('UNSAT', 'SAT') and probe_verdict.lower() == ground_truth:
        return 'ok'
    return ''


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raise_timeout(signum, frame):
    raise TimeoutError()


def _run_with_timeout(fn, timeout_s, on_timeout):
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
        'flow_loss_curve': '',
        'ground_truth': '', 'ground_truth_source': '',
        'soundness_flag': '',
        'error': '',
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


# ---------------------------------------------------------------------------
# Method: Hashemi-clipping (reach-set only — no random-sample SAT path)
# ---------------------------------------------------------------------------

def run_hashemi_clipping(loader, *, m: int, seed: int) -> dict:
    """Run Hashemi-clipping with **reach-set only** (no random-sample
    SAT path). Verdicts are UNSAT/UNKNOWN/ERROR — never SAT.
    """
    from examples.FlowConformal.experiments.baselines._common import (
        empirical_coverage_for_box, halfspace_disjoint_from_box,
        torch_callable,
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

        # NOTE: random-sample SAT path is INTENTIONALLY OMITTED here.
        # Probe v2 measures pure reach-set capability for both methods.

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
        'flow_loss_curve': '',
        'error': '',
    }


# ---------------------------------------------------------------------------
# Method: ours (flow-conformal + AMLS, falsifier OFF)
# ---------------------------------------------------------------------------

def run_ours(loader, *, hparams: dict, seed: int) -> dict:
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

    loss_curve = result.get('flow_train_loss_curve', [])
    return {
        'verdict': result['verdict'],
        'coverage_empirical': result.get('coverage_empirical', ''),
        'coverage_n_test': '',
        'q': result.get('q', ''),
        'epsilon_total': result.get('epsilon_total', ''),
        'delta_total': result.get('delta_total', ''),
        'flow_loss_curve': json.dumps(loss_curve) if loss_curve else '',
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
                   default_timeout_s, run_one, verifier_results):
    cell_skipped = False
    for inst_idx, item in enumerate(instances):
        instance_name = item[0]
        loader = item[1]
        loader_t = item[2] if len(item) > 2 else 0
        vnncomp_t = (vnncomp_t_map.get(instance_name, 0)
                     or loader_t or default_timeout_s)
        gt, gt_src = _ground_truth(verifier_results, instance_name)
        if cell_skipped:
            row = run_one.row_scaffold(instance_name, vnncomp_t)
            row['verdict'] = 'SKIPPED'
            row['error'] = 'skipped after instance 1 TIMEOUT'
            row['ground_truth'] = gt
            row['ground_truth_source'] = gt_src
            row['soundness_flag'] = ''
            writer.writerow(row); f_out.flush()
            print(f'[probe] {cell_label} inst {inst_idx + 1} {instance_name}'
                  f'  -> SKIPPED', flush=True)
            continue

        seed = (hash((cell_label, instance_name)) & 0x7FFFFFFF)
        print(f'[probe] {cell_label} inst {inst_idx + 1}/{len(instances)} '
              f'{instance_name} budget={vnncomp_t}s gt={gt}', flush=True)

        def call(loader=loader, seed=seed):
            return run_one.invoke(loader, seed)

        on_timeout = {'verdict': 'TIMEOUT',
                       'error': f'timeout {vnncomp_t}s'}
        res, wall = _run_with_timeout(call, vnncomp_t, on_timeout)

        row = run_one.row_scaffold(instance_name, vnncomp_t)
        row.update(res)
        row['wall_s'] = f'{wall:.1f}'
        row['ground_truth'] = gt
        row['ground_truth_source'] = gt_src
        row['soundness_flag'] = _soundness_flag(row['verdict'], gt)
        row = _maybe_round_floats(row)
        writer.writerow(row); f_out.flush()

        flag = row['soundness_flag']
        flag_s = f' [{flag}]' if flag else ''
        print(f'[probe]    verdict={row["verdict"]}{flag_s} wall={wall:.1f}s '
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
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--benchmarks', type=str,
                        default=','.join(ALL_BENCHMARKS))
    parser.add_argument('--methods', type=str,
                        default='hashemi_clipping,ours')
    parser.add_argument('--instances-per-cell', type=int, default=3)
    parser.add_argument('--default-timeout-s', type=int, default=600)
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
            print(f'[probe]   ours {nm:6s}: {d}', flush=True)
    print(f'[probe] ground-truth verifiers: {_SOUND_VERIFIERS}', flush=True)

    t_start = time.time()

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
        verifier_results = _load_sound_verifier_results(bench)
        print(f'[probe]   {len(instances)} instances; budgets: '
              f'{[vnncomp_t_map.get(item[0], 0) or (item[2] if len(item) > 2 else 0) for item in instances]}; '
              f'gt-verifiers: {list(verifier_results.keys())}', flush=True)

        if 'hashemi_clipping' in methods:
            for m in HASHEMI_M_VALUES:
                cell = _HashemiCell(bench, m)
                cell_label = f'[{bench} hashemi {cell.hp_name}]'
                _process_cell(writer=writer, f_out=f_out,
                              cell_label=cell_label,
                              instances=instances,
                              vnncomp_t_map=vnncomp_t_map,
                              default_timeout_s=args.default_timeout_s,
                              run_one=cell,
                              verifier_results=verifier_results)

        if 'ours' in methods:
            for hp_name, hp_dict in OURS_HPARAM_COMBOS:
                cell = _OursCell(bench, hp_name, hp_dict)
                cell_label = f'[{bench} ours {hp_name}]'
                _process_cell(writer=writer, f_out=f_out,
                              cell_label=cell_label,
                              instances=instances,
                              vnncomp_t_map=vnncomp_t_map,
                              default_timeout_s=args.default_timeout_s,
                              run_one=cell,
                              verifier_results=verifier_results)

    f_out.close()
    elapsed_min = (time.time() - t_start) / 60
    print(f'\n[probe] === probe v2 complete in {elapsed_min:.1f} min ===',
          flush=True)
    print(f'[probe] wrote {args.output_csv}', flush=True)


if __name__ == '__main__':
    main()
