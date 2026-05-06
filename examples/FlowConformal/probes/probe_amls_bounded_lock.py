"""Lock-in probe for ours+amls_bounded across all Exp 1/2 benchmarks.

Goal: pick the largest hparam config per benchmark that
  (a) completes 5/5 instances within the per-row VNN-COMP budget, AND
  (b) produces 0 FALSE_UNSATs against the consensus of *any* sound
      verifier (αβ-CROWN, NeuralSAT, PyRAT, NNEnum, NNV).

Image benchmarks (cifar10_resnet110, vit_small_cifar10) skip (b)
since no VNN-COMP sound-verifier ground-truth exists for them.

Adaptive search per benchmark: try configs from largest to smallest
until a config passes both criteria. Records every attempted cell so
the analysis can show the full back-off tree.

Configs (largest → smallest):
    mega:  n_train=10000, flow_epochs=2000, scenario_n_samples=2000
    full:  n_train= 5000, flow_epochs=2000, scenario_n_samples=2000
    large: n_train= 3000, flow_epochs=1500, scenario_n_samples=1500
    medium:n_train= 2000, flow_epochs=2000, scenario_n_samples=1000
    small: n_train= 1000, flow_epochs=1000, scenario_n_samples= 500
    tiny:  n_train=  500, flow_epochs= 500, scenario_n_samples= 250
    micro: n_train=  200, flow_epochs= 300, scenario_n_samples= 100
    nano:  n_train=  100, flow_epochs= 200, scenario_n_samples=  50

Verification method: amls_bounded (single-stage, post-design-doc).
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

# Reuse helpers from probe_v2.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_v2 import (  # noqa: E402
    _ground_truth,
    _load_sound_verifier_results,
    _read_vnncomp_timeouts,
    _run_with_timeout,
    _soundness_flag,
)


# ---------------------------------------------------------------------------
# Sweep matrix: configs in LARGEST → SMALLEST order
# ---------------------------------------------------------------------------

CONFIGS_LARGEST_TO_SMALLEST = [
    ('mega',   dict(n_train=10000, flow_epochs=2000, scenario_n_samples=2000)),
    ('full',   dict(n_train= 5000, flow_epochs=2000, scenario_n_samples=2000)),
    ('large',  dict(n_train= 3000, flow_epochs=1500, scenario_n_samples=1500)),
    ('medium', dict(n_train= 2000, flow_epochs=2000, scenario_n_samples=1000)),
    ('small',  dict(n_train= 1000, flow_epochs=1000, scenario_n_samples= 500)),
    ('tiny',   dict(n_train=  500, flow_epochs= 500, scenario_n_samples= 250)),
    ('micro',  dict(n_train=  200, flow_epochs= 300, scenario_n_samples= 100)),
    ('nano',   dict(n_train=  100, flow_epochs= 200, scenario_n_samples=  50)),
]

# Per-benchmark starting config to skip obviously-doomed cells. Based on
# probe v2 walls we already know what NOT to bother with.
PER_BENCHMARK_START_CONFIG = {
    # Tight VNN-COMP budgets — start small.
    'cora_2024':         'micro',   # 30s budget; even tiny was over
    'safenlp_2024':      'tiny',    # 20s budget; only tiny was close
    'vit_2023':          'micro',   # 100s budget; tiny TIMEOUTed in probe v2
    'collins_rul_cnn_2022': 'small',
    # Generous budgets — start at mega.
    'acasxu_2023':           'mega',
    'dist_shift_2023':       'mega',
    'linearizenn_2024':      'mega',
    'tllverify_2023':        'mega',
    'cifar10_resnet110':     'mega',
    'vit_small_cifar10':     'small',  # 600s budget; mega TIMEOUTed
    'tinyimagenet_2024':     'mega',
    'malbeware':             'mega',   # 100s budget; new addition for Exp 1 (uses amls_bounded_union)
    'metaroom_2023':         'mega',   # 210s budget; new addition for Exp 1 (uses amls_bounded_union)
}

# Image benchmarks that have no VNN-COMP sound-verifier ground truth.
# For these we skip the FALSE_UNSAT criterion and only use TIMEOUT-fit.
NO_GT_BENCHMARKS = {'cifar10_resnet110', 'vit_small_cifar10'}

ALL_BENCHMARKS = list(PER_BENCHMARK_START_CONFIG.keys())

# Phase-5d-locked conformal params (same as probe v2).
_ALPHA = 0.001
_SCENARIO_BETA = 0.001
_FLOW_CONFIG = 'base'
_VERIFICATION_METHOD_DEFAULT = 'amls_bounded'

# Per-benchmark verification-method overrides. Multi-class disjunctive
# specs (e.g. cora-style ``list[dict]`` with ``Hg = list[K HalfSpaces]``)
# need ``amls_bounded_union`` — running ``amls_bounded`` on those iterates
# K independent AMLS chains and TIMEOUTs at any reasonable budget.
_PER_BENCHMARK_VERIFICATION_METHOD = {
    'vit_2023':          'amls_bounded_union',  # 9-disjunct cls spec
    'malbeware':         'amls_bounded_union',  # 24-disjunct cls spec
    'metaroom_2023':     'amls_bounded_union',  # 19-disjunct cls spec
    'cifar100_2024':     'amls_bounded_union',  # 99-disjunct cls spec
    'tinyimagenet_2024': 'amls_bounded_union',  # 199-disjunct cls spec
}

# Per-benchmark ``amls_max_levels`` overrides. Default is the pipeline
# default (30). All disjunctive cls benchmarks run via amls_bounded_union
# (single chain) so the historical max_levels=5 override for vit_2023
# is no longer needed.
_PER_BENCHMARK_MAX_LEVELS = {}

_AMLS_MAX_LEVELS_DEFAULT = 30


def _verification_method_for(benchmark: str) -> str:
    return _PER_BENCHMARK_VERIFICATION_METHOD.get(
        benchmark, _VERIFICATION_METHOD_DEFAULT)


def _max_levels_for(benchmark: str) -> int:
    return _PER_BENCHMARK_MAX_LEVELS.get(
        benchmark, _AMLS_MAX_LEVELS_DEFAULT)

CSV_FIELDS = [
    'benchmark', 'instance', 'config', 'n_train', 'flow_epochs',
    'scenario_n_samples', 'verdict', 'wall_s', 'vnncomp_timeout_s',
    'coverage_empirical', 'q', 'epsilon_total', 'delta_total',
    'amls_bounded_eps_2_upper', 'amls_bounded_detected_unsafe',
    'amls_levels_used',
    'ground_truth', 'ground_truth_source', 'soundness_flag',
    'cell_status',  # 'pass' / 'fail_timeout' / 'fail_error' / 'fail_false_unsat'
    'error', 'timestamp',
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _config_index(name: str) -> int:
    for i, (n, _) in enumerate(CONFIGS_LARGEST_TO_SMALLEST):
        if n == name:
            return i
    raise KeyError(name)


def _fmt(v):
    if v is None or v == '':
        return ''
    if isinstance(v, float):
        if np.isnan(v):
            return ''
        return f'{v:.4f}'
    return str(v)


def _empty_row(benchmark, instance, config_name, hp, vnncomp_timeout_s):
    return {
        'benchmark': benchmark,
        'instance': instance,
        'config': config_name,
        'n_train': hp['n_train'],
        'flow_epochs': hp['flow_epochs'],
        'scenario_n_samples': hp['scenario_n_samples'],
        'verdict': '', 'wall_s': '',
        'vnncomp_timeout_s': vnncomp_timeout_s,
        'coverage_empirical': '', 'q': '',
        'epsilon_total': '', 'delta_total': '',
        'amls_bounded_eps_2_upper': '',
        'amls_bounded_detected_unsafe': '',
        'amls_levels_used': '',
        'ground_truth': '', 'ground_truth_source': '',
        'soundness_flag': '',
        'cell_status': '',
        'error': '',
        'timestamp': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }


# ---------------------------------------------------------------------------
# Run ours with amls_bounded
# ---------------------------------------------------------------------------

def run_ours_amls_bounded(loader, *, hparams: dict, seed: int,
                          verification_method: str = _VERIFICATION_METHOD_DEFAULT,
                          amls_max_levels: int = _AMLS_MAX_LEVELS_DEFAULT) -> dict:
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
                verification_method=verification_method,
                amls_max_levels=amls_max_levels,
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
            eps_sum = sum((r.get('epsilon_total') or 0.0)
                          for r in box_results)
            delta_min = min((r.get('delta_total') or 1.0)
                            for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')

    return {
        'verdict': result['verdict'],
        'coverage_empirical': result.get('coverage_empirical', ''),
        'q': result.get('q', ''),
        'epsilon_total': result.get('epsilon_total', ''),
        'delta_total': result.get('delta_total', ''),
        'amls_bounded_eps_2_upper': result.get(
            'amls_bounded_eps_2_upper', ''),
        'amls_bounded_detected_unsafe': result.get(
            'amls_bounded_detected_unsafe', ''),
        'amls_levels_used': result.get('amls_levels_used', ''),
        'error': '',
    }


# ---------------------------------------------------------------------------
# Per-benchmark adaptive cell loop
# ---------------------------------------------------------------------------

def evaluate_config(*, writer, f_out, benchmark, config_name, hp,
                    instances, vnncomp_t_map, default_timeout_s,
                    verifier_results, no_gt_benchmark, instances_per_cell):
    """Run ``instances_per_cell`` instances of ``benchmark`` with
    ``hp``. Returns dict { 'pass': bool, 'reasons': list[str] }.
    """
    failure_reasons: list[str] = []
    n_pass = 0
    for inst_idx in range(instances_per_cell):
        if inst_idx >= len(instances):
            break
        item = instances[inst_idx]
        instance_name = item[0]
        loader = item[1]
        loader_t = item[2] if len(item) > 2 else 0
        vnncomp_t = (vnncomp_t_map.get(instance_name, 0)
                     or loader_t or default_timeout_s)
        gt, gt_src = _ground_truth(verifier_results, instance_name)

        seed = (hash((benchmark, config_name, instance_name)) & 0x7FFFFFFF)
        print(f'    [{config_name}] inst {inst_idx + 1}/{instances_per_cell} '
              f'{instance_name} budget={vnncomp_t}s gt={gt}', flush=True)

        def call(loader=loader, seed=seed,
                 vm=_verification_method_for(benchmark),
                 ml=_max_levels_for(benchmark)):
            return run_ours_amls_bounded(
                loader, hparams=hp, seed=seed,
                verification_method=vm, amls_max_levels=ml)

        on_timeout = {'verdict': 'TIMEOUT',
                       'error': f'timeout {vnncomp_t}s'}
        res, wall = _run_with_timeout(call, vnncomp_t, on_timeout)

        row = _empty_row(benchmark, instance_name, config_name, hp,
                          vnncomp_t)
        row.update(res)
        row['wall_s'] = f'{wall:.1f}'
        row['ground_truth'] = gt
        row['ground_truth_source'] = gt_src
        row['soundness_flag'] = _soundness_flag(row['verdict'], gt)
        for k in ('coverage_empirical', 'q', 'epsilon_total',
                  'delta_total', 'amls_bounded_eps_2_upper'):
            if k in row and row[k] != '' and row[k] is not None:
                row[k] = _fmt(row[k])

        verdict = row['verdict']
        flag = row['soundness_flag']
        wall_str = f'{wall:.1f}s'

        if verdict == 'TIMEOUT':
            row['cell_status'] = 'fail_timeout'
            failure_reasons.append(
                f'TIMEOUT inst {inst_idx + 1} ({wall_str})')
            writer.writerow(row); f_out.flush()
            print(f'        verdict=TIMEOUT  wall={wall_str}', flush=True)
            return {'pass': False, 'reasons': failure_reasons}

        if verdict == 'ERROR':
            row['cell_status'] = 'fail_error'
            failure_reasons.append(
                f"ERROR inst {inst_idx + 1}: {row.get('error', '')}")
            writer.writerow(row); f_out.flush()
            print(f"        verdict=ERROR  err={row.get('error', '')}",
                  flush=True)
            return {'pass': False, 'reasons': failure_reasons}

        if (not no_gt_benchmark) and flag == 'FALSE_UNSAT':
            row['cell_status'] = 'fail_false_unsat'
            failure_reasons.append(
                f'FALSE_UNSAT on inst {inst_idx + 1}')
            writer.writerow(row); f_out.flush()
            print(f'        verdict=UNSAT  !!FALSE_UNSAT  wall={wall_str}',
                  flush=True)
            return {'pass': False, 'reasons': failure_reasons}

        row['cell_status'] = 'ok'
        writer.writerow(row); f_out.flush()
        n_pass += 1
        print(f'        verdict={verdict}  flag={flag}  wall={wall_str}',
              flush=True)

    if n_pass == instances_per_cell:
        return {'pass': True, 'reasons': []}
    return {'pass': False,
             'reasons': failure_reasons or ['short instance list']}


def lock_benchmark_config(*, writer, f_out, benchmark, instances,
                           vnncomp_t_map, default_timeout_s,
                           verifier_results, instances_per_cell,
                           start_config: str):
    """Adaptive search: largest → smallest config until one passes, or
    nano fails. Returns the winning config name or None.
    """
    no_gt_benchmark = benchmark in NO_GT_BENCHMARKS

    start_idx = _config_index(start_config)
    for idx in range(start_idx, len(CONFIGS_LARGEST_TO_SMALLEST)):
        cfg_name, hp = CONFIGS_LARGEST_TO_SMALLEST[idx]
        print(f'\n  trying config {cfg_name}: '
              f'n_train={hp["n_train"]}, '
              f'flow_epochs={hp["flow_epochs"]}, '
              f'scenario_n_samples={hp["scenario_n_samples"]}',
              flush=True)
        result = evaluate_config(
            writer=writer, f_out=f_out, benchmark=benchmark,
            config_name=cfg_name, hp=hp, instances=instances,
            vnncomp_t_map=vnncomp_t_map,
            default_timeout_s=default_timeout_s,
            verifier_results=verifier_results,
            no_gt_benchmark=no_gt_benchmark,
            instances_per_cell=instances_per_cell,
        )
        if result['pass']:
            print(f'  ✓ {benchmark} locked at config {cfg_name}', flush=True)
            return cfg_name
        print(f'  ✗ {cfg_name} failed: {"; ".join(result["reasons"])}',
              flush=True)

    print(f'  ✗✗ {benchmark}: no config fits down to nano', flush=True)
    return None


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--benchmarks', type=str,
                        default=','.join(ALL_BENCHMARKS))
    parser.add_argument('--instances-per-cell', type=int, default=5)
    parser.add_argument('--default-timeout-s', type=int, default=600)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',') if b.strip()]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(args.output_csv, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    f_out.flush()

    from examples.FlowConformal.experiments.baselines._common import (
        load_benchmark_instances,
    )

    print(f'[probe] benchmarks: {benchmarks}', flush=True)
    print(f'[probe] N inst/cell: {args.instances_per_cell}', flush=True)
    print(f'[probe] verification (default): {_VERIFICATION_METHOD_DEFAULT}',
          flush=True)
    print(f'[probe] verification overrides: {_PER_BENCHMARK_VERIFICATION_METHOD}',
          flush=True)
    print(f'[probe] output CSV:  {args.output_csv}', flush=True)

    locked_configs: dict[str, str | None] = {}
    t_start = time.time()

    for bench in benchmarks:
        print(f'\n[probe] === benchmark: {bench} === '
              f'(elapsed={(time.time()-t_start)/60:.1f}min)', flush=True)
        try:
            instances = load_benchmark_instances(
                bench, args.instances_per_cell,
            )
        except Exception as e:
            print(f'  load failed: {type(e).__name__}: {e}', flush=True)
            locked_configs[bench] = None
            continue
        if not instances:
            print(f'  no instances; skipping', flush=True)
            locked_configs[bench] = None
            continue

        vnncomp_t_map = _read_vnncomp_timeouts(bench)
        verifier_results = _load_sound_verifier_results(bench)
        budgets = [vnncomp_t_map.get(item[0], 0)
                   or (item[2] if len(item) > 2 else 0)
                   for item in instances]
        print(f'  {len(instances)} instances; budgets: {budgets}', flush=True)
        print(f'  ground-truth verifiers: {list(verifier_results.keys())}',
              flush=True)

        start_cfg = PER_BENCHMARK_START_CONFIG.get(bench, 'mega')

        winner = lock_benchmark_config(
            writer=writer, f_out=f_out, benchmark=bench,
            instances=instances, vnncomp_t_map=vnncomp_t_map,
            default_timeout_s=args.default_timeout_s,
            verifier_results=verifier_results,
            instances_per_cell=args.instances_per_cell,
            start_config=start_cfg,
        )
        locked_configs[bench] = winner

    f_out.close()

    elapsed_min = (time.time() - t_start) / 60
    print(f'\n[probe] === lock-in complete in {elapsed_min:.1f} min ===',
          flush=True)
    print(f'[probe] wrote {args.output_csv}', flush=True)
    print()
    print('=== Per-benchmark locked configs ===')
    for bench, cfg in locked_configs.items():
        cfg_str = cfg if cfg is not None else 'NO_FIT'
        print(f'  {bench:25s} → {cfg_str}')


if __name__ == '__main__':
    main()
