"""AMLS verification-budget probe (ours-only).

Hypothesis: the UNSAT-vs-UNKNOWN gap between ours-mega and Hashemi-clipping
m=8000 on probe v2 (tllverify, safenlp, linearizenn) is dominated by AMLS
rare-event-estimator variance, not by the flow's reach-set tightness or
the calibration set size m. With the same calibration data (m=8000) and
Phase 5d locked config (scenario_n_samples=2000), AMLS may produce wider
upper-confidence-bounds on Pr[unsafe] than needed to certify ε=0.001 —
even when the flow set is genuinely disjoint from the unsafe halfspace.

Sweep: ours at n_train=10000, flow_epochs=2000 (mega) on the benchmarks
where ours underperforms or matches, with ``scenario_n_samples`` taking
{2000, 4000, 8000, 16000, 32000}. All other knobs frozen.

Falsifier OFF; soundness flag against αβ-CROWN + sound-verifier
consensus (same as probe v2). Per-instance soft timeout from
``instances.csv``.

If higher AMLS budget converts UNKNOWNs to UNSATs without introducing
FALSE_UNSATs, the paper story is: "our method's reach-set is genuinely
tighter than Hashemi's; AMLS variance was the dominant verification
overhead, and matches Hashemi's UNSAT count (with 0% FUR vs Hashemi's
25-28%) once given comparable verification budget".
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

# Re-use helpers from probe_v2.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_v2 import (  # noqa: E402
    CSV_FIELDS,
    _empty_row,
    _ground_truth,
    _load_sound_verifier_results,
    _read_vnncomp_timeouts,
    _run_with_timeout,
    _soundness_flag,
    _maybe_round_floats,
)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

# Frozen flow-side hparams (ours-mega).
_BASE_HPARAMS = dict(
    n_train=10000,
    flow_epochs=2000,
)

# AMLS verification budgets to sweep.
SCENARIO_N_VALUES = [2000, 4000, 8000, 16000, 32000]

# Benchmarks where (a) ours underperforms vs Hashemi-clipping m=8000, or
# (b) we want to confirm no regression at higher AMLS budget.
DEFAULT_BENCHMARKS = [
    'safenlp_2024',          # 20s budget; only N=2000 will fit
    'tllverify_2023',        # 600s budget; ours mega 1 UNSAT vs Hashemi 3 UNSAT
    'linearizenn_2024',      # 900s budget; ours mega 2 vs Hashemi 2 (matched)
    'dist_shift_2023',       # 300s budget; control — both methods win
    'acasxu_2023',           # 116s budget; ours already wins
]

# Other Phase 5d-locked params.
_ALPHA = 0.001
_SCENARIO_BETA = 0.001
_FLOW_CONFIG = 'base'
_VERIFICATION_METHOD = 'amls'


# ---------------------------------------------------------------------------
# Method runner
# ---------------------------------------------------------------------------

def run_ours_with_amls_budget(loader, *, scenario_n: int, seed: int) -> dict:
    """Run ours at the ours-mega base config with the given AMLS sample
    budget. Returns the same dict shape as probe_v2.run_ours.
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
                scenario_n_samples=scenario_n,
                flow_config=_FLOW_CONFIG,
                verification_method=_VERIFICATION_METHOD,
                use_falsifier=False,
                seed=seed + 7919 * box_idx,
                **_BASE_HPARAMS,
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
# Cell
# ---------------------------------------------------------------------------

class _AmlsBudgetCell:
    def __init__(self, benchmark, scenario_n):
        self.benchmark = benchmark
        self.scenario_n = scenario_n
        self.hp_name = f'amls_N={scenario_n}'

    def row_scaffold(self, instance_name, vnncomp_t):
        return _empty_row(
            self.benchmark, instance_name, 'ours',
            self.hp_name, vnncomp_t,
            n_train=_BASE_HPARAMS['n_train'],
            flow_epochs=_BASE_HPARAMS['flow_epochs'],
            scenario_n_samples=self.scenario_n,
        )

    def invoke(self, loader, seed):
        return run_ours_with_amls_budget(
            loader, scenario_n=self.scenario_n, seed=seed,
        )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

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


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--output-csv', type=Path, required=True)
    parser.add_argument('--benchmarks', type=str,
                        default=','.join(DEFAULT_BENCHMARKS))
    parser.add_argument('--scenario-n-values', type=str,
                        default=','.join(str(v) for v in SCENARIO_N_VALUES),
                        help='Comma-sep list of scenario_n_samples values.')
    parser.add_argument('--instances-per-cell', type=int, default=3)
    parser.add_argument('--default-timeout-s', type=int, default=600)
    args = parser.parse_args()

    benchmarks = [b.strip() for b in args.benchmarks.split(',') if b.strip()]
    scenario_ns = [int(v.strip()) for v in args.scenario_n_values.split(',')
                   if v.strip()]

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    f_out = open(args.output_csv, 'w', newline='')
    writer = csv.DictWriter(f_out, fieldnames=CSV_FIELDS)
    writer.writeheader()
    f_out.flush()

    from examples.FlowConformal.experiments.baselines._common import (
        load_benchmark_instances,
    )

    print(f'[probe] benchmarks: {benchmarks}', flush=True)
    print(f'[probe] base ours: n_train={_BASE_HPARAMS["n_train"]} '
          f'flow_epochs={_BASE_HPARAMS["flow_epochs"]}', flush=True)
    print(f'[probe] scenario_n_samples sweep: {scenario_ns}', flush=True)
    print(f'[probe] N inst/cell: {args.instances_per_cell}', flush=True)
    print(f'[probe] output CSV:  {args.output_csv}', flush=True)

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

        for sn in scenario_ns:
            cell = _AmlsBudgetCell(bench, sn)
            cell_label = f'[{bench} ours {cell.hp_name}]'
            _process_cell(
                writer=writer, f_out=f_out,
                cell_label=cell_label,
                instances=instances,
                vnncomp_t_map=vnncomp_t_map,
                default_timeout_s=args.default_timeout_s,
                run_one=cell,
                verifier_results=verifier_results,
            )

    f_out.close()
    elapsed_min = (time.time() - t_start) / 60
    print(f'\n[probe] === amls-budget probe complete in '
          f'{elapsed_min:.1f} min ===', flush=True)
    print(f'[probe] wrote {args.output_csv}', flush=True)


if __name__ == '__main__':
    main()
