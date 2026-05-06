"""Aggregate the gate-only FUR study into a per-benchmark + pooled table.

Pulls FUR for five tools per benchmark:

* **ours** and **hashemi-clipping** — read from this study's
  ``gate_fur_<benchmark>_<tool>.csv`` (re-run with falsifier disabled).
* **probstar**, **saver**, **rs** — read from the original Phase 1/2
  output CSVs and filtered to SAT-ground-truth rows. These tools never
  invoked APGD in their pipelines, so their existing rows are already
  gate-only and require no re-run.

Reports:

* Per-benchmark counts: SAT-instances total, evaluated (excludes
  ERROR/SKIPPED/TIMEOUT), false UNSATs, FUR with 95% Wilson CI half-width.
* Pooled across all benchmarks for each tool.

Writes a CSV table to
``gate_fur_study/outputs/_summary_gate_only_fur.csv`` and prints a
human-readable table to stdout.
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from examples.FlowConformal.experiments.gate_fur_study._common import (
    GATE_FUR_BENCHMARKS,
)

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'outputs'
_EXP1_OUT = _HERE.parent / 'exp1_vnncomp_subset' / 'outputs'
_EXP2_OUT = _HERE.parent / 'exp2_prob_scale' / 'outputs'
_GT_CSVS = {
    'exp1': _HERE.parent / 'exp1_vnncomp_subset' / 'ground_truth.csv',
    'exp2': _HERE.parent / 'exp2_prob_scale' / 'ground_truth.csv',
}

_EXP1_BENCHMARKS = (
    'acasxu_2023', 'collins_rul_cnn_2022', 'dist_shift_2023',
    'linearizenn_2024', 'malbeware', 'metaroom_2023', 'tllverify_2023',
)
_EXP2_BENCHMARKS = ('tinyimagenet_2024', 'cifar100_2024')

# (tool_label, csv_filename_template, source_dir_for_that_bench)
# Filenames use the existing on-disk conventions:
#   gate-only:  gate_fur_<bench>_<tool>.csv      (this study)
#   exp1 baselines:  exp1_<bench>_<tool>.csv
#   exp2 baselines:  exp2_<bench>_<tool>.csv


def _exp_for(benchmark: str) -> str:
    if benchmark in _EXP1_BENCHMARKS:
        return 'exp1'
    if benchmark in _EXP2_BENCHMARKS:
        return 'exp2'
    raise KeyError(benchmark)


def _baseline_dir(benchmark: str) -> Path:
    return _EXP1_OUT if _exp_for(benchmark) == 'exp1' else _EXP2_OUT


def _baseline_prefix(benchmark: str) -> str:
    return 'exp1' if _exp_for(benchmark) == 'exp1' else 'exp2'


def _load_sat_keys(benchmark: str) -> set:
    """Return the set of (onnx_basename, vnnlib_basename) keys for the
    SAT-ground-truth instances of ``benchmark``.
    """
    gt_csv = _GT_CSVS[_exp_for(benchmark)]
    sat_keys: set = set()
    if not gt_csv.exists():
        return sat_keys
    with open(gt_csv, newline='') as f:
        for r in csv.DictReader(f):
            if r['benchmark'].strip() != benchmark:
                continue
            if r['ground_truth'].strip().upper() == 'SAT':
                sat_keys.add(
                    (r['onnx_file'].strip(), r['vnnlib_file'].strip()))
    return sat_keys


def _read_verdicts(csv_path: Path) -> List[Tuple[str, str, str]]:
    """Return a list of (onnx_basename, vnnlib_basename, verdict_upper)
    for one tool/benchmark CSV. Returns ``[]`` if the file is missing.
    """
    if not csv_path.exists():
        return []
    rows: List[Tuple[str, str, str]] = []
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            v = r.get('verdict', '').strip().upper()
            o = Path(r.get('onnx_file', '').strip()).name
            n = Path(r.get('vnnlib_file', '').strip()).name
            rows.append((o, n, v))
    return rows


def _wilson_ci_half(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval half-width for k successes out of n."""
    if n == 0:
        return float('nan')
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    # Centre may differ from p; the half-width around the centre is what
    # we report. (For very small n, this is the right thing to show.)
    _ = centre
    return half


def _summarise(name: str, rows: List[Tuple[str, str, str]],
               sat_keys: set) -> Dict[str, object]:
    """Filter rows to those whose key is in sat_keys and tabulate verdicts."""
    n_sat_total = len(sat_keys)
    seen: Dict[Tuple[str, str], str] = {}
    for o, n, v in rows:
        if (o, n) in sat_keys:
            seen[(o, n)] = v
    n_eval = sum(1 for v in seen.values() if v in {'UNSAT', 'UNKNOWN', 'SAT'})
    n_false_unsat = sum(1 for v in seen.values() if v == 'UNSAT')
    n_unknown = sum(1 for v in seen.values() if v == 'UNKNOWN')
    n_sat_caught = sum(1 for v in seen.values() if v == 'SAT')
    n_other = sum(1 for v in seen.values()
                  if v not in {'UNSAT', 'UNKNOWN', 'SAT'})
    fur = n_false_unsat / n_eval if n_eval > 0 else float('nan')
    half = _wilson_ci_half(n_false_unsat, n_eval)
    return {
        'tool': name,
        'n_sat_total': n_sat_total,
        'n_evaluated': n_eval,
        'n_false_unsat': n_false_unsat,
        'n_unknown': n_unknown,
        'n_sat_caught': n_sat_caught,
        'n_other': n_other,
        'fur': fur,
        'fur_ci95_half': half,
    }


def _pool_rows(per_bench: Dict[str, List[Dict[str, object]]],
               tool: str) -> Dict[str, object]:
    """Pool counts across benchmarks for one tool."""
    n_total = 0
    n_eval = 0
    n_false = 0
    for rows in per_bench.values():
        for r in rows:
            if r['tool'] != tool:
                continue
            n_total += int(r['n_sat_total'])
            n_eval += int(r['n_evaluated'])
            n_false += int(r['n_false_unsat'])
    fur = n_false / n_eval if n_eval > 0 else float('nan')
    half = _wilson_ci_half(n_false, n_eval)
    return {
        'benchmark': '_POOLED_',
        'tool': tool,
        'n_sat_total': n_total,
        'n_evaluated': n_eval,
        'n_false_unsat': n_false,
        'n_unknown': '',
        'n_sat_caught': '',
        'n_other': '',
        'fur': fur,
        'fur_ci95_half': half,
    }


def _fmt_row(b: str, r: Dict[str, object]) -> str:
    fur = r['fur']
    half = r['fur_ci95_half']
    fur_s = f'{fur:.4f}' if isinstance(fur, float) and not math.isnan(fur) else 'n/a'
    half_s = f'±{half:.4f}' if isinstance(half, float) and not math.isnan(half) else ''
    return (f'  {b:<28s} {r["tool"]:<18s}'
            f' n_sat={r["n_sat_total"]:>4} eval={r["n_evaluated"]:>4}'
            f' false_unsat={r["n_false_unsat"]:>4}'
            f' FUR={fur_s} {half_s}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write-csv', type=Path,
                   default=_OUT_DIR / '_summary_gate_only_fur.csv')
    args = p.parse_args()

    tools_in_study = ('ours', 'hashemi_clipping')
    baseline_tools = ('probstar', 'saver', 'rs')
    all_tools = tools_in_study + baseline_tools

    per_bench: Dict[str, List[Dict[str, object]]] = {}

    for benchmark in GATE_FUR_BENCHMARKS:
        sat_keys = _load_sat_keys(benchmark)
        bench_rows: List[Dict[str, object]] = []

        # Gate-only re-runs for ours + Hashemi
        for tool in tools_in_study:
            csv_path = _OUT_DIR / f'gate_fur_{benchmark}_{tool}.csv'
            verdicts = _read_verdicts(csv_path)
            summary = _summarise(tool, verdicts, sat_keys)
            summary['benchmark'] = benchmark
            summary['source'] = str(csv_path.relative_to(_HERE.parent))
            bench_rows.append(summary)

        # Already-gate-only baselines from Phase 1/2
        baseline_dir = _baseline_dir(benchmark)
        baseline_prefix = _baseline_prefix(benchmark)
        for tool in baseline_tools:
            csv_path = baseline_dir / f'{baseline_prefix}_{benchmark}_{tool}.csv'
            verdicts = _read_verdicts(csv_path)
            summary = _summarise(tool, verdicts, sat_keys)
            summary['benchmark'] = benchmark
            summary['source'] = str(csv_path.relative_to(_HERE.parent))
            bench_rows.append(summary)

        per_bench[benchmark] = bench_rows

    # Pooled rows per tool
    pooled_rows = [_pool_rows(per_bench, t) for t in all_tools]

    # Print
    print('=== Gate-only FUR study ===\n')
    for benchmark in GATE_FUR_BENCHMARKS:
        print(f'[{benchmark}]')
        for r in per_bench[benchmark]:
            print(_fmt_row(benchmark, r))
        print()

    print('[POOLED across all benchmarks]')
    for r in pooled_rows:
        print(_fmt_row('(pooled)', r))

    # CSV
    args.write_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ['benchmark', 'tool', 'n_sat_total', 'n_evaluated',
                  'n_false_unsat', 'n_unknown', 'n_sat_caught', 'n_other',
                  'fur', 'fur_ci95_half', 'source']
    with open(args.write_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for benchmark in GATE_FUR_BENCHMARKS:
            for r in per_bench[benchmark]:
                row = {k: r.get(k, '') for k in fieldnames}
                w.writerow(row)
        for r in pooled_rows:
            row = {k: r.get(k, '') for k in fieldnames}
            w.writerow(row)
    print(f'\nWrote {args.write_csv}')


if __name__ == '__main__':
    main()
