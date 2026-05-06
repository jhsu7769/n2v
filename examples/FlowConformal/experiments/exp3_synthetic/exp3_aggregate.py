"""Aggregate Exp 3 per-(benchmark, score, spec, method) CSVs into
``exp3_comparison_table.csv``.

Exp 3 has finer cell granularity than Exp 1/2 — every cell is a
``(benchmark, score, spec_type, method, seed)`` tuple. The aggregate
reduces over seeds to give one row per cell with verdict-counts and
median wall.

Schema:
    benchmark, method, score, spec_type,
    n_seeds,
    n_unsat, n_unknown, n_sat, n_not_implemented, n_error,
    median_wall_s, p10_wall_s, p90_wall_s

Volume-comparison columns (volume_ratio_vs_exact, volume_estimate)
are not yet wired through the new runners — they need a parallel
volume-MC pass after the verdict pipeline. Tracked as a follow-up.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_aggregate
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, List, Optional, Tuple

from examples.FlowConformal.experiments.exp3_synthetic._benchmarks import (
    EXP3_BENCHMARKS,
    EXP3_SCORES,
    EXP3_SPECS,
)

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'outputs'

_AGGREGATE_FIELDS = [
    'benchmark', 'method', 'score', 'spec_type',
    'n_seeds',
    'n_unsat', 'n_unknown', 'n_sat',
    'n_not_implemented', 'n_error',
    'median_wall_s', 'p10_wall_s', 'p90_wall_s',
]


def _percentile(vals: list, q: float) -> Optional[float]:
    if not vals:
        return None
    s = sorted(vals)
    if len(s) == 1:
        return s[0]
    rank = q * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def _parse_filename(name: str) -> Optional[Tuple[str, str, str, str]]:
    """Parse our convention into (benchmark, score, spec_type, method).

    Accepted shapes:
      * ours: ``exp3_<benchmark>_<score>_<spec>_ours.csv``
      * hashemi: ``exp3_<benchmark>_<spec>_hashemi_<naive|clipping>.csv``
    """
    if not (name.startswith('exp3_') and name.endswith('.csv')):
        return None
    stem = name[len('exp3_'):-len('.csv')]
    # ours runner: contains a known score
    for bench in EXP3_BENCHMARKS:
        if not stem.startswith(bench + '_'):
            continue
        rest = stem[len(bench) + 1:]
        for score in EXP3_SCORES:
            if not rest.startswith(score + '_'):
                continue
            rest2 = rest[len(score) + 1:]
            for spec in EXP3_SPECS:
                if rest2 == f'{spec}_ours':
                    return bench, score, spec, 'ours'
        for spec in EXP3_SPECS:
            if rest.startswith(f'{spec}_'):
                method = rest[len(spec) + 1:]
                if method.startswith('hashemi'):
                    # ``hashemi_clipping`` or ``hashemi_naive``
                    return bench, '', spec, method
    return None


def aggregate(out_dir: Path = _OUT_DIR) -> Path:
    rows_by_cell: Dict[Tuple[str, str, str, str], List[dict]] = defaultdict(list)
    for p in sorted(out_dir.glob('exp3_*.csv')):
        if p.name in ('exp3_comparison_table.csv',
                      'exp3_volume_comparison.csv',
                      'exp3_geo_transforms.csv'):
            continue
        parsed = _parse_filename(p.name)
        if parsed is None:
            continue
        bench, score, spec, method = parsed
        if any(s.endswith('_smoke') for s in (bench, method)):
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                rows_by_cell[(bench, score, spec, method)].append(r)

    summary_path = out_dir / 'exp3_comparison_table.csv'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_AGGREGATE_FIELDS)
        w.writeheader()
        for (bench, score, spec, method), rows in sorted(rows_by_cell.items()):
            counts = defaultdict(int)
            walls = []
            for r in rows:
                counts[r.get('verdict', '').strip()] += 1
                try:
                    walls.append(float(r['wall_s']))
                except (KeyError, ValueError):
                    pass
            w.writerow({
                'benchmark': bench, 'method': method,
                'score': score or '',
                'spec_type': spec,
                'n_seeds': len(rows),
                'n_unsat': counts.get('UNSAT', 0),
                'n_unknown': counts.get('UNKNOWN', 0),
                'n_sat': counts.get('SAT', 0),
                'n_not_implemented': counts.get('NOT_IMPLEMENTED', 0),
                'n_error': counts.get('ERROR', 0),
                'median_wall_s': (
                    f'{median(walls):.3f}' if walls else ''),
                'p10_wall_s': (
                    f'{_percentile(walls, 0.10):.3f}' if walls else ''),
                'p90_wall_s': (
                    f'{_percentile(walls, 0.90):.3f}' if walls else ''),
            })
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=_OUT_DIR)
    args = p.parse_args()
    summary = aggregate(args.out_dir)
    print(f'Wrote {summary}')
    with open(summary) as f:
        print(f.read())


if __name__ == '__main__':
    main()
