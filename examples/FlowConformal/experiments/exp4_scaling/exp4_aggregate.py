"""Aggregate Exp 4 per-(depth, method) CSVs into ``exp4_scaling_summary.csv``.

Reads every ``outputs/exp4_d<D>_<method>.csv`` in the Exp 4 directory
and produces a single summary row per ``(depth, method)`` with the
columns the heatmap and scaling-curve figures consume:

    depth, width, n_params, method,
    n_instances, n_unsat, n_unknown, n_sat, n_timeout, n_error,
    median_wall_s, p10_wall_s, p90_wall_s, n_false_unsat

``n_false_unsat`` is the count of rows where ``verdict == 'UNSAT'`` but
``ground_truth == 'unsat'`` is FALSE. Since Exp 4 is UNSAT-by-
construction, every UNSAT verdict is correct and ``n_false_unsat == 0``
unless a verifier returned UNSAT on a row whose generator marked it
otherwise (shouldn't happen in this experiment).

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.exp4_scaling.exp4_aggregate
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Optional

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'outputs'

_AGGREGATE_FIELDS = [
    'depth', 'width', 'n_params', 'method',
    'n_instances',
    'n_unsat', 'n_unknown', 'n_sat', 'n_timeout', 'n_error',
    'median_wall_s', 'p10_wall_s', 'p90_wall_s',
    'n_false_unsat', 'n_false_sat',
]


_FILENAME_RE = re.compile(r'^exp4_d(\d+)_(?P<method>[a-z_]+)\.csv$')


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


def aggregate(out_dir: Path = _OUT_DIR) -> Path:
    """Aggregate all per-(depth, method) CSVs in ``out_dir`` and write
    ``exp4_scaling_summary.csv``. Returns the path of the summary CSV.
    """
    rows_by_cell = defaultdict(list)
    width = None
    n_params_by_depth = {}
    for p in sorted(out_dir.glob('exp4_d*_*.csv')):
        m = _FILENAME_RE.match(p.name)
        if not m:
            continue
        depth = int(m.group(1))
        method = m.group('method')
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                rows_by_cell[(depth, method)].append(r)
                if width is None:
                    try:
                        width = int(r['width'])
                    except (KeyError, ValueError):
                        pass
                try:
                    n_params_by_depth[depth] = int(r['n_params'])
                except (KeyError, ValueError):
                    pass

    summary_path = out_dir / 'exp4_scaling_summary.csv'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_AGGREGATE_FIELDS)
        w.writeheader()
        for (depth, method), rows in sorted(rows_by_cell.items()):
            counts = defaultdict(int)
            walls = []
            n_false_unsat = 0
            n_false_sat = 0
            for r in rows:
                v = r.get('verdict', '').strip()
                counts[v] += 1
                try:
                    walls.append(float(r['wall_s']))
                except (KeyError, ValueError):
                    pass
                gt = r.get('ground_truth', '').strip().lower()
                if v == 'UNSAT' and gt and gt != 'unsat':
                    n_false_unsat += 1
                if v == 'SAT' and gt and gt != 'sat':
                    n_false_sat += 1
            n_inst = len(rows)
            w.writerow({
                'depth': depth,
                'width': width if width is not None else '',
                'n_params': n_params_by_depth.get(depth, ''),
                'method': method,
                'n_instances': n_inst,
                'n_unsat': counts.get('UNSAT', 0),
                'n_unknown': counts.get('UNKNOWN', 0),
                'n_sat': counts.get('SAT', 0),
                'n_timeout': counts.get('TIMEOUT', 0),
                'n_error': counts.get('ERROR', 0),
                'median_wall_s': (
                    f'{median(walls):.3f}' if walls else ''),
                'p10_wall_s': (
                    f'{_percentile(walls, 0.10):.3f}' if walls else ''),
                'p90_wall_s': (
                    f'{_percentile(walls, 0.90):.3f}' if walls else ''),
                'n_false_unsat': n_false_unsat,
                'n_false_sat': n_false_sat,
            })
    return summary_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out-dir', type=Path, default=_OUT_DIR)
    args = p.parse_args()
    summary = aggregate(args.out_dir)
    print(f'Wrote {summary}')
    # Print the table to stdout for quick inspection.
    with open(summary) as f:
        print(f.read())


if __name__ == '__main__':
    main()
