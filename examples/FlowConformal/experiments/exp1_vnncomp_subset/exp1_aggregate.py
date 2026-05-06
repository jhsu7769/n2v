"""Aggregate Exp 1 per-(benchmark, method) CSVs into ``exp1_comparison_table.csv``.

Reads every ``outputs/exp1_<benchmark>_<method>.csv`` and produces a
comparison table with one row per ``(benchmark, method)``:

    benchmark, method,
    n_instances, n_unsat, n_unknown, n_sat, n_timeout, n_error, n_skipped,
    median_wall_s, p10_wall_s, p90_wall_s,
    n_false_unsat, n_false_sat, fur_pct, fsr_pct,
    ground_truth_source

``fur_pct`` (false-UNSAT rate %) and ``fsr_pct`` (false-SAT rate %) are
computed against VNN-COMP 2025 sound-verifier consensus — read from the
αβ-CROWN / NeuralSAT / PyRAT / NNV / NNEnum results CSVs at
``~/v/other/VNNCOMP/vnncomp2025_results/<tool>/<bench_dir>/results.csv``.
A verdict is "true" (consensus) if any sound verifier returned that
label, "conflict" if both UNSAT and SAT appear, "unknown" if neither.

Soundness columns:
    * ``n_false_unsat`` = rows where our verdict=UNSAT but consensus=sat
    * ``n_false_sat``   = rows where our verdict=SAT but consensus=unsat
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'outputs'

_GROUND_TRUTH_CSV = _HERE / 'ground_truth.csv'

# Known benchmarks for filename parsing. Update when the Exp 1 roster
# changes (also update build_ground_truth.py). Forward-compat list:
# includes both the original roster and the malbeware/metaroom_2023
# additions.
_KNOWN_BENCHMARKS = (
    'acasxu_2023',
    'collins_rul_cnn_2022',
    'dist_shift_2023',
    'linearizenn_2024',
    'tllverify_2023',
    'malbeware',
    'metaroom_2023',
    # New multi-output benches (smoke-decided whether to include in the
    # final table). Listed here so the aggregator parses
    # exp1_<bench>_<method>.csv even before they're committed to the sweep.
    'lsnc_relu',
    'relusplitter',
)

_AGGREGATE_FIELDS = [
    'benchmark', 'method',
    'n_instances',
    'n_unsat', 'n_unknown', 'n_sat', 'n_timeout', 'n_error', 'n_skipped',
    'median_wall_s', 'p10_wall_s', 'p90_wall_s',
    'n_false_unsat', 'n_false_sat', 'fur_pct', 'fsr_pct',
    'ground_truth_source',
]

def _parse_filename(name: str) -> Optional[Tuple[str, str]]:
    """Parse ``exp1_<benchmark>_<method>.csv`` into ``(bench, method)``.

    Greedy regex ambiguity (e.g. ``acasxu_2023_hashemi_clipping`` could
    parse as bench=``acasxu_2023_hashemi``, method=``clipping``) is
    resolved by checking each known benchmark as a prefix.
    """
    if not (name.startswith('exp1_') and name.endswith('.csv')):
        return None
    stem = name[len('exp1_'):-len('.csv')]
    for bench in _KNOWN_BENCHMARKS:
        if stem.startswith(bench + '_'):
            method = stem[len(bench) + 1:]
            return bench, method
    return None


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


def load_ground_truth(bench: str) -> Dict[Tuple[str, str], str]:
    """Read the pre-computed ground-truth CSV produced by
    :mod:`build_ground_truth` and return a dict
    ``(onnx_basename, vnnlib_basename) -> ground_truth`` where
    ``ground_truth`` ∈ ``{'sat', 'unsat', 'unknown'}``.

    The SAT-wins rule applied in ``build_ground_truth`` means there is
    no ``'conflict'`` value here — any disagreement resolves to SAT
    when at least one tool reports SAT, otherwise to UNSAT/unknown.
    """
    out: Dict[Tuple[str, str], str] = {}
    if not _GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(
            f'ground-truth CSV not found at {_GROUND_TRUTH_CSV}; '
            f'run `python -m examples.FlowConformal.experiments.'
            f'build_ground_truth` first to generate it.')
    with open(_GROUND_TRUTH_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('benchmark', '').strip() != bench:
                continue
            onnx = r.get('onnx_file', '').strip()
            vnn = r.get('vnnlib_file', '').strip()
            gt = r.get('ground_truth', '').strip().lower()
            out[(onnx, vnn)] = gt
    return out


def aggregate(out_dir: Path = _OUT_DIR) -> Path:
    """Aggregate per-(benchmark, method) CSVs and write
    ``exp1_comparison_table.csv``.
    """
    rows_by_cell: Dict[Tuple[str, str], list] = defaultdict(list)
    for p in sorted(out_dir.glob('exp1_*.csv')):
        if p.name == 'exp1_comparison_table.csv':
            continue
        parsed = _parse_filename(p.name)
        if parsed is None:
            continue
        bench, method = parsed
        # Skip legacy '_smoke' CSVs from earlier prototyping.
        if method.endswith('_smoke') or method == 'smoke':
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                rows_by_cell[(bench, method)].append(r)

    summary_path = out_dir / 'exp1_comparison_table.csv'
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_AGGREGATE_FIELDS)
        w.writeheader()
        for (bench, method), rows in sorted(rows_by_cell.items()):
            consensus = load_ground_truth(bench)
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
                onnx_name = r.get('onnx_file', '').strip()
                vnn_name = r.get('vnnlib_file', '').strip()
                gt = consensus.get((onnx_name, vnn_name), 'unknown')
                if v == 'UNSAT' and gt == 'sat':
                    n_false_unsat += 1
                elif v == 'SAT' and gt == 'unsat':
                    n_false_sat += 1
            n = len(rows)
            n_unsat = counts.get('UNSAT', 0)
            n_sat = counts.get('SAT', 0)
            fur_pct = (100.0 * n_false_unsat / n_unsat) if n_unsat else 0.0
            fsr_pct = (100.0 * n_false_sat / n_sat) if n_sat else 0.0
            w.writerow({
                'benchmark': bench, 'method': method,
                'n_instances': n,
                'n_unsat': n_unsat,
                'n_unknown': counts.get('UNKNOWN', 0),
                'n_sat': n_sat,
                'n_timeout': counts.get('TIMEOUT', 0),
                'n_error': counts.get('ERROR', 0),
                'n_skipped': counts.get('SKIPPED', 0),
                'median_wall_s': (
                    f'{median(walls):.3f}' if walls else ''),
                'p10_wall_s': (
                    f'{_percentile(walls, 0.10):.3f}' if walls else ''),
                'p90_wall_s': (
                    f'{_percentile(walls, 0.90):.3f}' if walls else ''),
                'n_false_unsat': n_false_unsat,
                'n_false_sat': n_false_sat,
                'fur_pct': f'{fur_pct:.2f}',
                'fsr_pct': f'{fsr_pct:.2f}',
                'ground_truth_source': 'vnncomp2025_sat_wins_8tools',
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
