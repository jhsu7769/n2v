"""Aggregate Exp 2 per-(benchmark, method) CSVs into ``exp2_comparison_table.csv``.

Same structure as :mod:`examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_aggregate`,
but with Exp 2-specific ground-truth dispatch:

* ``vit_2023``, ``tinyimagenet_2024``, ``cifar100_2024`` — VNN-COMP 2025
  sound-verifier consensus (αβ-CROWN ∪ NeuralSAT ∪ PyRAT ∪ NNV ∪
  NNEnum) read from
  ``~/v/other/VNNCOMP/vnncomp2025_results/<tool>/results.csv``.
* ``cifar10_resnet110`` — *no* VNN-COMP equivalent (the network is
  Cohen RS's pretrained 110-layer CIFAR-10 ResNet, not a competition
  benchmark). Per Exp 2 design, ground truth here is
  "the model's own clean prediction is robust within ε" — there's no
  external reference. We emit ``ground_truth='not_applicable'`` and
  skip false-UNSAT / false-SAT counting.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_aggregate
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Dict, Optional, Tuple

_HERE = Path(__file__).resolve().parent
_OUT_DIR = _HERE / 'outputs'
_GROUND_TRUTH_CSV = _HERE / 'ground_truth.csv'

# VNN-COMP-format benchmarks that have entries in ``ground_truth.csv``.
_BENCH_VNNCOMP = ('vit_2023', 'tinyimagenet_2024', 'cifar100_2024')
# cifar10_resnet110 has no VNN-COMP equivalent; emits
# ``ground_truth_source='not_applicable_resnet110'`` and skips
# false-UNSAT/SAT counting.
_BENCH_NON_VNNCOMP = ('cifar10_resnet110',)
_KNOWN_BENCHMARKS = _BENCH_VNNCOMP + _BENCH_NON_VNNCOMP

_AGGREGATE_FIELDS = [
    'benchmark', 'method',
    'n_instances',
    'n_unsat', 'n_unknown', 'n_sat', 'n_timeout', 'n_error', 'n_skipped',
    'median_wall_s', 'p10_wall_s', 'p90_wall_s',
    'n_false_unsat', 'n_false_sat', 'fur_pct', 'fsr_pct',
    'ground_truth_source',
]


def _parse_filename(name: str) -> Optional[Tuple[str, str]]:
    if not (name.startswith('exp2_') and name.endswith('.csv')):
        return None
    stem = name[len('exp2_'):-len('.csv')]
    for bench in _KNOWN_BENCHMARKS:
        if stem.startswith(bench + '_'):
            return bench, stem[len(bench) + 1:]
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
    :mod:`examples.FlowConformal.experiments.build_ground_truth` and
    return ``(onnx_basename, vnnlib_basename) -> ground_truth`` where
    ``ground_truth`` ∈ ``{'sat', 'unsat', 'unknown'}``.

    Returns an empty dict for non-VNN-COMP benchmarks (e.g.
    cifar10_resnet110) — caller should branch on that.
    """
    if bench not in _BENCH_VNNCOMP:
        return {}
    if not _GROUND_TRUTH_CSV.exists():
        raise FileNotFoundError(
            f'ground-truth CSV not found at {_GROUND_TRUTH_CSV}; '
            f'run `python -m examples.FlowConformal.experiments.'
            f'build_ground_truth` first.')
    out: Dict[Tuple[str, str], str] = {}
    with open(_GROUND_TRUTH_CSV, newline='') as f:
        for r in csv.DictReader(f):
            if r.get('benchmark', '').strip() != bench:
                continue
            onnx = r.get('onnx_file', '').strip()
            vnn = r.get('vnnlib_file', '').strip()
            gt = r.get('ground_truth', '').strip().lower()
            out[(onnx, vnn)] = gt
    return out


def _instance_keys(row: dict) -> Tuple[str, str]:
    """Get ``(onnx, vnnlib)`` keys for VNN-COMP ground-truth lookup.

    Different runners write different column names — αβ-CROWN runner
    writes ``onnx_file`` / ``vnnlib_file``, ours/Hashemi write
    ``instance`` (e.g. ``pgd_2_3_16.onnx+pgd_2_3_16_2446.vnnlib``).
    Try both shapes.
    """
    onnx = row.get('onnx_file', '').strip()
    vnn = row.get('vnnlib_file', '').strip()
    if onnx and vnn:
        return onnx, vnn
    inst = row.get('instance', '').strip()
    if '+' in inst:
        a, b = inst.split('+', 1)
        return a.strip(), b.strip()
    return '', ''


def aggregate(out_dir: Path = _OUT_DIR) -> Path:
    rows_by_cell: Dict[Tuple[str, str], list] = defaultdict(list)
    for p in sorted(out_dir.glob('exp2_*.csv')):
        if p.name in ('exp2_comparison_table.csv', 'exp2_summary_table.csv'):
            continue
        parsed = _parse_filename(p.name)
        if parsed is None:
            continue
        bench, method = parsed
        if method.endswith('_smoke') or method == 'smoke':
            continue
        with open(p, newline='') as f:
            for r in csv.DictReader(f):
                rows_by_cell[(bench, method)].append(r)

    summary_path = out_dir / 'exp2_comparison_table.csv'
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
            applicable_gt = bench in _BENCH_VNNCOMP
            for r in rows:
                v = r.get('verdict', '').strip()
                counts[v] += 1
                try:
                    walls.append(float(r['wall_s']))
                except (KeyError, ValueError):
                    pass
                if applicable_gt:
                    onnx_name, vnn_name = _instance_keys(r)
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
                'n_false_unsat': n_false_unsat if applicable_gt else '',
                'n_false_sat': n_false_sat if applicable_gt else '',
                'fur_pct': f'{fur_pct:.2f}' if applicable_gt else '',
                'fsr_pct': f'{fsr_pct:.2f}' if applicable_gt else '',
                'ground_truth_source': (
                    'vnncomp2025_sat_wins_8tools' if applicable_gt
                    else 'not_applicable_resnet110'),
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
