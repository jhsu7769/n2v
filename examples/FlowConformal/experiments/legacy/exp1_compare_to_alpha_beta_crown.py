"""Compare Exp 1 results against VNN-COMP sound verifier CSVs.

For each verifier in ``_VERIFIERS`` and each benchmark in ``_BENCHMARKS``,
joins our CSV with the verifier's ``results.csv`` by (onnx_basename,
vnnlib_basename) and prints / writes:

  - Agreement count (UNSAT-UNSAT, SAT-SAT)
  - False UNSAT count (we say UNSAT, verifier says SAT) — soundness violation
  - False SAT count (we say SAT, verifier says UNSAT) — falsifier bug
  - Coverage (ours: how often we returned a non-UNKNOWN verdict)

VNN-COMP results CSV format (5-column, no header):
    benchmark, onnx_path, vnnlib_path, prep_time, verdict, verify_time

Our CSV format (with header):
    benchmark, onnx_file, vnnlib_file, seed, verdict, ...

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_compare_to_alpha_beta_crown
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path


# Map our short benchmark name -> VNN-COMP results subdir.
_BENCHMARKS = {
    'acasxu': '2025_acasxu_2023',
    'collins_rul_cnn': '2025_collins_rul_cnn_2022',
    'cora': '2025_cora_2024',
    'dist_shift': '2025_dist_shift_2023',
    'linearizenn': '2025_linearizenn_2024',
    'metaroom': '2025_metaroom_2023',
    'safenlp': '2025_safenlp_2024',
    'tllverify': '2025_tllverifybench_2023',
}

# Sound verifiers we compare against.
_VERIFIERS = ['alpha_beta_crown', 'neuralsat', 'pyrat', 'cora', 'nnenum',
              'nnv']

_VNNCOMP_RESULTS_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_results'))

_OUR_OUTPUT_DIR = Path(__file__).parent / 'outputs'

# ACAS Xu lives in the ablations dir, not in our outputs dir; allow
# discovery from there.
_ACASXU_OUR_CSV = (Path(__file__).resolve().parents[2] /
                   'ablations' / 'outputs' /
                   'acasxu_sweep_flow_conformal.csv')


def _normalize_verdict(v: str) -> str:
    """Normalise verdict strings across our + VNN-COMP CSVs to one of
    {'sat', 'unsat', 'unknown', 'timeout', 'error', 'skipped'}.
    """
    if v is None:
        return 'unknown'
    v = v.strip().lower()
    # Our pipeline emits SAT/UNSAT/UNKNOWN/SKIPPED/TIMEOUT/ERROR.
    # VNN-COMP emits sat/unsat/unknown/timeout/error/holds/violated.
    if v in ('sat', 'violated'):
        return 'sat'
    if v in ('unsat', 'holds'):
        return 'unsat'
    if v in ('timeout',):
        return 'timeout'
    if v in ('error',):
        return 'error'
    if v in ('skipped',):
        return 'skipped'
    return 'unknown'


def _our_csv(benchmark: str, smoke: bool = False) -> Path:
    """Return path to our per-benchmark results CSV.

    For ACAS Xu: ablations/outputs/acasxu_sweep_flow_conformal.csv.
    For others: outputs/exp1_<bench>_ours[_smoke].csv.
    """
    if benchmark == 'acasxu':
        return _ACASXU_OUR_CSV
    suffix = '_smoke' if smoke else ''
    return _OUR_OUTPUT_DIR / f'exp1_{benchmark}_ours{suffix}.csv'


def _load_our_csv(path: Path) -> dict[tuple[str, str], str]:
    """Read our CSV and return {(onnx_basename, vnnlib_basename): verdict_norm}.

    If multiple seeds exist, takes the first non-UNKNOWN verdict (or the
    last UNKNOWN if all are UNKNOWN).
    """
    if not path.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    seen: dict[tuple[str, str], list[str]] = defaultdict(list)
    with open(path, newline='') as f:
        reader = csv.DictReader(f)
        # ACAS Xu's CSV uses a slightly different schema: no 'benchmark'
        # / 'seed' columns. Detect by header.
        for row in reader:
            onnx = row.get('onnx_file', '').strip()
            vnn = row.get('vnnlib_file', '').strip()
            if not onnx or not vnn:
                continue
            v = _normalize_verdict(row.get('verdict', ''))
            seen[(onnx, vnn)].append(v)
    for k, vlist in seen.items():
        # prefer a determinate verdict over UNKNOWN/SKIPPED/ERROR/TIMEOUT
        nonun = [v for v in vlist if v in ('sat', 'unsat')]
        out[k] = nonun[0] if nonun else vlist[0]
    return out


def _load_verifier_csv(verifier: str, vnncomp_dir: str
                        ) -> dict[tuple[str, str], str]:
    """Read VNN-COMP per-verifier per-benchmark results.csv. Returns
    {(onnx_basename, vnnlib_basename): verdict_norm}. Empty if missing.
    """
    p = _VNNCOMP_RESULTS_ROOT / verifier / vnncomp_dir / 'results.csv'
    if not p.exists():
        return {}
    out: dict[tuple[str, str], str] = {}
    with open(p, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 5:
                continue
            onnx = Path(row[1].strip()).name
            vnn = Path(row[2].strip()).name
            verdict = _normalize_verdict(row[4])
            out[(onnx, vnn)] = verdict
    return out


def _agreement(ours: dict, theirs: dict) -> dict:
    """Compute agreement metrics between our verdicts and a verifier's.
    Only over keys present in BOTH dicts.
    """
    keys = set(ours) & set(theirs)
    n = len(keys)
    counts = {
        'overlap': n,
        'unsat_unsat': 0, 'sat_sat': 0,
        'false_unsat': 0,    # we unsat, they sat (soundness violation)
        'false_sat': 0,      # we sat, they unsat (falsifier bug)
        'we_decisive_they_not': 0,
        'they_decisive_we_not': 0,
        'both_indecisive': 0,
    }
    for k in keys:
        a = ours[k]; b = theirs[k]
        a_dec = a in ('sat', 'unsat')
        b_dec = b in ('sat', 'unsat')
        if a == 'unsat' and b == 'unsat':
            counts['unsat_unsat'] += 1
        elif a == 'sat' and b == 'sat':
            counts['sat_sat'] += 1
        elif a == 'unsat' and b == 'sat':
            counts['false_unsat'] += 1
        elif a == 'sat' and b == 'unsat':
            counts['false_sat'] += 1
        elif a_dec and not b_dec:
            counts['we_decisive_they_not'] += 1
        elif b_dec and not a_dec:
            counts['they_decisive_we_not'] += 1
        else:
            counts['both_indecisive'] += 1
    return counts


def main():
    print('=' * 78)
    print(f'Exp 1: comparison against VNN-COMP 2025 sound verifiers')
    print('=' * 78)
    rows = []
    for bench, vnncomp_dir in _BENCHMARKS.items():
        # Try smoke first; if not present, try full.
        for smoke in (False, True):
            our_path = _our_csv(bench, smoke=smoke)
            if our_path.exists():
                break
        ours = _load_our_csv(our_path)
        if not ours:
            print(f'\n[{bench}] no CSV at {our_path}; skipping')
            continue
        print(f'\n[{bench}] our CSV: {our_path.name}  ({len(ours)} rows)')
        for verifier in _VERIFIERS:
            theirs = _load_verifier_csv(verifier, vnncomp_dir)
            if not theirs:
                continue
            c = _agreement(ours, theirs)
            print(f'  vs {verifier:<20s}  '
                  f'overlap={c["overlap"]:4d}  '
                  f'agree(unsat/sat)={c["unsat_unsat"]}/'
                  f'{c["sat_sat"]}  '
                  f'FU={c["false_unsat"]}  FS={c["false_sat"]}  '
                  f'we-only={c["we_decisive_they_not"]}  '
                  f'them-only={c["they_decisive_we_not"]}  '
                  f'both?={c["both_indecisive"]}')
            rows.append({
                'benchmark': bench, 'verifier': verifier, **c,
            })

    # Optional: dump aggregate to CSV for plotting / table-generation.
    out_csv = _OUR_OUTPUT_DIR / 'exp1_comparison_table.csv'
    if rows:
        _OUR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        fields = ['benchmark', 'verifier', 'overlap', 'unsat_unsat',
                  'sat_sat', 'false_unsat', 'false_sat',
                  'we_decisive_they_not', 'they_decisive_we_not',
                  'both_indecisive']
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f'\nAggregate written to {out_csv}')


if __name__ == '__main__':
    main()
