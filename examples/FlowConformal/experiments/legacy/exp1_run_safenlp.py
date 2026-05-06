"""Exp 1: VNN-COMP safenlp benchmark sweep — our method only.

Note: safenlp uses 30-D embeddings with two output classes. Properties
are simple ``Y_0 <= Y_1`` halfspaces.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_safenlp \
        --smoke
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
    run_sweep,
)


_BENCHMARK_NAME = 'safenlp'
_BENCHMARK_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/safenlp_2024'
))
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV_FULL = _OUT_DIR / f'exp1_{_BENCHMARK_NAME}_ours.csv'
_OUT_CSV_SMOKE = _OUT_DIR / f'exp1_{_BENCHMARK_NAME}_ours_smoke.csv'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='Run only first 2 instances; write to *_smoke.csv.')
    p.add_argument('--seeds', type=int, default=1)
    p.add_argument('--timeout', type=int, default=300)
    args = p.parse_args()
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = _OUT_CSV_SMOKE if args.smoke else _OUT_CSV_FULL
    run_sweep(
        benchmark_name=_BENCHMARK_NAME,
        benchmark_root=_BENCHMARK_ROOT,
        out_csv=out_csv,
        n_seeds=args.seeds,
        smoke_n_instances=2 if args.smoke else None,
        per_instance_timeout_s=args.timeout,
    )


if __name__ == '__main__':
    main()
