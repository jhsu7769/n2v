# DEPRECATED for main runs — kept as smoke artefact only.
"""Exp 1: VNN-COMP metaroom benchmark sweep — our method only.

DEPRECATED for main runs. The 2026-04-27 paper-experiments plan drops
metaroom from the seven-benchmark VNN-COMP execution order because the
ONNX export hard-codes ``batch=1`` along several reshape ops, which
forces the generic wrapper into a per-sample loop that pushes per-instance
wall-clock beyond the 116 s VNN-COMP budget. The script remains so that
existing smoke artefacts (``outputs/exp1_metaroom_ours_smoke.csv``) stay
reproducible, but it is **NOT** part of the Exp 1 sweep.

Note: metaroom uses a CNN with input shape ``(3, 32, 56)`` and output
``(20,)``. The generic wrapper auto-detects this shape; smoke tests
verify load + run.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
        examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_metaroom \
        --smoke
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
    run_sweep,
)


_BENCHMARK_NAME = 'metaroom'
_BENCHMARK_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/metaroom_2023'
))
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV_FULL = _OUT_DIR / f'exp1_{_BENCHMARK_NAME}_ours.csv'
_OUT_CSV_SMOKE = _OUT_DIR / f'exp1_{_BENCHMARK_NAME}_ours_smoke.csv'


def main():
    print('[metaroom] DEPRECATED for main runs — kept as smoke artefact only.',
          flush=True)
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
