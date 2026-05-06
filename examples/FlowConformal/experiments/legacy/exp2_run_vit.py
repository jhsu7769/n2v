"""Exp 2: VNN-COMP vit_2023 sweep (our method only).

Runs the locked Phase 5d AMLS pipeline on the vit_2023 instances. αβ-CROWN
and other sound verifiers are documented at analysis time as having
limited or zero coverage on this benchmark — Exp 2 measures our scaling
behavior, not theirs.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_vit \\
        --smoke
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
    load_instance, parse_instances_csv,
)
from examples.FlowConformal.experiments.exp2_prob_scale._common import (
    add_common_args, get_pipeline_kwargs, run_sweep,
)


_BENCHMARK_NAME = 'vit_2023'
_BENCHMARK_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/vit_2023'
))
_OUT_DIR = Path(__file__).parent / 'outputs'


def _make_loader(onnx_rel: str, vnn_rel: str):
    name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'

    def _load():
        network, boxes, spec = load_instance(_BENCHMARK_ROOT, onnx_rel, vnn_rel)
        return network, boxes, spec, name

    return name, _load


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()

    instances_csv = _BENCHMARK_ROOT / 'instances.csv'
    if not instances_csv.exists():
        raise SystemExit(f'instances.csv not found at {instances_csv}')

    rows = parse_instances_csv(instances_csv)
    n = 2 if args.smoke else args.instances
    rows = rows[:n]
    instances = [_make_loader(o, v) for (o, v, _t) in rows]

    suffix = '_smoke' if args.smoke else ''
    falsify_tag = '_falsify' if args.falsify_first else ''
    default = _OUT_DIR / f'exp2_{_BENCHMARK_NAME}_ours{falsify_tag}{suffix}.csv'
    out_csv = args.output_csv or default

    run_sweep(
        benchmark_name=_BENCHMARK_NAME,
        instances=instances,
        out_csv=out_csv,
        pipeline_kwargs=get_pipeline_kwargs(args.falsify_first),
        timeout_s=args.timeout,
        seed=args.seed,
    )


if __name__ == '__main__':
    main()
