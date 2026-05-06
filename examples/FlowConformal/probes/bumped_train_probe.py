"""Bumped-training probe (read-only diagnostic).

Re-runs the production pipeline on a chosen instance with
``n_train`` / ``flow_epochs`` overridden upward from the per-benchmark
defaults. Used to test whether the false UNSATs on metaroom are flow
capacity / under-training failures (i.e., the flow doesn't see U
because there aren't enough training samples or epochs to capture the
adversarial direction). Mirrors the same upward bump applied earlier
to the acasxu / tinyimagenet / cifar100 false UNSATs (n_train=20K,
flow_epochs=5K) — 1 of 3 of those flipped to UNKNOWN, motivating
the same test on metaroom.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.bumped_train_probe \\
        --benchmark metaroom_2023 --instance-idx 14 \\
        --n-train 20000 --flow-epochs 5000
"""
from __future__ import annotations

import argparse
import time

import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.probabilistic.verify_flow import run_verification_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--n-train', type=int, required=True)
    p.add_argument('--flow-epochs', type=int, required=True)
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[bump] {args.benchmark} idx={args.instance_idx}: {vnn_rel}')
    print(f'[bump] bumped cfg: n_train={args.n_train} flow_epochs={args.flow_epochs}')

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]

    lb, ub = boxes[0]
    torch.manual_seed(args.seed)
    t0 = time.time()
    result = run_verification_pipeline(
        network=network, input_lb=lb, input_ub=ub, spec=spec,
        alpha=cfg['alpha'],
        n_train=args.n_train,
        flow_epochs=args.flow_epochs,
        flow_config=cfg['flow_config'],
        scenario_n_samples=cfg['scenario_n_samples'],
        scenario_beta=0.001,
        verification_method=cfg['verification_method'],
        amls_max_levels=cfg['amls_max_levels'],
        seed=args.seed,
        use_falsifier=cfg.get('use_falsifier', False),
    )
    wall = time.time() - t0
    print(f'[bump] verdict={result["verdict"]}  '
          f'eps_2={result.get("amls_bounded_eps_2_upper")}  '
          f'detected={result.get("amls_bounded_detected_unsafe")}  '
          f'levels={result.get("amls_levels_used")}  '
          f'wall={wall:.1f}s')


if __name__ == '__main__':
    main()
