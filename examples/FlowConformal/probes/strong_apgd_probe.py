"""Strong-APGD probe (read-only diagnostic).

Tests whether bumping the falsifier budget catches the false UNSATs
that the current production config (3 restarts × 25 steps) misses.

Concrete check: load the ground-truth-SAT instances where our
framework reported UNSAT (false UNSATs), run APGD with a substantially
bumped budget (50 restarts × 200 steps by default — 13× more than
production), and report whether APGD finds a counterexample.

If yes → bumping the falsifier budget would correctly flip these to
SAT, eliminating the false UNSATs.

If no → these are truly hard counterexamples beyond what gradient-
based falsification can find at any reasonable budget.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.strong_apgd_probe \\
        --benchmark acasxu_2023 --instance-idx 45
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    list_instances, load_one_instance,
)
from n2v.utils.falsify import falsify


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--method', type=str, default='apgd',
                   choices=['random', 'pgd', 'apgd', 'random+pgd',
                            'random+pgd+apgd'],
                   help='Falsifier method (default apgd).')
    p.add_argument('--restarts', type=int, default=50,
                   help='APGD/PGD restarts.')
    p.add_argument('--steps', type=int, default=200,
                   help='APGD/PGD steps per restart.')
    p.add_argument('--n-samples', type=int, default=100000,
                   help='Random samples (for random/random+pgd/cascade).')
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[apgd] {args.benchmark} idx={args.instance_idx}: {vnn_rel}',
          flush=True)

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    lb, ub = boxes[0]

    print(f'[apgd] method={args.method}  restarts={args.restarts}  '
          f'steps={args.steps}  n_samples={args.n_samples}', flush=True)
    t0 = time.time()
    fr_result, fr_cex = falsify(
        network, lb, ub, spec, method=args.method,
        n_restarts=args.restarts, n_steps=args.steps,
        n_samples=args.n_samples, seed=args.seed,
    )
    wall = time.time() - t0
    verdict = 'SAT' if fr_result == 0 else 'UNKNOWN'
    print(f'[apgd] verdict={verdict}  wall={wall:.1f}s', flush=True)

    if fr_cex is not None:
        x_cex, y_cex = fr_cex
        from n2v.utils.verify_specification import _parse_property_groups
        groups = _parse_property_groups(spec)
        # Compute phi to confirm in-U
        per_group = []
        for group in groups:
            per_hs = []
            for hs in group:
                G = np.asarray(hs.G, dtype=np.float64)
                g = np.asarray(hs.g, dtype=np.float64).flatten()
                per_hs.append(float((G @ y_cex.flatten() - g).max()))
            per_group.append(min(per_hs))
        cex_phi = float(max(per_group))
        print(f'[apgd] counterexample phi(network(x))={cex_phi:.4e}  '
              f'(<=0 means in U)', flush=True)


if __name__ == '__main__':
    main()
