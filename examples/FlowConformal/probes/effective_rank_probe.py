"""Effective-rank probe (read-only diagnostic).

Asks: how rank-deficient is the network's reach set in practice? If
the singular values of (y_tr - mean) decay sharply, PCA preprocessing
would let a full-rank flow on the projected space avoid the lobe-
smearing failure mode. If singular values are flat, PCA isn't going
to help.

The probe also reports per-rank reconstruction error, so we can pick
a target k that captures (1-eps) of the variance.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.effective_rank_probe \\
        --benchmark lsnc_relu --instance-idx 0
"""
from __future__ import annotations

import argparse

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG, list_instances, load_one_instance,
)
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import _forward


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--n-samples', type=int, default=5000,
                   help='Number of (x, y) pairs to compute SVD on.')
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, _t = instances[args.instance_idx]
    print(f'[rank] {args.benchmark} idx={args.instance_idx}: {vnn_rel}',
          flush=True)

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    input_dim = int(np.prod(lb.shape))

    print(f'[rank] sampling {args.n_samples} inputs (input_dim={input_dim})',
          flush=True)
    x = _sample_box(lb_t, ub_t, n_samples=args.n_samples, seed=args.seed)
    y = _forward(network, x).detach().cpu().numpy()
    output_dim = y.shape[1]
    print(f'[rank] output_dim={output_dim}', flush=True)

    # Center and SVD
    y_centered = y - y.mean(axis=0, keepdims=True)
    # Standardize each output dim so dims with different scales don't
    # dominate the singular spectrum (matches what production whitening does).
    y_std = y_centered.std(axis=0, keepdims=True).clip(min=1e-8)
    y_w = y_centered / y_std

    U, S, Vt = np.linalg.svd(y_w, full_matrices=False)
    # Singular values squared / N gives variance explained per component
    var = (S ** 2) / args.n_samples
    var_total = var.sum()
    var_frac = var / var_total
    cum_var = np.cumsum(var_frac)

    print()
    print(f'[rank] ===== Singular spectrum of whitened y_tr =====')
    print(f'  k    sing_val      var_frac      cum_var')
    for k in range(len(S)):
        print(f'  {k+1:>2}   {S[k]:>10.4f}   {var_frac[k]:>10.4f}    {cum_var[k]:>7.4f}')

    # Effective rank thresholds
    print()
    for thresh in [0.50, 0.80, 0.90, 0.95, 0.99, 0.999]:
        k_needed = int(np.searchsorted(cum_var, thresh) + 1)
        print(f'[rank] k for {thresh*100:>5.1f}% variance: {k_needed}/{output_dim}')

    # Hypothesis: input_dim < output_dim => rank deficiency expected
    if input_dim < output_dim:
        print()
        print(f'[rank] input_dim ({input_dim}) < output_dim ({output_dim}) → '
              f'reach manifold has intrinsic dim ≤ {input_dim}')
        # How much variance is captured by the first input_dim components?
        var_at_input_dim = float(cum_var[input_dim - 1])
        print(f'[rank]   variance captured by k={input_dim}: '
              f'{var_at_input_dim:.4f}')
        if var_at_input_dim >= 0.99:
            print(f'[rank] ✅ STRONG case for PCA: top {input_dim} components '
                  f'capture {var_at_input_dim:.1%} of variance.')
            print(f'[rank]    Flow trained on PCA-projected outputs would have')
            print(f'[rank]    full-rank density on the projected space → no')
            print(f'[rank]    structural off-manifold mass.')
        elif var_at_input_dim >= 0.90:
            print(f'[rank] ~ MODERATE case for PCA: top {input_dim} components')
            print(f'[rank]    capture {var_at_input_dim:.1%}. Some residual.')
        else:
            print(f'[rank] ❌ WEAK case for PCA: top {input_dim} components only')
            print(f'[rank]    capture {var_at_input_dim:.1%}. The reach manifold')
            print(f'[rank]    is not well approximated by a linear subspace.')
            print(f'[rank]    PCA likely does not help; need nonlinear projection.')


if __name__ == '__main__':
    main()
