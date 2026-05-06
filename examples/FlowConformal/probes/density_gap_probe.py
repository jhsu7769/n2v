"""Density-gap probe (read-only diagnostic).

Tests the hypothesis: if AMLS-found witnesses on lsnc_relu UNSAT-GT
instances are flow over-extrapolation artifacts, they should live in
LOW-DENSITY regions of the flow's output distribution. A density-based
conformal score would cut them off; the current latent-norm score
doesn't.

Concrete check: train the flow as usual, compute negative log-density
(``LogDetFlowScore``) for:

  * the m calibration outputs (the "in-distribution" reference)
  * the AMLS witness ``y*`` returned by ``amls_bounded_certify_spec_union``

If ``log p(y*)`` is in the deep tail of the calibration log-density
distribution (e.g., below the 1st percentile), then density-based
conformal calibration would exclude ``y*`` from the calibrated set →
AMLS would never search there → spurious-witness failures collapse.

If ``log p(y*)`` is near the median of the calibration log-densities,
the witness is in a high-density region of the flow and density-based
scoring would NOT help.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.density_gap_probe \\
        --benchmark lsnc_relu --instance-idx 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import _forward, run_verification_pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--n-cal', type=int, default=2000,
                   help='Calibration samples to compute density on.')
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[dgap] {args.benchmark} idx={args.instance_idx}: {vnn_rel}')

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    # ---- Train flow + run AMLS as usual ----
    print('[dgap] running pipeline (train + AMLS)...')
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    result = run_verification_pipeline(
        network=network, input_lb=lb, input_ub=ub, spec=spec,
        alpha=cfg['alpha'],
        n_train=cfg['n_train'],
        flow_epochs=cfg['flow_epochs'],
        flow_config=cfg['flow_config'],
        scenario_n_samples=cfg['scenario_n_samples'],
        scenario_beta=0.001,
        verification_method=cfg['verification_method'],
        amls_max_levels=cfg['amls_max_levels'],
        seed=args.seed,
        use_falsifier=False,
    )
    pipe_wall = time.time() - t0
    print(f'[dgap] pipeline: verdict={result["verdict"]}  '
          f'eps_2={result.get("amls_bounded_eps_2_upper")}  '
          f'detected={result.get("amls_bounded_detected_unsafe")}  '
          f'wall={pipe_wall:.1f}s')

    flow = result['flow']
    y_mean_np = result['y_mean']
    y_std_np = result['y_std']
    # Keep all whitening / score computation on CPU since the trained
    # FlowMatching model returned by the pipeline lives on CPU (see the
    # ``f_j.to('cpu').eval()`` line in verify_flow.py). Push y values
    # back to CPU before whitening.
    y_mean = torch.as_tensor(y_mean_np, dtype=torch.float32)
    y_std = torch.as_tensor(y_std_np, dtype=torch.float32)

    # ---- Pull AMLS witness y* from per-HS results ----
    # The pipeline nests amls_bounded_result inside scenario_result, not
    # at the top level — fetched here defensively from both spots.
    amls_b_result = (
        result.get('amls_bounded_result')
        or (result.get('scenario_result') or {}).get('amls_bounded_result')
    )
    witness_y_w = None  # whitened witness (LogDetFlowScore expects whitened)
    if amls_b_result is not None:
        for group in (amls_b_result.per_hs_results or []):
            for hs_res in group:
                if (hs_res is not None
                        and getattr(hs_res, 'detected_unsafe', False)
                        and getattr(hs_res, 'worst_y', None) is not None):
                    # worst_y is already in whitened y-space (AMLS runs there)
                    witness_y_w = torch.as_tensor(
                        np.asarray(hs_res.worst_y), dtype=torch.float32)
                    break
            if witness_y_w is not None:
                break

    if witness_y_w is None:
        print('[dgap] no detected witness (AMLS chain exhausted) — '
              'density-gap test only meaningful for over-extrapolation cases.')
        return

    # ---- Compute LogDetFlowScore for n_cal calibration samples ----
    cal_seed = args.seed + 1_000_000
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    x_ca = _sample_box(lb_t, ub_t, n_samples=args.n_cal, seed=cal_seed)
    y_ca = _forward(network, x_ca).detach().cpu()
    y_ca_w = (y_ca - y_mean) / y_std

    # LogDetFlowScore returns the negative log-density up to a constant:
    # s(y) = (1/2)||z||^2 + ∫trace(dv/dx)ds, monotone in -log p(y).
    # Higher s = lower density = more outlier-like.
    score_fn = LogDetFlowScore(flow, t=1.0, n_steps=30, method='rk4',
                               batch_size=256)
    print(f'[dgap] computing log-det density score for {args.n_cal} '
          f'calibration samples + 1 witness...')
    t1 = time.time()
    cal_scores = score_fn(y_ca_w).detach().cpu().numpy()
    witness_score = float(score_fn(witness_y_w.unsqueeze(0)).item())
    score_wall = time.time() - t1
    print(f'[dgap] density scoring took {score_wall:.1f}s')

    # ---- Report distribution + gap ----
    cal_sorted = np.sort(cal_scores)
    pcts = [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]
    print()
    print('[dgap] ========= Calibration -log-density score distribution =========')
    print('  (higher score = lower density = more outlier-like)')
    for pct in pcts:
        v = float(np.percentile(cal_scores, pct))
        print(f'  pct {pct:>5}: {v:>10.4f}')

    print()
    print(f'[dgap] AMLS witness score: {witness_score:.4f}')

    # Empirical CDF: fraction of calibration scores BELOW witness score.
    # If witness is at, say, 99.9% of cal scores, then a density threshold
    # at the 99.9% level would EXCLUDE this witness.
    frac_below = float((cal_scores < witness_score).mean())
    rank = int(np.searchsorted(cal_sorted, witness_score))
    print(f'[dgap] witness rank: {rank}/{args.n_cal} '
          f'(empirical CDF = {frac_below:.4f})')
    print()

    if frac_below >= 0.99:
        print('[dgap] ✅ WITNESS IS A DENSITY OUTLIER.')
        print('[dgap]    Conformal density threshold at 99% would EXCLUDE this')
        print('[dgap]    witness from the calibrated set. Density-based scoring')
        print('[dgap]    would plausibly fix the over-extrapolation false UNKNOWN.')
    elif frac_below >= 0.5:
        print(f'[dgap] ~ WITNESS IS IN THE TAIL (rank {frac_below:.1%}).')
        print('[dgap]   A tighter density threshold (e.g., 99.5%) might cut it,')
        print('[dgap]   at the cost of stricter coverage. Worth a follow-up.')
    else:
        print(f'[dgap] ❌ WITNESS IS IN A HIGH-DENSITY REGION (rank {frac_below:.1%}).')
        print('[dgap]    Density-based scoring would NOT exclude it. The flow')
        print('[dgap]    has substantial mass on U; the spurious extrapolation')
        print('[dgap]    is not a low-density-tail phenomenon.')


if __name__ == '__main__':
    main()
