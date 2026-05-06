"""kNN-distance score probe (read-only diagnostic).

Tests whether a kNN-distance nonconformity score separates spurious
from real AMLS witnesses on lsnc_relu, where the density-based score
gave inconsistent results (worked on inst 0 tail extrapolation, failed
on inst 14 lobe smearing).

Hypothesis: kNN distance to training samples naturally adapts to the
true reach manifold without making continuity / smoothness
assumptions. A spurious witness (whether tail or lobe) should be far
from all training samples in some sense; a real witness should sit
close to the training-data-defined manifold.

Concrete check: train the flow as usual, get the AMLS witness y*,
compute kNN distance from y* to the m training outputs (k=10), and
compare to the kNN-distance distribution of the calibration set.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.knn_score_probe \\
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
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import _forward, run_verification_pipeline


def knn_distance(query: np.ndarray, refs: np.ndarray, k: int = 10) -> np.ndarray:
    """For each row of ``query``, return the mean Euclidean distance to
    its ``k`` nearest neighbors in ``refs``.

    query: (Q, d), refs: (R, d). Returns (Q,) array.
    """
    # Vectorized pairwise distance (memory-permitting)
    # ||q - r||^2 = ||q||^2 + ||r||^2 - 2 q@r
    q2 = (query * query).sum(axis=1)[:, None]  # (Q, 1)
    r2 = (refs * refs).sum(axis=1)[None, :]    # (1, R)
    cross = query @ refs.T                      # (Q, R)
    d2 = q2 + r2 - 2 * cross
    d2 = np.maximum(d2, 0)  # numerical floor
    d = np.sqrt(d2)
    # Take mean of k nearest
    if k >= refs.shape[0]:
        return d.mean(axis=1)
    sorted_d = np.sort(d, axis=1)
    return sorted_d[:, :k].mean(axis=1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--k', type=int, default=10,
                   help='Number of nearest neighbors to average over.')
    p.add_argument('--n-cal', type=int, default=2000)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[knn] {args.benchmark} idx={args.instance_idx}: {vnn_rel}', flush=True)

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    # ---- Train flow + run AMLS as usual to get the witness ----
    print('[knn] running pipeline (train + AMLS)...', flush=True)
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
    print(f'[knn] pipeline: verdict={result["verdict"]}  '
          f'eps_2={result.get("amls_bounded_eps_2_upper")}  '
          f'detected={result.get("amls_bounded_detected_unsafe")}  '
          f'wall={pipe_wall:.1f}s', flush=True)

    y_mean_np = result['y_mean']
    y_std_np = result['y_std']
    y_mean = torch.as_tensor(y_mean_np, dtype=torch.float32)
    y_std = torch.as_tensor(y_std_np, dtype=torch.float32)

    # ---- Pull AMLS witness y* (whitened) ----
    amls_b_result = (
        result.get('amls_bounded_result')
        or (result.get('scenario_result') or {}).get('amls_bounded_result')
    )
    witness_y_w = None
    if amls_b_result is not None:
        for group in (amls_b_result.per_hs_results or []):
            for hs_res in group:
                if (hs_res is not None
                        and getattr(hs_res, 'detected_unsafe', False)
                        and getattr(hs_res, 'worst_y', None) is not None):
                    witness_y_w = np.asarray(hs_res.worst_y, dtype=np.float64)
                    break
            if witness_y_w is not None:
                break

    if witness_y_w is None:
        print('[knn] no detected witness — kNN-distance test only meaningful for')
        print('[knn]   over-extrapolation cases.', flush=True)
        return

    # ---- Reference set: m training outputs (whitened), as the
    # ---- "true reach approximation" the kNN compares against ----
    train_seed = args.seed  # default flow_seed = seed in pipeline
    n_train = cfg['n_train']
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    print(f'[knn] regenerating {n_train} TRAINING outputs as kNN reference...',
          flush=True)
    x_tr = _sample_box(lb_t, ub_t, n_samples=n_train, seed=train_seed)
    y_tr = _forward(network, x_tr).detach().cpu()
    y_tr_w = ((y_tr - y_mean) / y_std).numpy()

    # Calibration outputs (whitened) — to get the kNN-distance
    # distribution of "in-distribution" points.
    cal_seed = args.seed + 1_000_000
    x_ca = _sample_box(lb_t, ub_t, n_samples=args.n_cal, seed=cal_seed)
    y_ca = _forward(network, x_ca).detach().cpu()
    y_ca_w = ((y_ca - y_mean) / y_std).numpy()

    # ---- Compute kNN distances ----
    print(f'[knn] computing k={args.k}-NN distances...', flush=True)
    t1 = time.time()
    cal_knn = knn_distance(y_ca_w, y_tr_w, k=args.k)
    witness_knn = knn_distance(witness_y_w[None, :], y_tr_w, k=args.k)[0]
    knn_wall = time.time() - t1
    print(f'[knn] kNN scoring took {knn_wall:.1f}s', flush=True)

    # ---- Report ----
    cal_sorted = np.sort(cal_knn)
    pcts = [1, 5, 25, 50, 75, 95, 99, 99.5, 99.9, 100]
    print()
    print(f'[knn] ========= Calibration k={args.k}-NN distance distribution '
          f'=========')
    print('  (higher distance = farther from training data = more outlier-like)')
    for pct in pcts:
        v = float(np.percentile(cal_knn, pct))
        print(f'  pct {pct:>5}: {v:>10.4f}')

    print()
    print(f'[knn] AMLS witness kNN-distance: {witness_knn:.4f}')

    frac_below = float((cal_knn < witness_knn).mean())
    rank = int(np.searchsorted(cal_sorted, witness_knn))
    print(f'[knn] witness rank: {rank}/{args.n_cal} '
          f'(empirical CDF = {frac_below:.4f})', flush=True)
    print()

    if frac_below >= 0.99:
        print('[knn] ✅ WITNESS IS A kNN-DISTANCE OUTLIER.')
        print('[knn]    kNN-based conformal threshold at 99% would EXCLUDE this')
        print('[knn]    witness. kNN-distance scoring would plausibly fix this case.')
    elif frac_below >= 0.5:
        print(f'[knn] ~ WITNESS IS IN THE TAIL (rank {frac_below:.1%}).')
        print('[knn]   A tighter threshold (e.g., 99.5%) might cut it.')
    else:
        print(f'[knn] ❌ WITNESS IS NEAR THE TRAINING MANIFOLD '
              f'(rank {frac_below:.1%}).')
        print('[knn]    kNN-distance scoring would NOT exclude it. Either the')
        print('[knn]    spurious witness is genuinely close to training data, or')
        print('[knn]    the kNN metric does not separate spurious from in-set.')


if __name__ == '__main__':
    main()
