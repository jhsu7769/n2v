"""Exp 3 (synthetic validation): 5D / 10D / 20D 1-Lipschitz nets — our method only.

For each ``dim in {5, 10, 20}``:
  - construct an identity-activation 1-Lipschitz net (seed=0; same net
    across calibration seeds so the exact-volume reference is fixed),
  - run the AMLS Phase-5d pipeline for ``K`` calibration seeds,
  - MC-estimate the calibrated reach-set volume,
  - compute the closed-form exact volume (via ``exact_volume_linear_net``).

Output CSV columns:
    dim, seed, verdict, q, volume_estimate, volume_ratio_vs_exact,
    coverage_empirical, train_s, verify_s, total_s

Smoke usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_synthetic --smoke
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.experiments.exp3_synthetic.networks import (
    OneLipschitzNet,
)
from examples.FlowConformal.experiments.exp3_synthetic.exact_volumes import (
    exact_volume_linear_net,
)
from n2v.probabilistic.flow.sampling import sample_box
from n2v.probabilistic.flow.sets import ProbabilisticSet
from n2v.sets.halfspace import HalfSpace


_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'exp3_synthetic_ours.csv'

# >>> LOCKED Phase 5d config <<<
_VERIFICATION_METHOD = 'amls'
_ALPHA = 0.001
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_SCENARIO_BETA = 0.001

_INPUT_RADIUS = 0.5  # input box [-0.5, 0.5]^dim
_NET_SEED = 0  # fixed across calibration seeds


def _build_unsat_spec(dim: int) -> HalfSpace:
    """Halfspace ``y[0] >= 1e6`` — well outside the typical output range
    (1-Lipschitz nets on [-0.5, 0.5]^dim produce outputs of magnitude
    O(sqrt(dim) * 0.5)). Always UNSAT, so we test volume tightness
    rather than verdict correctness.
    """
    G = np.zeros((1, dim), dtype=np.float64)
    G[0, 0] = -1.0  # -y[0] <= -1e6  <=>  y[0] >= 1e6
    g = np.array([[-1e6]])
    return HalfSpace(G, g)


def _flow_volume(result: dict, net, lb_t: torch.Tensor, ub_t: torch.Tensor,
                 dim: int, *, n_mc: int = 200_000, seed: int = 0,
                 ) -> tuple[float, float]:
    """MC-estimate volume of {y : score_fn(y) <= q} over a bbox of
    training-output extents (5% padded)."""
    x = sample_box(lb_t, ub_t, n_samples=4_000, seed=seed + 12345)
    with torch.no_grad():
        y = net(x)
    y_lo = y.min(dim=0).values
    y_hi = y.max(dim=0).values
    pad = 0.05 * (y_hi - y_lo).clamp(min=1e-6)
    bbox = (y_lo - pad, y_hi + pad)
    s = ProbabilisticSet(
        score_fn=result['score_fn'], threshold=result['q'],
        m=8000, ell=7999, epsilon=_ALPHA, dim=dim,
    )
    vol, se = s.estimate_volume(n_samples=n_mc, bounding_box=bbox)
    return float(vol), float(se)


def run_one(dim: int, seed: int, *, smoke: bool = False) -> dict:
    """Train+calibrate+verify on the dim-D synthetic net."""
    torch.manual_seed(seed)
    net = OneLipschitzNet(dim=dim, n_layers=4, activation='identity',
                          seed=_NET_SEED).eval()
    spec = _build_unsat_spec(dim)
    lb = (-_INPUT_RADIUS * np.ones(dim)).astype(np.float64)
    ub = (_INPUT_RADIUS * np.ones(dim)).astype(np.float64)

    flow_epochs = 200 if smoke else _FLOW_EPOCHS
    n_train = 1_000 if smoke else _N_TRAIN
    scen_n = 200 if smoke else _SCENARIO_N

    t0 = time.time()
    result = run_verification_pipeline(
        network=net,
        input_lb=lb, input_ub=ub, spec=spec,
        alpha=_ALPHA,
        n_train=n_train, flow_epochs=flow_epochs,
        flow_config='base',
        scenario_n_samples=scen_n, scenario_beta=_SCENARIO_BETA,
        verification_method=_VERIFICATION_METHOD,
        seed=seed,
        use_falsifier=False,  # Exp 3: no falsifier per finalized plan
        sat_backend='none',
    )
    total_s = time.time() - t0

    if result['verdict'] in ('UNSAT', 'UNKNOWN') and result.get('score_fn') is not None:
        lb_t = torch.as_tensor(lb, dtype=torch.float32)
        ub_t = torch.as_tensor(ub, dtype=torch.float32)
        n_mc_vol = 50_000 if smoke else 200_000
        vol, _ = _flow_volume(result, net, lb_t, ub_t, dim,
                              n_mc=n_mc_vol, seed=seed)
    else:
        vol = float('nan')

    return {
        'dim': dim,
        'seed': seed,
        'verdict': result['verdict'],
        'q': result.get('q'),
        'volume_estimate': vol,
        'coverage_empirical': result.get('coverage_empirical'),
        'train_s': result.get('flow_train_time_s'),
        'verify_s': result.get('verification_time_s'),
        'total_s': total_s,
        'amls_levels_used': result.get('amls_levels_used', ''),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='1 dim (5D), 1 seed, reduced epochs; ~30-60s')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--dims', type=str, default='5,10,20',
                        help='comma-separated list of dims')
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        dims = [5]
        seeds = [0]
    else:
        dims = [int(d) for d in args.dims.split(',')]
        seeds = list(range(args.seeds))

    # Pre-compute exact volumes once per dim (net is fixed at _NET_SEED).
    exact_vol_by_dim = {}
    for dim in dims:
        ref_net = OneLipschitzNet(
            dim=dim, n_layers=4, activation='identity', seed=_NET_SEED,
        ).eval()
        lb = -_INPUT_RADIUS * np.ones(dim)
        ub = _INPUT_RADIUS * np.ones(dim)
        ev = exact_volume_linear_net(ref_net, lb, ub, alpha=_ALPHA)
        exact_vol_by_dim[dim] = ev
        print(f'  dim={dim}  exact_vol(1-alpha) = {ev:.6g}', flush=True)

    out_path = _OUT_CSV if not args.smoke else _OUT_CSV.with_suffix('.smoke.csv')
    fields = ['dim', 'seed', 'verdict', 'q', 'volume_estimate',
              'volume_ratio_vs_exact', 'coverage_empirical',
              'train_s', 'verify_s', 'total_s', 'amls_levels_used']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for dim in dims:
            for seed in seeds:
                print(f'[dim={dim} seed={seed}] running...', flush=True)
                row = run_one(dim, seed, smoke=args.smoke)
                ev = exact_vol_by_dim[dim]
                ratio = (row['volume_estimate'] / ev) if ev > 0 else float('nan')
                row['volume_ratio_vs_exact'] = ratio
                print(f'  verdict={row["verdict"]} q={row["q"]} '
                      f'vol={row["volume_estimate"]:.6g} '
                      f'ratio={ratio:.3f} cov={row["coverage_empirical"]} '
                      f'total_s={row["total_s"]:.1f}', flush=True)
                writer.writerow({k: row.get(k, '') for k in fields})
                f.flush()
    print(f'\nWrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
