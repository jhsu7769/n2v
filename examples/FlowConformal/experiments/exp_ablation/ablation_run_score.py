"""Score-function ablation row: hyperrect, Mahalanobis ellipsoid,
GMM(10), flow.

Combined with Exp 3 dims per the 2026-04-27 paper-experiments plan:
the ablation now spans **4 networks x 4 score families x K seeds**:

    network in {3d_banana, 5d_1lip_id, 10d_1lip_id, 20d_1lip_id}
    score   in {hyperrect, ellipsoid, gmm, flow}

The 3D banana benchmark uses the cached Star-union ground truth
(``ThreeBlobClassifier3D``); the 1-Lipschitz nets use the closed-form
identity-activation reference (``exact_volume_linear_net``). This gives
us a "score-function x output-dim" scaling story for the paper.

Why 3D banana, not ACAS Xu, for the original score ablation:
    The ACAS Xu pipeline trains a flow + uses verification_method=amls,
    which intertwines the score function with the verification path
    (FlowScore is the only score AMLS knows how to use). The other
    score families (hyperrect, ellipsoid, GMM) only make sense as
    *volume-tightness probes* against a known exact reach set.

Output schema:

    network, dim, score, seed, threshold, volume, volume_ratio,
    empirical_coverage, fit_time_s

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.\
ablation_run_score --smoke

Wall-clock estimate (full sweep, K=5 seeds):
    Per-seed:   hyperrect ~5s, ellipsoid ~5s, gmm ~10s, flow ~3-5 min.
    4 nets x 4 scores x 5 seeds ~= 75 min total.
"""
from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import (
    _train_flow, exact_star_union_volume,
)
from examples.FlowConformal.networks import ThreeBlobClassifier3D
from examples.FlowConformal.experiments.exp3_synthetic.networks import (
    OneLipschitzNet,
)
from examples.FlowConformal.experiments.exp3_synthetic.exact_volumes import (
    exact_volume_linear_net,
)
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.sampling import sample_box, sample_l_inf_ball
from n2v.probabilistic.flow.scores import (
    BallScore, EllipsoidScore, FlowScore, GMMScore, HyperrectScore,
)
from n2v.probabilistic.flow.sets import ProbabilisticSet


_OUT_DIR = Path(__file__).parent / 'outputs'

_ALPHA = 0.001

# --- Network specs: (name, dim, builder, sampler, exact-volume-fn) ---


def _banana_sampler(net, n: int, seed: int) -> torch.Tensor:
    x_center_t = torch.zeros(3, dtype=torch.float32)
    x = sample_l_inf_ball(x_center=x_center_t, radius=1.0,
                          n_samples=n, seed=seed, dim=3)
    with torch.no_grad():
        return net(x)


def _onelip_sampler(net, n: int, seed: int) -> torch.Tensor:
    dim = net.dim
    lb_t = -0.5 * torch.ones(dim, dtype=torch.float32)
    ub_t = +0.5 * torch.ones(dim, dtype=torch.float32)
    x = sample_box(lb_t, ub_t, n_samples=n, seed=seed)
    with torch.no_grad():
        return net(x)


def _exact_volume_3d_banana(*, smoke: bool) -> float:
    torch.manual_seed(0)
    gt = ThreeBlobClassifier3D().eval()
    if smoke:
        v, _ = exact_star_union_volume(gt, x_center=np.zeros(3), radius=1.0,
                                       output_dim=3, n_mc=20_000)
    else:
        v, _ = exact_star_union_volume(gt, x_center=np.zeros(3), radius=1.0,
                                       output_dim=3)
    return v


def _exact_volume_onelip(dim: int) -> float:
    net = OneLipschitzNet(dim=dim, n_layers=4, activation='identity',
                          seed=0).eval()
    lb = -0.5 * np.ones(dim)
    ub = +0.5 * np.ones(dim)
    return exact_volume_linear_net(net, lb, ub, alpha=_ALPHA)


def _build_banana(seed: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    return ThreeBlobClassifier3D().eval()


def _build_onelip(dim: int):
    def _b(seed: int) -> torch.nn.Module:
        # Net is fixed at seed=0 across calibration seeds (mirrors
        # exp3_run_synthetic's _NET_SEED policy).
        return OneLipschitzNet(dim=dim, n_layers=4, activation='identity',
                               seed=0).eval()
    return _b


_NETWORK_SPECS = {
    '3d_banana': dict(
        dim=3, builder=_build_banana,
        sampler=_banana_sampler,
        exact_vol_smoke=lambda: _exact_volume_3d_banana(smoke=True),
        exact_vol_full=lambda: _exact_volume_3d_banana(smoke=False),
    ),
    '5d_1lip_id': dict(
        dim=5, builder=_build_onelip(5),
        sampler=_onelip_sampler,
        exact_vol_smoke=lambda: _exact_volume_onelip(5),
        exact_vol_full=lambda: _exact_volume_onelip(5),
    ),
    '10d_1lip_id': dict(
        dim=10, builder=_build_onelip(10),
        sampler=_onelip_sampler,
        exact_vol_smoke=lambda: _exact_volume_onelip(10),
        exact_vol_full=lambda: _exact_volume_onelip(10),
    ),
    '20d_1lip_id': dict(
        dim=20, builder=_build_onelip(20),
        sampler=_onelip_sampler,
        exact_vol_smoke=lambda: _exact_volume_onelip(20),
        exact_vol_full=lambda: _exact_volume_onelip(20),
    ),
}


def _build_ellipsoid_score(y_calib: torch.Tensor) -> EllipsoidScore:
    mu = y_calib.mean(dim=0)
    diff = (y_calib - mu).cpu().numpy().astype(np.float64)
    cov = (diff.T @ diff) / max(1, diff.shape[0] - 1)
    cov += 1e-6 * np.eye(cov.shape[0])
    cov_inv = np.linalg.inv(cov)
    return EllipsoidScore(
        center=mu, cov_inv=torch.as_tensor(cov_inv, dtype=torch.float32),
    )


def run_one_seed_one_network(network_name: str, seed: int, *,
                              smoke: bool, scores: tuple[str, ...]
                              ) -> list[dict]:
    spec = _NETWORK_SPECS[network_name]
    dim = spec['dim']
    net = spec['builder'](seed)

    n_train = 1_000 if smoke else 5_000
    n_calib = 1_000 if smoke else 2_000
    n_test = 500 if smoke else 2_000
    flow_epochs = 200 if smoke else 2_000
    n_mc_volume = 20_000 if smoke else 200_000

    sampler = spec['sampler']
    y_tr = sampler(net, n_train, seed)
    y_ca = sampler(net, n_calib, seed + 1_000_000)
    y_te = sampler(net, n_test, seed + 2_000_000)

    ell = int(math.ceil((n_calib + 1) * (1 - _ALPHA)))

    y_all = torch.cat([y_tr, y_ca, y_te], dim=0)
    lo = y_all.min(dim=0).values
    hi = y_all.max(dim=0).values
    pad = 0.05 * (hi - lo).clamp(min=1e-6)
    bbox = (lo - pad, hi + pad)

    rows = []

    def _record(name: str, score_fn, t0):
        thresh = calibrate(score_fn(y_ca), ell).item()
        s = ProbabilisticSet(score_fn=score_fn, threshold=thresh,
                             m=n_calib, ell=ell, epsilon=_ALPHA, dim=dim)
        vol, _se = s.estimate_volume(n_samples=n_mc_volume, bounding_box=bbox)
        cov = s.contains(y_te).float().mean().item()
        rows.append(dict(network=network_name, dim=dim, score=name, seed=seed,
                         threshold=thresh, volume=vol, empirical_coverage=cov,
                         fit_time_s=time.time() - t0))

    if 'hyperrect' in scores:
        t0 = time.time()
        score_fn = HyperrectScore(center=y_ca.mean(dim=0),
                                  scales=y_ca.std(dim=0).clamp(min=1e-8))
        _record('hyperrect', score_fn, t0)

    if 'ball' in scores:
        t0 = time.time()
        score_fn = BallScore(center=y_ca.mean(dim=0))
        _record('ball', score_fn, t0)

    if 'ellipsoid' in scores:
        t0 = time.time()
        score_fn = _build_ellipsoid_score(y_ca)
        _record('ellipsoid', score_fn, t0)

    if 'gmm' in scores:
        t0 = time.time()
        score_fn = GMMScore.fit(y_ca, n_components=10, random_state=seed)
        _record('gmm', score_fn, t0)

    if 'flow' in scores:
        t0 = time.time()
        flow = _train_flow(y_tr, dim, flow_epochs, seed)
        score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4',
                             batch_size=65536, atol=1e-5, rtol=1e-5)
        _record('flow', score_fn, t0)

    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='1 network (3d_banana), 1 seed; ~30-60s')
    p.add_argument('--seeds', type=int, default=5)
    p.add_argument('--networks', nargs='+',
                   choices=tuple(_NETWORK_SPECS.keys()),
                   default=tuple(_NETWORK_SPECS.keys()))
    p.add_argument('--scores', nargs='+',
                   choices=('hyperrect', 'ball', 'ellipsoid', 'gmm', 'flow'),
                   default=('hyperrect', 'ellipsoid', 'gmm', 'flow'))
    args = p.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        nets = ['3d_banana']
        seeds = [0]
        scores = ('hyperrect', 'ellipsoid', 'gmm')  # flow ~3 min, skip
        out_path = _OUT_DIR / 'ablation_score_smoke.csv'
    else:
        nets = list(args.networks)
        seeds = list(range(args.seeds))
        scores = tuple(args.scores)
        out_path = _OUT_DIR / 'ablation_score.csv'

    # Pre-compute exact-volume reference per network.
    exact_by_net = {}
    for name in nets:
        if args.smoke:
            ev = _NETWORK_SPECS[name]['exact_vol_smoke']()
        else:
            ev = _NETWORK_SPECS[name]['exact_vol_full']()
        # 3D banana floor is (1-alpha)*Star-union; 1-Lip already returned
        # (1-alpha) * det * box.
        if name == '3d_banana':
            floor = (1.0 - _ALPHA) * ev
        else:
            floor = ev
        exact_by_net[name] = floor
        print(f'  network={name:<14} exact-vol floor = {floor:.6g}', flush=True)

    fields = ['network', 'dim', 'score', 'seed', 'threshold', 'volume',
              'volume_ratio', 'empirical_coverage', 'fit_time_s']
    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for net_name in nets:
            for seed in seeds:
                print(f'[network={net_name} seed={seed}] running...',
                      flush=True)
                rows = run_one_seed_one_network(
                    net_name, seed, smoke=args.smoke, scores=scores,
                )
                floor = exact_by_net[net_name]
                for r in rows:
                    ratio = (r['volume'] / floor) if (
                        floor > 0 and r['volume'] == r['volume']
                    ) else float('nan')
                    r['volume_ratio'] = ratio
                    w.writerow({k: r.get(k, '') for k in fields})
                    f.flush()
                    print(f'  {r["score"]:<10} thr={r["threshold"]:.4f} '
                          f'vol={r["volume"]:.4g} ratio={ratio:.3f} '
                          f'cov={r["empirical_coverage"]:.4f} '
                          f'fit={r["fit_time_s"]:.1f}s', flush=True)
    print(f'Wrote {out_path}', flush=True)


if __name__ == '__main__':
    main()
