"""Exp 3 (synthetic validation): geometric transformation suite.

Stresses the "geometry-aware" claim of flow-matching reach sets: as the
reach set rotates / translates / nonlinearises away from axis-aligned,
hyperrect-style baselines blow up but a flow should hold tight.

Four networks (5D, identity input radius 0.5):

    - identity_axis_aligned    W = I (axis-aligned box reach set)
    - rotated                  W = random orthogonal (parallelepiped)
    - translated               y = W·x + b with W=I and a fixed offset
                               so the reach set is a translated box
    - nonlinear                1-Lipschitz net with tanh activation
                               (curved reach set)

Default 5 seeds × 4 networks. Spec is the trivially-UNSAT halfspace
``y[0] >= 1e6`` so verdicts are uniformly UNSAT and the headline metric
is volume tightness vs. ground-truth (closed-form for the linear nets,
MC for the nonlinear net).

The falsifier is OFF per Exp 3 plan.

Output CSV columns:
    network, seed, verdict, q, volume_estimate, volume_ratio_vs_exact,
    coverage_empirical, train_s, verify_s, total_s

Smoke usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp3_synthetic.exp3_run_geo_transforms \\
        --smoke
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

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
_OUT_CSV = _OUT_DIR / 'exp3_geo_transforms_ours.csv'

# >>> LOCKED Phase 5d config <<<
_VERIFICATION_METHOD = 'amls'
_ALPHA = 0.001
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_SCENARIO_BETA = 0.001

_DIM = 5
_INPUT_RADIUS = 0.5
_NET_SEED = 0


# ---------------------------------------------------------------------------
# Per-network constructors
# ---------------------------------------------------------------------------

class _IdentityNet(nn.Module):
    """y = x. Reach set is the axis-aligned input box."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.activation_name = 'identity'
        self.W_list = [torch.eye(dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def total_weight(self) -> torch.Tensor:
        return torch.eye(self.dim)


class _TranslatedIdentity(nn.Module):
    """y = x + b. Reach set is a translated axis-aligned box; the
    closed-form linear-net volume formula applies (det = 1)."""

    def __init__(self, dim: int, b: torch.Tensor):
        super().__init__()
        self.dim = dim
        self.activation_name = 'identity'
        self.b = nn.Parameter(b.clone(), requires_grad=False)
        self.W_list = [torch.eye(dim)]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.b

    def total_weight(self) -> torch.Tensor:
        return torch.eye(self.dim)


def _make_identity_net(dim: int) -> nn.Module:
    return _IdentityNet(dim).eval()


def _make_rotated_net(dim: int) -> nn.Module:
    """Random orthogonal 1-Lipschitz net (identity activation). The
    reach set is a rotated parallelepiped of unit determinant in
    ``W_total``. Uses a fixed seed so the network is shared across
    calibration seeds."""
    return OneLipschitzNet(dim=dim, n_layers=4, activation='identity',
                           seed=_NET_SEED).eval()


def _make_translated_net(dim: int) -> nn.Module:
    gen = torch.Generator().manual_seed(_NET_SEED)
    b = (torch.randn(dim, generator=gen) * 0.3).to(torch.float32)
    return _TranslatedIdentity(dim=dim, b=b).eval()


def _make_nonlinear_net(dim: int) -> nn.Module:
    """1-Lipschitz net with tanh activation — curved reach set; no
    closed-form volume formula but MC ground truth is well-posed."""
    return OneLipschitzNet(dim=dim, n_layers=4, activation='tanh',
                           seed=_NET_SEED).eval()


_NETWORK_BUILDERS = {
    'identity_axis_aligned': _make_identity_net,
    'rotated': _make_rotated_net,
    'translated': _make_translated_net,
    'nonlinear': _make_nonlinear_net,
}


# ---------------------------------------------------------------------------
# Ground-truth volume per network
# ---------------------------------------------------------------------------

def _exact_volume_for(name: str, net: nn.Module, lb: np.ndarray, ub: np.ndarray,
                      *, smoke: bool, seed: int = 0) -> float:
    """Return the (1-alpha)-conformal floor volume.

    For linear (identity-activation) nets we use the closed-form. For
    the nonlinear net (tanh) we estimate via MC: sample x uniformly in
    the input box, push through the net, and report the volume of the
    bounding box of outputs scaled by (1-alpha) — a *loose* but
    consistent surrogate (lemma: any reach-set volume estimate that
    upper-bounds the true reach set gives a sound floor).
    """
    activation = getattr(net, 'activation_name', 'identity')
    if activation == 'identity':
        return exact_volume_linear_net(net, lb, ub, alpha=_ALPHA)
    # MC bounding-box surrogate.
    n_mc = 5_000 if smoke else 50_000
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    x = sample_box(lb_t, ub_t, n_samples=n_mc, seed=seed + 99991)
    with torch.no_grad():
        y = net(x)
    y_lo = y.min(dim=0).values
    y_hi = y.max(dim=0).values
    bbox_vol = float(torch.prod(torch.clamp(y_hi - y_lo, min=1e-12)).item())
    return (1.0 - _ALPHA) * bbox_vol


# ---------------------------------------------------------------------------
# Spec + volume for one run
# ---------------------------------------------------------------------------

def _build_unsat_spec(dim: int) -> HalfSpace:
    """Trivially-UNSAT halfspace ``y[0] >= 1e6``."""
    G = np.zeros((1, dim), dtype=np.float64)
    G[0, 0] = -1.0
    g = np.array([[-1e6]])
    return HalfSpace(G, g)


def _flow_volume(result: dict, net, lb_t: torch.Tensor, ub_t: torch.Tensor,
                 dim: int, *, n_mc: int = 200_000, seed: int = 0,
                 ) -> tuple[float, float]:
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


def run_one(network_name: str, seed: int, *, smoke: bool = False) -> dict:
    torch.manual_seed(seed)
    builder = _NETWORK_BUILDERS[network_name]
    net = builder(_DIM)
    spec = _build_unsat_spec(_DIM)
    lb = (-_INPUT_RADIUS * np.ones(_DIM)).astype(np.float64)
    ub = (_INPUT_RADIUS * np.ones(_DIM)).astype(np.float64)

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
        vol, _ = _flow_volume(result, net, lb_t, ub_t, _DIM,
                              n_mc=n_mc_vol, seed=seed)
    else:
        vol = float('nan')

    return {
        'network': network_name,
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
                        help='1 network (rotated), 1 seed; ~30-60s')
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--networks', nargs='+',
                        choices=tuple(_NETWORK_BUILDERS.keys()),
                        default=tuple(_NETWORK_BUILDERS.keys()))
    args = parser.parse_args()

    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        nets = ['rotated']
        seeds = [0]
    else:
        nets = list(args.networks)
        seeds = list(range(args.seeds))

    # Pre-compute exact (or MC-floor) volumes per network.
    lb = -_INPUT_RADIUS * np.ones(_DIM)
    ub = _INPUT_RADIUS * np.ones(_DIM)
    exact_vol_by_net = {}
    for name in nets:
        ref = _NETWORK_BUILDERS[name](_DIM)
        ev = _exact_volume_for(name, ref, lb, ub, smoke=args.smoke)
        exact_vol_by_net[name] = ev
        print(f'  network={name:<22} exact/MC floor vol = {ev:.6g}', flush=True)

    out_path = _OUT_CSV if not args.smoke else _OUT_CSV.with_suffix('.smoke.csv')
    fields = ['network', 'seed', 'verdict', 'q', 'volume_estimate',
              'volume_ratio_vs_exact', 'coverage_empirical',
              'train_s', 'verify_s', 'total_s', 'amls_levels_used']
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for name in nets:
            for seed in seeds:
                print(f'[network={name} seed={seed}] running...', flush=True)
                row = run_one(name, seed, smoke=args.smoke)
                ev = exact_vol_by_net[name]
                v = row['volume_estimate']
                ratio = (v / ev) if (ev > 0 and v == v) else float('nan')
                row['volume_ratio_vs_exact'] = ratio
                vstr = f'{v:.6g}' if v == v else 'nan'
                print(f'  verdict={row["verdict"]} q={row["q"]} '
                      f'vol={vstr} '
                      f'ratio={ratio:.3f} cov={row["coverage_empirical"]} '
                      f'total_s={row["total_s"]:.1f}', flush=True)
                writer.writerow({k: row.get(k, '') for k in fields})
                f.flush()
    print(f'\nWrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
