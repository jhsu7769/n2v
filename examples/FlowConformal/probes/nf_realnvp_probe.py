"""Normalizing-flow (RealNVP) probe (read-only diagnostic).

Drops in a small RealNVP coupling-flow as the density model in place of
the production FlowMatching CNF, runs the same lsnc_relu (or any)
instance, and reports:

  * Whether the verdict changes
  * Whether the AMLS witness (if any) is in a low-density region of
    the NF (i.e., would be cut by density-based conformal calibration)
  * Wall-time comparison

Why RealNVP: it gives EXACT log-density in O(d) per sample (no ODE
integration needed for the Jacobian — it's a sum of log-scale
coefficients), so density-based conformal scoring is cheap and
unambiguous. Quick scratch implementation; not optimized.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.nf_realnvp_probe \\
        --benchmark lsnc_relu --instance-idx 0
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.probabilistic.flow.amls_bounded import (
    amls_bounded_certify_spec_union,
)
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import _forward, _whiten_halfspace
from n2v.utils.verify_specification import (
    _parse_property_groups, distribute_and_of_or_of_and,
)


# -------------------- RealNVP coupling flow --------------------

class CouplingLayer(nn.Module):
    """Affine coupling layer: split x in half via a binary mask, map
    masked-half identity, transform unmasked-half by an affine function
    of masked-half. Log-det is sum of log-scale coefficients."""

    def __init__(self, dim: int, hidden: int, mask: Tensor):
        super().__init__()
        self.register_buffer('mask', mask.float())
        in_dim = dim
        # Output 2*dim: first dim = scale logits (s), second dim = shift (t)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2 * dim),
        )
        # Init last layer to small values so the layer starts near identity
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.scale_clamp = 5.0  # clamp log-scale to avoid blowup

    def forward(self, x: Tensor):
        """Forward: x → y. Returns (y, log|det dy/dx|)."""
        x_masked = x * self.mask
        st = self.net(x_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale_clamp  # bounded log-scale
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        y = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y: Tensor):
        """Inverse: y → x. Returns (x, log|det dx/dy|)."""
        y_masked = y * self.mask
        st = self.net(y_masked)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s) * self.scale_clamp
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        x = y_masked + (1.0 - self.mask) * ((y - t) * torch.exp(-s))
        log_det = -s.sum(dim=-1)
        return x, log_det


class RealNVP(nn.Module):
    """Stack of coupling layers with alternating masks. Base distribution
    is N(0, I) on the latent space."""

    def __init__(self, dim: int, n_layers: int = 8, hidden: int = 128):
        super().__init__()
        self.dim = dim
        masks = []
        for i in range(n_layers):
            mask = torch.zeros(dim)
            mask[i % 2 :: 2] = 1.0  # alternating odd/even
            masks.append(mask)
        self.layers = nn.ModuleList(
            [CouplingLayer(dim, hidden, m) for m in masks]
        )

    def forward(self, x: Tensor):
        """x → y. Returns (y, log|det dy/dx|)."""
        log_det_total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, ld = layer(x)
            log_det_total = log_det_total + ld
        return x, log_det_total

    def inverse(self, y: Tensor):
        """y → x. Returns (x, log|det dx/dy|)."""
        log_det_total = torch.zeros(y.shape[0], device=y.device, dtype=y.dtype)
        for layer in reversed(self.layers):
            y, ld = layer.inverse(y)
            log_det_total = log_det_total + ld
        return y, log_det_total

    def log_prob(self, y: Tensor) -> Tensor:
        """log p(y) = log p_z(z) + log|det dz/dy| where z = inverse(y)."""
        z, log_det_inv = self.inverse(y)
        # base log-density of standard Gaussian
        log_p_z = -0.5 * (z * z).sum(dim=-1) - 0.5 * self.dim * np.log(2 * np.pi)
        return log_p_z + log_det_inv

    def neg_log_prob_score(self, y: Tensor) -> Tensor:
        """Nonconformity score: s(y) = -log p(y). Higher = more outlier."""
        return -self.log_prob(y)


def train_realnvp(model: RealNVP, y_data: Tensor, n_epochs: int,
                  lr: float = 1e-3, batch_size: int = 256, seed: int = 47):
    torch.manual_seed(seed)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    n = y_data.shape[0]
    losses = []
    for epoch in range(n_epochs):
        perm = torch.randperm(n)
        total = 0.0
        n_batch = 0
        for i in range(0, n, batch_size):
            batch = y_data[perm[i:i + batch_size]]
            optim.zero_grad()
            log_p = model.log_prob(batch)
            loss = -log_p.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total += float(loss.item())
            n_batch += 1
        losses.append(total / max(n_batch, 1))
        if epoch % max(1, n_epochs // 10) == 0:
            print(f'  epoch {epoch:>4}: nll={losses[-1]:.3f}')
    return losses


# -------------------- Probe driver --------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--n-train', type=int, default=None,
                   help='Override n_train (default: per-benchmark cfg).')
    p.add_argument('--nf-epochs', type=int, default=2000,
                   help='RealNVP training epochs (default 2000).')
    p.add_argument('--nf-layers', type=int, default=8)
    p.add_argument('--nf-hidden', type=int, default=128)
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[nf] {args.benchmark} idx={args.instance_idx}: {vnn_rel}')

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]
    n_train = args.n_train or cfg['n_train']

    # ---- Step 1: sample training data + whiten ----
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    # Pick the device the RealNVP + all downstream ops will live on.
    # Keep on the same device the network is on to avoid CPU↔GPU thrash
    # for the m calibration forward passes.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[nf] using device={device}')

    print(f'[nf] sampling {n_train} training inputs + forward through network')
    x_tr = _sample_box(lb_t, ub_t, n_samples=n_train, seed=args.seed)
    y_tr = _forward(network, x_tr).detach().to(device)
    y_mean = y_tr.mean(dim=0); y_std = y_tr.std(dim=0).clamp_min(1e-8)
    y_tr_w = (y_tr - y_mean) / y_std
    output_dim = y_tr_w.shape[1]
    print(f'[nf] output dim={output_dim}')

    # ---- Step 2: train RealNVP on whitened outputs ----
    print(f'[nf] training RealNVP: {args.nf_layers} layers, hidden={args.nf_hidden}, '
          f'epochs={args.nf_epochs}')
    model = RealNVP(output_dim, n_layers=args.nf_layers, hidden=args.nf_hidden).to(device)
    t0 = time.time()
    train_realnvp(model, y_tr_w, n_epochs=args.nf_epochs, seed=args.seed)
    train_wall = time.time() - t0
    print(f'[nf] NF training wall: {train_wall:.1f}s')
    model.eval()

    # ---- Step 3: calibrate on m fresh samples using density score ----
    cal_seed = args.seed + 1_000_000
    m = 8000
    x_ca = _sample_box(lb_t, ub_t, n_samples=m, seed=cal_seed)
    y_ca = _forward(network, x_ca).detach().to(device)
    y_ca_w = (y_ca - y_mean) / y_std
    with torch.no_grad():
        cal_scores = model.neg_log_prob_score(y_ca_w).detach().cpu().numpy()
    sorted_scores = np.sort(cal_scores)
    ell = m  # use largest score as q
    q_density = float(sorted_scores[ell - 1])
    print(f'[nf] calibrated density-score threshold q = {q_density:.4f} '
          f'(largest of {m} cal scores; ell=m)')

    # Also report a few percentiles so we know the score distribution
    for pct in [5, 50, 95, 99, 99.9, 100]:
        v = float(np.percentile(cal_scores, pct))
        print(f'[nf]   cal score pct {pct:>5}: {v:>10.4f}')

    # ---- Step 4: run AMLS but with the NF as the flow ----
    # WARNING: AMLS bounded was written for FlowMatching CNF (uses
    # ``flow_ode`` with ``velocity_field``). The RealNVP is a different
    # interface — ``flow.forward(z)`` for sampling. We only need the
    # *forward* direction (z -> y) for AMLS, so we wrap RealNVP in a
    # minimal shim that mimics FlowODE's sampling interface.
    class NFAsFlowShim:
        """Adapter so AMLS-bounded sees a flow-like object that maps
        z (latent Gaussian) → y (output) via NF.forward."""
        def __init__(self, nf_model):
            self.nf_model = nf_model
            # Mimic FlowODE.solve_ode signature minimally
        def to(self, device):
            self.nf_model = self.nf_model.to(device)
            return self
        def eval(self):
            self.nf_model.eval()
            return self
        def parameters(self):
            return self.nf_model.parameters()

    # The amls_bounded code uses _push_through_flow which calls
    # ``flow_ode.solve_ode(z, t_span, ...)`` etc. We need to monkey-patch
    # _push_through_flow to use our NF.forward instead. Cleanest: vendor
    # the AMLS code locally with a different push function.

    # Actually simpler: AMLS calls _push_through_flow internally, which
    # dispatches to whatever flow_ode interface. Let's check what it
    # expects (FlowODE-like with .velocity_field and ODE solving) - so
    # we can't trivially substitute. The cleanest path is to vendor a
    # mini-AMLS that uses NF.forward.
    print('[nf] running mini-AMLS with NF.forward as the push function...')

    # ---- Vendored mini-AMLS-bounded (single-OR-group, single chain) ----
    raw_groups = _parse_property_groups(spec)
    raw_groups = distribute_and_of_or_of_and(raw_groups)
    y_mean_np = y_mean.detach().cpu().numpy()
    y_std_np = y_std.detach().cpu().numpy()
    whitened_groups = [
        [_whiten_halfspace(hs, y_mean_np, y_std_np) for hs in group]
        for group in raw_groups
    ]
    # Use the first group only for this probe (lsnc_relu has one OR-group
    # after distribute_and_of_or_of_and).
    halfspaces = whitened_groups[0]

    import math
    from scipy.stats import norm

    N = cfg['scenario_n_samples']
    K_max = cfg['amls_max_levels']
    quantile = 0.1
    n_mcmc = 10
    step = 0.3
    beta = 0.001
    # Use the calibrated density threshold as the "in-set" gate by passing
    # only those z whose forward image has score <= q. Equivalent to
    # truncating to the high-density region.
    # For a fair side-by-side with the production framework, also run
    # with the latent-norm bounded ball ||z|| <= q_old (not used here —
    # we focus on density-only since that's the experimental question).

    G_list = [(torch.from_numpy(np.asarray(hs.G, dtype=np.float32)).to(device),
               torch.from_numpy(np.asarray(hs.g, dtype=np.float32).flatten()).to(device))
              for hs in halfspaces]

    def phi_union(y):
        per_hs = []
        for G, g in G_list:
            margins = (y @ G.T) - g  # (N, n_constraints)
            per_hs.append(margins.max(dim=1).values)  # (N,)
        return torch.stack(per_hs, dim=1).min(dim=1).values  # (N,)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    # torch.Generator on CUDA: use device-aware Generator for the AMLS RNG.
    gen = torch.Generator(device=device).manual_seed(args.seed)
    z = torch.randn(N, output_dim, generator=gen, device=device)
    with torch.no_grad():
        y, _ = model.forward(z)
    phi = phi_union(y)
    print(f'[nf] init phi_min={float(phi.min()):.4f}  '
          f'frac_in_U={float((phi <= 0).float().mean()):.4f}')

    # If init has detection, report and stop
    if (phi <= 0).any():
        n_in = int((phi <= 0).sum().item())
        # Check density of detected witnesses
        with torch.no_grad():
            scores = model.neg_log_prob_score(y).detach().cpu().numpy()
        det_mask = (phi <= 0).numpy()
        det_scores = scores[det_mask]
        in_set_frac = float((det_scores <= q_density).mean())
        print(f'[nf] DETECTED at level 0: {n_in}/{N} in U  '
              f'(of those: {in_set_frac:.1%} also have density-score <= q_density)')
        print(f'[nf]   det witness scores: min={det_scores.min():.4f}  '
              f'median={np.median(det_scores):.4f}  max={det_scores.max():.4f}')
        print(f'[nf]   q_density={q_density:.4f}')
        # If most detections are below q_density they are "real"; if most
        # are above, they are out-of-distribution → density score would
        # cut them.
        if in_set_frac < 0.5:
            print('[nf] ✅ majority of NF witnesses are LOW-DENSITY '
                  '(would be cut by density-score conformal). Strong')
            print('[nf]    evidence that density-based score helps.')
        else:
            print('[nf] ❌ majority of NF witnesses are HIGH-DENSITY '
                  '(would NOT be cut). Density score does not help here.')
        return

    # If no detection at level 0, run a quick AMLS chain on phi
    print('[nf] no init detection; running AMLS chain on phi(NF(z))...')
    tau_prev = math.inf
    K = 0
    for level in range(K_max):
        K = level + 1
        tau_unclamped = float(torch.quantile(phi, quantile).item())
        tau_k = tau_unclamped
        if tau_k >= tau_prev:
            tau_k = tau_prev - 1e-12
        if tau_k <= 0.0:
            print(f'[nf] level {K}: tau dropped to <= 0; chain reached U.')
            break
        # Resample bottom quantile
        keep_count = max(1, int(round(N * quantile)))
        _kept_phi, keep_idx = torch.topk(phi, k=keep_count, largest=False)
        sample_idx = torch.randint(0, keep_count, (N,), generator=gen, device=device)
        z = z[keep_idx[sample_idx]].clone()
        # Quick MCMC: random-walk on z, accept if new phi <= tau_k
        for _ in range(n_mcmc):
            z_prop = z + step * torch.randn(N, output_dim, generator=gen, device=device)
            with torch.no_grad():
                y_prop, _ = model.forward(z_prop)
            phi_prop = phi_union(y_prop)
            mh_pass = torch.rand(N, generator=gen, device=device) < torch.exp(
                0.5 * ((z * z).sum(dim=1) - (z_prop * z_prop).sum(dim=1)))
            level_pass = phi_prop <= tau_k
            accept = mh_pass & level_pass
            z = torch.where(accept.unsqueeze(-1), z_prop, z)
            with torch.no_grad():
                y, _ = model.forward(z)
            phi = phi_union(y)
        tau_prev = tau_k
        if (phi <= 0).any():
            print(f'[nf] level {K}: detected witness in U.')
            break
        if level % 5 == 0 or level == K_max - 1:
            print(f'[nf] level {K}: tau_k={tau_k:.4f}  phi_min={float(phi.min()):.4f}')

    final_det = bool((phi <= 0).any().item())
    pi_hat = (quantile ** K) if not final_det else (quantile ** K) * float((phi <= 0).float().mean())
    print(f'[nf] final: detected={final_det}  K={K}  pi_hat≈{pi_hat:.3e}')
    print(f'[nf] verdict: {"UNKNOWN" if final_det else "UNSAT"}')


if __name__ == '__main__':
    main()
