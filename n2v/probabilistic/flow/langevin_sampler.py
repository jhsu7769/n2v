"""Latent-space Langevin / MALA sampling for rare-event flow-set detection.

C3 (derived novelty): given a calibrated FlowODE and an unsafe polyhedron
``U = { y : G y <= g }``, sample from a *tilted* latent target

    log pi_target(z) = log p_z(z) - lambda * max(0, slack(f(z))) + const
                     = -||z||^2 / 2 - lambda * max(0, slack(f(z))) + const

with ``f := flow_ode.inverse`` and ``slack(y) := max_i (G_i y - g_i)``.
By the VNN-LIB convention, ``slack(y) <= 0`` iff ``y in U``.

The key algorithmic difference vs. AMLS (C1 -- random-walk MH at each
level) and IS (C2 -- pure reweighting): Langevin uses the FLOW's
gradient ``J_f(z)^T G_{i*}^T`` (computed via PyTorch autograd through
the ODE solver, since torchdiffeq is autograd-aware). This drifts each
chain DETERMINISTICALLY toward the unsafe set rather than relying on
chance Gaussian draws to land there.

Algorithm:
    1. Initialise K chains at z_0 ~ N(0, I_d).
    2. For each step:
       - Forward pass: y = f(z) (through the ODE).
       - Compute slack(y).
       - If slack(y) > 0: backprop slack through f to get
         ``grad_z slack(f(z)) = J_f(z)^T G_{i*}^T``.
       - Form ``g(z) = -z + lambda * 1[slack > 0] * (-J_f^T G_{i*}^T)``
         = ``grad_z log pi_target(z)``.
       - Update: ``z' = z + (h/2) g(z) + sqrt(h) xi``, ``xi ~ N(0, I)``.
    3. (Optional MALA accept/reject for asymptotic-unbiased sampling.)
    4. Detect: any ``slack(y_i) <= 0`` over all collected samples?

References:
    Welling-Teh ICML 2011 -- SGLD.
    Roberts-Tweedie 1996 -- MALA.
    Neal 2011 -- HMC chapter (escalation path).
    Tit-Furon-Rousset AISTATS 2023 -- gradient-AMLS for NN robustness.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class LangevinResult:
    """Result of a single-HalfSpace latent Langevin run.

    Attributes:
        pi_hat: Detection-mode point estimate of ``P_flow(y in U)``;
            here the empirical fraction of post-warmup samples (across
            all chains) with ``slack <= 0``. Use as a diagnostic; the
            biased target distribution makes this an UPPER bias of the
            true mass (it concentrates on U). Operationally important
            output is ``detected_unsafe``, NOT ``pi_hat``.
        n_samples: Total post-warmup samples collected (n_chains * n_samples).
        n_in_U: Number of post-warmup samples with ``slack <= 0``.
        detected_unsafe: True iff ANY chain's trajectory (warmup OR
            samples) reached a state with ``slack(f(z)) <= 0``.
        final_phi: MIN ``slack(f(z))`` observed across all states (warmup
            and samples). Non-positive iff ``detected_unsafe``.
        worst_y: ``(d,)`` array -- the y at which the min-slack was
            achieved (the deepest-into-U witness if detected).
        n_chains: K, number of parallel chains.
        n_warmup: Burn-in steps per chain.
        n_steps: Sampling steps per chain (after warmup).
        accept_rate: Mean MALA acceptance rate across chains and steps;
            None when ``use_mala=False``.
        mean_grad_norm: Mean ``||grad_z log pi_target(z)||`` across all
            updates. A diagnostic; very large values indicate the flow
            Jacobian is exploding and the step size should be reduced.
        lambda_tilt: Tilt parameter passed through.
        step_size: Langevin step size passed through.
    """
    pi_hat: float
    n_samples: int
    n_in_U: int
    detected_unsafe: bool
    final_phi: float
    worst_y: np.ndarray
    n_chains: int
    n_warmup: int
    n_steps: int
    accept_rate: Optional[float]
    mean_grad_norm: float
    lambda_tilt: float
    step_size: float


def _slack_torch(y: torch.Tensor, G: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Differentiable slack: max_i (G_i y - g_i) per row.

    Args:
        y: (N, d) outputs.
        G: (k, d) constraint matrix.
        g: (k,) RHS.

    Returns:
        (N,) tensor of max-row slacks. Non-positive iff in U.
    """
    margins = y @ G.T - g[None, :]  # (N, k)
    return margins.max(dim=1).values  # (N,)


def langevin_sample_toward_unsafe(
    flow_ode,
    halfspace,
    *,
    n_chains: int = 100,
    n_warmup: int = 200,
    n_samples: int = 500,
    step_size: float = 0.05,
    lambda_tilt: float = 5.0,
    use_mala: bool = False,
    grad_clip: float = 1e3,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: str = 'cpu',
) -> LangevinResult:
    """Run latent-space Langevin / MALA toward ``U = { y : G y <= g }``.

    The Langevin step uses ``grad_z log pi_target(z)`` with
    ``log pi_target(z) = -||z||^2/2 - lambda * max(0, slack(f(z)))`` and
    ``f = flow_ode.inverse``. Gradient is via PyTorch autograd through
    the ODE solver (torchdiffeq is autograd-aware).

    Args:
        flow_ode: trained ``FlowODE`` instance.
        halfspace: object with ``.G`` (k, d) and ``.g`` (k, 1) or (k,).
        n_chains: K parallel chains. Default 100.
        n_warmup: burn-in steps. Default 200.
        n_samples: sampling steps after warmup (per chain). Default 500.
        step_size: Langevin step ``h``. Default 0.05.
        lambda_tilt: tilt strength. Default 5.0.
        use_mala: if True, apply MALA accept/reject for exact stationarity.
            For DETECTION purposes (the gate criterion), False is
            sufficient and ~2x faster.
        grad_clip: clip ``||grad||`` to this magnitude per chain step
            (numerical safeguard against flow-Jacobian explosion).
        seed: RNG seed.
        t, n_ode_steps, ode_method, ode_atol, ode_rtol: passed to
            ``flow_ode.inverse``.
        device: torch device for chain tensors.

    Returns:
        ``LangevinResult``. Detection flag is the operative output.
    """
    if n_chains <= 0:
        raise ValueError(f'n_chains must be positive, got {n_chains}')
    if n_warmup < 0:
        raise ValueError(f'n_warmup must be >= 0, got {n_warmup}')
    if n_samples < 0:
        raise ValueError(f'n_samples must be >= 0, got {n_samples}')
    if step_size <= 0:
        raise ValueError(f'step_size must be positive, got {step_size}')
    if lambda_tilt < 0:
        raise ValueError(f'lambda_tilt must be >= 0, got {lambda_tilt}')

    G_np = np.asarray(halfspace.G, dtype=np.float64)
    g_np = np.asarray(halfspace.g, dtype=np.float64).flatten()
    dim = G_np.shape[1]
    K = n_chains

    if seed is not None:
        torch.manual_seed(int(seed) & 0x7FFFFFFF)
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    dev = torch.device(device)
    G_t = torch.from_numpy(G_np).float().to(dev)
    g_t = torch.from_numpy(g_np).float().to(dev)

    # Initialise K chains at z ~ N(0, I_d).
    z = torch.from_numpy(rng.standard_normal((K, dim)).astype(np.float32)).to(dev)

    h = float(step_size)
    sqrt_h = float(np.sqrt(h))

    # Track best (smallest) slack ever seen and its y.
    best_phi = float('inf')
    best_y = np.zeros(dim, dtype=np.float64)

    detected = False
    n_in_U_accum = 0
    n_samples_accum = 0
    accept_count = 0
    accept_total = 0
    grad_norm_sum = 0.0
    grad_norm_count = 0

    def _grad_log_pi(z_in: torch.Tensor):
        """Return (grad, slack, y) where grad = grad_z log pi_target(z).

        - grad: (K, d) tensor.
        - slack: (K,) tensor of max-row slacks.
        - y: (K, d) tensor of f(z) outputs.
        """
        z_req = z_in.detach().clone().requires_grad_(True)
        y = flow_ode.inverse(
            z_req, t=t, n_steps=n_ode_steps, method=ode_method,
            atol=ode_atol, rtol=ode_rtol,
        )
        slack = _slack_torch(y, G_t, g_t)
        # Tilt term: -lambda * max(0, slack) summed (per-row independence
        # implied by chain rule: each chain's grad depends only on its
        # own slack).
        # We want grad_z [ -||z||^2/2 - lambda * max(0, slack(f(z))) ].
        # Compute the sum so backward gives a per-chain gradient via
        # broadcasting:
        relu_slack = torch.clamp(slack, min=0.0)
        loss = (-0.5 * (z_req ** 2).sum() - lambda_tilt * relu_slack.sum())
        grad = torch.autograd.grad(loss, z_req, create_graph=False)[0]
        # Optional gradient norm clipping (per chain).
        if grad_clip is not None and grad_clip > 0:
            gnorm = grad.norm(dim=1, keepdim=True)
            scale = torch.clamp(grad_clip / (gnorm + 1e-12), max=1.0)
            grad = grad * scale
        return grad.detach(), slack.detach(), y.detach()

    def _log_pi(z_in: torch.Tensor, slack: torch.Tensor) -> torch.Tensor:
        """log pi_target(z) up to additive const. (K,)."""
        return (
            -0.5 * (z_in ** 2).sum(dim=1)
            - lambda_tilt * torch.clamp(slack, min=0.0)
        )

    def _record(slack_t: torch.Tensor, y_t: torch.Tensor):
        """Update the running best-slack witness from a (K,) slack/y batch."""
        nonlocal best_phi, best_y, detected
        slack_np = slack_t.cpu().numpy().astype(np.float64)
        y_np = y_t.cpu().numpy().astype(np.float64)
        idx = int(np.argmin(slack_np))
        if slack_np[idx] < best_phi:
            best_phi = float(slack_np[idx])
            best_y = y_np[idx].copy()
        if (slack_np <= 0.0).any():
            detected = True

    total_steps = n_warmup + n_samples
    for step_idx in range(total_steps):
        grad, slack, y = _grad_log_pi(z)
        _record(slack, y)

        # Diagnostic: mean grad norm.
        gn = float(grad.norm(dim=1).mean().item())
        grad_norm_sum += gn
        grad_norm_count += 1

        # Langevin proposal.
        xi = torch.randn_like(z)
        z_prop = z + 0.5 * h * grad + sqrt_h * xi

        if use_mala:
            # Compute gradient at proposal for MALA.
            grad_prop, slack_prop, y_prop = _grad_log_pi(z_prop)
            _record(slack_prop, y_prop)

            # log q(z'|z) and log q(z|z') under Gaussian proposals.
            # log q(z'|z) = -||z' - z - (h/2) grad(z)||^2 / (2h)
            mean_fwd = z + 0.5 * h * grad
            mean_bwd = z_prop + 0.5 * h * grad_prop
            log_q_fwd = -((z_prop - mean_fwd) ** 2).sum(dim=1) / (2.0 * h)
            log_q_bwd = -((z - mean_bwd) ** 2).sum(dim=1) / (2.0 * h)

            log_pi_z = _log_pi(z, slack)
            log_pi_zp = _log_pi(z_prop, slack_prop)

            log_alpha = log_pi_zp - log_pi_z + log_q_bwd - log_q_fwd
            u = torch.rand(K, device=dev)
            accept = (torch.log(u + 1e-30) < log_alpha).float().unsqueeze(1)
            accept_count += int(accept.sum().item())
            accept_total += K
            z = accept * z_prop + (1.0 - accept) * z
        else:
            z = z_prop

        # If we are past warmup, count this iterate's samples toward
        # the empirical mass.
        if step_idx >= n_warmup:
            # Re-evaluate at the (possibly accepted/rejected) current z.
            # For pure Langevin we already have slack at the pre-step z;
            # at this point z has been updated, so the slack at this iterate
            # would require another forward pass. To keep cost down, we
            # use the slack at the PREVIOUS state (pre-update), which is a
            # standard approximation in SGLD / MALA chain accounting (the
            # post-update slack will be recorded at the start of the next
            # iteration via _grad_log_pi's _record).
            slack_np = slack.cpu().numpy().astype(np.float64)
            n_in_U_accum += int((slack_np <= 0.0).sum())
            n_samples_accum += K

    # Final pass to record the post-loop state's slack for completeness
    # (and to ensure the witness reflects the very last iterate too).
    with torch.no_grad():
        y_final = flow_ode.inverse(
            z, t=t, n_steps=n_ode_steps, method=ode_method,
            atol=ode_atol, rtol=ode_rtol,
        )
        slack_final = _slack_torch(y_final, G_t, g_t)
    _record(slack_final, y_final)

    pi_hat = (
        n_in_U_accum / n_samples_accum
        if n_samples_accum > 0 else 0.0
    )
    accept_rate = (
        accept_count / accept_total if (use_mala and accept_total > 0) else None
    )
    mean_grad_norm = (
        grad_norm_sum / grad_norm_count if grad_norm_count > 0 else 0.0
    )

    return LangevinResult(
        pi_hat=float(pi_hat),
        n_samples=int(n_samples_accum),
        n_in_U=int(n_in_U_accum),
        detected_unsafe=bool(detected),
        final_phi=float(best_phi),
        worst_y=best_y,
        n_chains=K,
        n_warmup=n_warmup,
        n_steps=n_samples,
        accept_rate=accept_rate,
        mean_grad_norm=float(mean_grad_norm),
        lambda_tilt=float(lambda_tilt),
        step_size=float(step_size),
    )


@dataclass
class LangevinSpecResult:
    """Aggregate Langevin result for an AND-of-OR-of-AND spec.

    Mirrors AMLSSpecResult / ISSpecResult. The spec is UNSAT-disjoint iff
    every group is disjoint, and a group is disjoint iff every member
    HalfSpace is disjoint (no Langevin chain detected). Detection in any
    HalfSpace flips the verdict to UNKNOWN.
    """
    unsat_certified: bool
    detected_any: bool
    per_hs_results: list  # list[list[LangevinResult]]
    spec_groups: list


def langevin_certify_spec(
    flow_ode,
    spec_groups,
    *,
    n_chains: int = 100,
    n_warmup: int = 200,
    n_samples: int = 500,
    step_size: float = 0.05,
    lambda_tilt: float = 5.0,
    use_mala: bool = False,
    grad_clip: float = 1e3,
    seed: Optional[int] = 0,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    device: str = 'cpu',
) -> LangevinSpecResult:
    """Run latent Langevin per HalfSpace across an AND-of-OR-of-AND spec.

    Mirrors :func:`amls_certify_spec` / :func:`is_tilted_certify_spec`:
    outer AND across groups, inner OR within each group. A group is
    disjoint iff every HalfSpace in it is disjoint (no detection); the
    spec is UNSAT iff at least one group is fully disjoint.
    """
    per_hs_results: list = []
    detected_any = False
    h_idx = 0
    for group in spec_groups:
        group_results = []
        for hs in group:
            sub_seed = (
                None if seed is None
                else (int(seed) + h_idx * 7919) & 0x7FFFFFFF
            )
            r = langevin_sample_toward_unsafe(
                flow_ode=flow_ode, halfspace=hs,
                n_chains=n_chains, n_warmup=n_warmup, n_samples=n_samples,
                step_size=step_size, lambda_tilt=lambda_tilt,
                use_mala=use_mala, grad_clip=grad_clip,
                seed=sub_seed, t=t, n_ode_steps=n_ode_steps,
                ode_method=ode_method, ode_atol=ode_atol, ode_rtol=ode_rtol,
                device=device,
            )
            group_results.append(r)
            h_idx += 1
            if r.detected_unsafe:
                detected_any = True
        per_hs_results.append(group_results)

    unsat_certified = False
    for group_results in per_hs_results:
        if all(not r.detected_unsafe for r in group_results):
            unsat_certified = True
            break

    return LangevinSpecResult(
        unsat_certified=unsat_certified,
        detected_any=detected_any,
        per_hs_results=per_hs_results,
        spec_groups=spec_groups,
    )
