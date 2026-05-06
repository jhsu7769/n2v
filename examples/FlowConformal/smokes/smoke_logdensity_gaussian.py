"""Smoke test: sign-error canary for LogDetFlowScore on a Gaussian target.

Trains a small flow on N(mean, cov), computes Spearman rank correlation
of both naive ||phi(y)||_2 and -log p_theta(y) against the analytical
-log p. Pass criteria (both must hold):

  1. rho_logdet > 0.6          — catches sign-flip / integration-direction
                                  bugs (a sign error drives rho toward 0
                                  or negative).
  2. abs(rho_logdet - rho_naive) < 0.25
                                  — the two scores should be close-ish on a
                                  unimodal target; a large gap indicates the
                                  divergence-correction term is wildly off.

What this smoke does NOT do: validate absolute correctness of the
log-density formula. That is covered by
`tests/unit/probabilistic/flow/test_logdet_scores.py::test_linear_flow_matches_analytical`
(hits the analytical score on a hand-built linear flow to 1e-3). This
smoke only catches regressions on a trained flow.
"""
import numpy as np
import torch
from scipy.stats import spearmanr

from n2v.probabilistic.flow.logdet_scores import LogDetFlowScore
from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.scores import FlowScore
from n2v.probabilistic.flow.train import train_flow


def main():
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    d = 2
    mean = torch.tensor([0.5, -0.3])
    cov = torch.tensor([[1.0, 0.3], [0.3, 0.5]])
    L = torch.linalg.cholesky(cov)
    n = 2000
    y_train = mean + torch.randn(n, d) @ L.T

    # standardize_outputs=False: on a pure-Gaussian target, whitening collapses
    # the transport to identity and both scores degenerate to ||y||_2. Without
    # standardization the flow must learn the full affine map.
    # Training budget chosen to stay near the 25k-step sweet spot: 800 ep x 8
    # batches = ~6400 steps. Enough for a decent 2D affine approximation
    # without taking 4 minutes.
    vf = VelocityField(dim=d, hidden=64, n_layers=3, activation='silu').to(device)
    vf, _ = train_flow(
        vf, y_train.to(device),
        n_epochs=800, batch_size=256, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=10,
        use_ema=True, standardize_outputs=False,
    )
    flow = FlowODE(vf.eval())

    # Evaluate on a scatter of test points.
    y_test = mean + torch.randn(500, d) @ L.T

    # Analytical -log p(y) for N(mean, cov).
    diff = y_test - mean
    inv_cov = torch.linalg.inv(cov)
    quad = (diff @ inv_cov * diff).sum(dim=1)
    neg_logp = 0.5 * quad  # drop constants; ranking only

    # FlowScore integrates through the ODE without detaching; wrap inference
    # in no_grad so the returned tensor is numpy-convertible. (Production
    # callers get this via ProbabilisticSet.estimate_volume.)
    with torch.no_grad():
        naive = FlowScore(flow, t=1.0, n_steps=30, method='rk4')(y_test).cpu().numpy()
        logdet = LogDetFlowScore(
            flow, t=1.0, n_steps=30, method='dopri5', atol=1e-4, rtol=1e-4,
        )(y_test).cpu().numpy()

    rho_naive, _ = spearmanr(neg_logp.numpy(), naive)
    rho_logdet, _ = spearmanr(neg_logp.numpy(), logdet)

    print(f'Spearman correlation vs analytical -log p(y):')
    print(f'  naive FlowScore       rho = {rho_naive:.4f}')
    print(f'  LogDetFlowScore       rho = {rho_logdet:.4f}')

    # Canary 1: sign error would drive rho near zero or negative.
    assert rho_logdet > 0.6, (
        f'LogDetFlowScore rank correlation too low: {rho_logdet:.4f} '
        f'(sign error in divergence integral? integration direction?)'
    )
    # Canary 2: logdet should be in the same ballpark as naive on a unimodal
    # target. A big gap means the divergence correction is wildly off.
    gap = abs(rho_logdet - rho_naive)
    assert gap < 0.25, (
        f'|rho_logdet - rho_naive| = {gap:.4f} > 0.25 — divergence term '
        f'may be scaled incorrectly (naive={rho_naive:.4f}, '
        f'logdet={rho_logdet:.4f}).'
    )
    print('OK')


if __name__ == '__main__':
    main()
