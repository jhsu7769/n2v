"""Tests for C0 / scenario_v2: QMC+antithetic sampling and multi-restart
min-margin aggregation.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.scenario_verify import (
    _qmc_sample_latents,
    certify_halfspace_disjoint,
)
from n2v.sets.halfspace import HalfSpace


# -------- _qmc_sample_latents antithetic helper tests --------


def test_qmc_antithetic_shape():
    """antithetic=True returns the requested n_samples shape."""
    z = _qmc_sample_latents(64, 4, seed=0, antithetic=True)
    assert z.shape == (64, 4)


def test_qmc_antithetic_pairs_z_with_neg_z():
    """For even n_samples, the second half equals the negation of the first."""
    n = 32
    z = _qmc_sample_latents(n, 3, seed=0, antithetic=True).numpy()
    half = n // 2
    np.testing.assert_allclose(z[half:], -z[:half], rtol=0, atol=0)


def test_qmc_antithetic_uses_half_sobol_count():
    """The first half of antithetic output equals plain QMC at ceil(n/2) points."""
    n = 50
    z_anti = _qmc_sample_latents(n, 4, seed=123, antithetic=True).numpy()
    n_base = (n + 1) // 2  # ceil
    z_plain = _qmc_sample_latents(n_base, 4, seed=123, antithetic=False).numpy()
    np.testing.assert_allclose(z_anti[:n_base], z_plain, rtol=0, atol=0)


def test_qmc_antithetic_odd_n_samples():
    """Odd n_samples works: ceil(n/2) Sobol points, mirrored, trimmed to n."""
    n = 33
    z = _qmc_sample_latents(n, 2, seed=0, antithetic=True)
    assert z.shape == (n, 2)
    assert torch.isfinite(z).all()


# -------- certify_halfspace_disjoint accepts qmc+antithetic --------


def _train_small_2d_flow(seed: int = 0):
    from n2v.probabilistic.flow.model import VelocityField
    from n2v.probabilistic.flow.ode import FlowODE
    from n2v.probabilistic.flow.train import train_flow

    torch.manual_seed(seed)
    vf = VelocityField(dim=2, hidden=64, n_layers=2,
                       activation='silu', time_embed='concat')
    rng = np.random.default_rng(seed)
    y_train = torch.from_numpy(
        rng.standard_normal((2000, 2)).astype(np.float32)
    )
    vf, _ = train_flow(
        vf, y_train, n_epochs=200, batch_size=512, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=5,
        use_ema=True, standardize_outputs=False, time_sampling='uniform',
    )
    vf.eval()
    return FlowODE(vf)


@pytest.mark.slow
def test_certify_halfspace_disjoint_qmc_antithetic_accepted():
    """sampling_strategy='qmc+antithetic' is accepted and runs."""
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    res = certify_halfspace_disjoint(
        flow_ode=flow, threshold_q=3.0, halfspace=hs,
        n_samples=200, beta_2=0.001, seed=0,
        sampling_strategy='qmc+antithetic',
    )
    assert res.disjoint is True
    assert res.epsilon_2 == pytest.approx(math.log(1.0 / 0.001) / 200, rel=1e-6)


@pytest.mark.slow
def test_certify_halfspace_disjoint_invalid_strategy_message_mentions_qmc_antithetic():
    """Error for unknown strategy mentions 'qmc+antithetic'."""
    flow = _train_small_2d_flow(seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    with pytest.raises(ValueError, match='qmc\\+antithetic'):
        certify_halfspace_disjoint(
            flow_ode=flow, threshold_q=3.0, halfspace=hs,
            n_samples=100, beta_2=0.001, seed=0,
            sampling_strategy='not_a_real_strategy',
        )


# -------- scenario_v2 end-to-end tests --------


@pytest.mark.slow
def test_scenario_v2_runs_end_to_end_and_min_margin_aggregation():
    """verification_method='scenario_v2' runs end-to-end, and the
    aggregated verdict matches the min-worst-max-margin across the K
    restarts.

    Uses RotatedBananaNet (small, fast) with a trivial unreachable spec
    that should certify UNSAT under both 'scenario' and 'scenario_v2'.
    """
    from examples.FlowConformal.benchmarks._common import (
        _certify_spec_on_flow_v2,
        _extract_min_worst_max_margin,
        run_verification_pipeline,
    )
    from examples.FlowConformal.networks import RotatedBananaNet

    torch.manual_seed(0)
    net = RotatedBananaNet().eval()
    common = dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
    )
    res = run_verification_pipeline(
        seed=0, verification_method='scenario_v2', **common,
    )
    # Trivially-unreachable spec: should certify UNSAT.
    assert res['verdict'] == 'UNSAT'
    sr = res['scenario_result']
    assert 'aggregate_metadata' in sr
    meta = sr['aggregate_metadata']
    assert meta['method'] == 'scenario_v2'
    assert meta['n_restarts'] == 5
    assert len(meta['per_run_min_margin']) == 5
    # Aggregate min must equal the actual min over per-run mins.
    assert meta['aggregate_min_margin'] == min(meta['per_run_min_margin'])
    # Effective n_samples reported as K * N.
    assert sr['n_samples_used'] == 200 * 5


@pytest.mark.slow
def test_scenario_default_unchanged_after_v2_addition():
    """Default verification_method='scenario' (no kwarg) still produces
    bit-identical q to an explicit verification_method='scenario'.
    """
    from examples.FlowConformal.benchmarks._common import run_verification_pipeline
    from examples.FlowConformal.networks import RotatedBananaNet

    torch.manual_seed(0)
    net = RotatedBananaNet().eval()
    common = dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
    )
    r1 = run_verification_pipeline(seed=0, **common)
    r2 = run_verification_pipeline(
        seed=0, verification_method='scenario', **common,
    )
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-9)
    assert r1['verdict'] == r2['verdict']
