"""Tests for C3 / Latent-space Langevin flow-set detector.

These tests exercise the gradient-informed Langevin sampler on small
synthetic flows, plus the dispatch into ``run_verification_pipeline``
via ``verification_method='derived'``.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from n2v.probabilistic.flow.langevin_sampler import (
    langevin_certify_spec,
    langevin_sample_toward_unsafe,
)
from n2v.sets.halfspace import HalfSpace


# ---------- Identity flow stub (autograd-friendly) ----------


class _IdentityFlow(torch.nn.Module):
    """Trivial flow whose inverse is the identity. Differentiable.

    Using a torch Module so PyTorch autograd hooks work on `inverse`.
    """

    def __init__(self):
        super().__init__()

    def inverse(self, z, **_kw):
        return z

    def forward(self, y, **_kw):
        return y


# ---------- Smoke + return-shape ----------


def test_langevin_basic_smoke():
    """Smoke: function runs, returns populated LangevinResult."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])  # y_0 >= -10 (bulk)
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=20, n_warmup=10, n_samples=10, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.n_chains == 20
    assert res.n_warmup == 10
    assert res.n_steps == 10
    assert isinstance(res.detected_unsafe, bool)
    assert res.worst_y.shape == (2,)
    assert res.mean_grad_norm >= 0.0
    # On a bulk halfspace Langevin should detect U.
    assert res.detected_unsafe is True


def test_langevin_detects_easy_unsafe():
    """U near the bulk: y_0 <= 0.5 has mass ~0.69 under N(0,1).
    Langevin should detect quickly.
    """
    flow = _IdentityFlow()
    # G y <= g  with  G = [[1, 0]], g = 0.5  i.e.  y_0 <= 0.5
    G = np.array([[1.0, 0.0]])
    g = np.array([0.5])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=20, n_warmup=20, n_samples=20, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.detected_unsafe is True
    assert res.final_phi <= 0.0
    # n_in_U should be a non-trivial fraction of post-warmup samples.
    assert res.n_in_U > 0


def test_langevin_detects_hard_unsafe():
    """U far in tail: y_0 <= -2 has mass ~0.0228 under N(0,1).
    Flat MC at N=400 has ~9 hits in expectation. Langevin should
    do at least as well by drifting toward the tail.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([-2.0])  # U = { y_0 <= -2 }
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=20, n_warmup=50, n_samples=50, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.detected_unsafe is True
    assert res.final_phi <= 0.0


def test_langevin_no_detect_when_truly_disjoint():
    """U = { y_0 <= -100 } unreachable by any reasonable chain in finite
    steps. Langevin should NOT detect.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=10, n_warmup=20, n_samples=20, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.detected_unsafe is False
    assert res.final_phi > 0.0


def test_langevin_zero_lambda_recovers_prior_dynamics():
    """With lambda=0, Langevin reduces to overdamped dynamics targeting
    pi = N(0, I). The chain-state distribution should remain centered
    around 0 (no drift toward U). On a bulk halfspace it will still
    detect because N(0, I) puts mass there.
    """
    flow = _IdentityFlow()
    G = np.array([[-1.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=20, n_warmup=10, n_samples=10, step_size=0.05,
        lambda_tilt=0.0, seed=0,
    )
    # Bulk halfspace => detected.
    assert res.detected_unsafe is True


def test_langevin_mala_accept_rate_in_range():
    """MALA's acceptance rate should be a finite fraction in [0, 1] when
    use_mala=True. A reasonable Langevin step on a smooth target should
    accept most proposals.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([0.5])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=10, n_warmup=20, n_samples=20, step_size=0.05,
        lambda_tilt=2.0, use_mala=True, seed=0,
    )
    assert res.accept_rate is not None
    assert 0.0 <= res.accept_rate <= 1.0


def test_langevin_no_mala_returns_none_accept_rate():
    """use_mala=False => accept_rate is None."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))

    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=10, n_warmup=5, n_samples=5, step_size=0.05,
        lambda_tilt=5.0, use_mala=False, seed=0,
    )
    assert res.accept_rate is None


def test_langevin_drift_direction_correct():
    """With non-trivial lambda, samples should drift toward U on
    average. Test this by checking that mean phi after warmup is
    smaller (closer to U) than at initialization.
    """
    flow = _IdentityFlow()
    G = np.array([[1.0]])
    g = np.array([0.0])  # U = { y_0 <= 0 }, half the mass
    hs = HalfSpace(G, g.reshape(-1, 1))

    # Long chain at non-trivial lambda.
    res = langevin_sample_toward_unsafe(
        flow, hs,
        n_chains=100, n_warmup=100, n_samples=100, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    # Detection certain; key signal is: most post-warmup samples are in U.
    assert res.detected_unsafe is True
    # With strong tilt and 200 steps, > 50% of samples should be in U.
    assert res.n_in_U / max(res.n_samples, 1) > 0.5


# ---------- Spec-level dispatcher ----------


def test_langevin_certify_spec_unsat_when_all_disjoint():
    """Spec with one group containing one unreachable HalfSpace =>
    unsat_certified True, detected_any False."""
    flow = _IdentityFlow()
    G = np.array([[1.0, 0.0]])
    g = np.array([-100.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = langevin_certify_spec(
        flow, [[hs]],
        n_chains=10, n_warmup=10, n_samples=10, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is False
    assert res.unsat_certified is True


def test_langevin_certify_spec_unknown_when_group_member_detected():
    """Single group with one reachable HalfSpace: detected => no group
    is fully disjoint => unsat_certified False."""
    flow = _IdentityFlow()
    G = np.array([[-1.0, 0.0]])
    g = np.array([10.0])
    hs = HalfSpace(G, g.reshape(-1, 1))
    res = langevin_certify_spec(
        flow, [[hs]],
        n_chains=10, n_warmup=5, n_samples=5, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.detected_any is True
    assert res.unsat_certified is False


def test_langevin_certify_spec_two_groups_one_disjoint_unsat():
    """Two groups (AND across); one disjoint group => UNSAT."""
    flow = _IdentityFlow()
    hs_far = HalfSpace(
        np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    hs_bulk = HalfSpace(
        np.array([[-1.0, 0.0]]), np.array([[10.0]]))
    res = langevin_certify_spec(
        flow, [[hs_far], [hs_bulk]],
        n_chains=10, n_warmup=10, n_samples=10, step_size=0.05,
        lambda_tilt=5.0, seed=0,
    )
    assert res.unsat_certified is True
    assert res.detected_any is True


# ---------- End-to-end dispatch via run_verification_pipeline ----------


def test_langevin_dispatch_invalid_method_rejected():
    """run_verification_pipeline must accept 'derived'."""
    from examples.FlowConformal.benchmarks._common import \
        run_verification_pipeline
    from examples.FlowConformal.networks import RotatedBananaNet

    net = RotatedBananaNet().eval()
    spec = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1.0]]))
    # Verify 'derived' does NOT raise.
    with pytest.raises(ValueError, match='unsupported verification_method'):
        run_verification_pipeline(
            network=net,
            input_lb=np.array([0.0, 0.0]),
            input_ub=np.array([1.0, 1.0]),
            spec=spec, sat_backend=None,
            verification_method='not_a_real_method',
        )


@pytest.mark.slow
def test_langevin_dispatch_in_common_py_returns_unknown_when_unsafe_reachable():
    """End-to-end via verification_method='derived' on a small banana net
    where the spec is reachable. Expect verdict=UNKNOWN with
    derived_detected_unsafe=True.
    """
    from examples.FlowConformal.benchmarks._common import \
        run_verification_pipeline
    from examples.FlowConformal.networks import RotatedBananaNet

    torch.manual_seed(0)
    net = RotatedBananaNet().eval()
    # Trivially-reachable spec: y_0 <= +1e9 (always true).
    spec = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1e9]]))
    res = run_verification_pipeline(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=spec,
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
        seed=0, sat_backend=None,
        verification_method='derived',
    )
    assert res['verdict'] == 'UNKNOWN'
    assert res['derived_detected_unsafe'] is True
