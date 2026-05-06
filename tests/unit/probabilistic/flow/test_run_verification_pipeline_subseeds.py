"""Tests for sub-seed instrumentation in run_verification_pipeline.

The pipeline should accept optional flow_seed/cal_seed/scenario_seed kwargs.
When unset, they default to the existing `seed` arg. When set, they control
their respective stage independently.
"""
import numpy as np
import pytest
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.networks import RotatedBananaNet
from n2v.sets.halfspace import HalfSpace


@pytest.fixture(scope='module')
def banana_net():
    """Module-scoped network. RotatedBananaNet.__init__ runs 2000 training
    steps consuming global RNG, so constructing it once and reusing avoids
    network-init drift between tests that vary only sub-seeds.
    """
    torch.manual_seed(0)
    return RotatedBananaNet().eval()


def _trivial_args(net):
    return dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),  # easy UNSAT
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
    )


@pytest.mark.slow
def test_default_subseeds_match_single_seed(banana_net):
    """Passing only seed=k should give same q as passing seed=k explicitly."""
    r1 = run_verification_pipeline(seed=42, **_trivial_args(banana_net))
    r2 = run_verification_pipeline(seed=42, flow_seed=42, cal_seed=42,
                                   scenario_seed=42, **_trivial_args(banana_net))
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-6)


@pytest.mark.slow
def test_independent_flow_seed_changes_q(banana_net):
    """Varying flow_seed alone should produce different q."""
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=42, flow_seed=1, cal_seed=42,
                                   scenario_seed=42, **common)
    r2 = run_verification_pipeline(seed=42, flow_seed=2, cal_seed=42,
                                   scenario_seed=42, **common)
    assert r1['q'] != r2['q'], 'flow_seed should affect q'


@pytest.mark.slow
def test_independent_cal_seed_changes_q(banana_net):
    """Varying cal_seed alone should produce different q AND must NOT
    perturb flow training (final loss must match)."""
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=42, flow_seed=42, cal_seed=1,
                                   scenario_seed=42, **common)
    r2 = run_verification_pipeline(seed=42, flow_seed=42, cal_seed=2,
                                   scenario_seed=42, **common)
    # Non-leakage: same flow_seed must produce identical flow training.
    assert r1['flow_train_loss_final'] is not None
    assert r1['flow_train_loss_final'] == pytest.approx(
        r2['flow_train_loss_final'], rel=1e-9), \
        'cal_seed must not affect flow training'
    # cal_seed must affect calibration -> q.
    assert r1['q'] != r2['q'], 'cal_seed should affect q'


@pytest.mark.slow
def test_scenario_seed_does_not_change_q(banana_net):
    """scenario_seed should affect scenario sampling but not q (which is
    determined by flow training and calibration, both upstream)."""
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=42, flow_seed=42, cal_seed=42,
                                   scenario_seed=1, **common)
    r2 = run_verification_pipeline(seed=42, flow_seed=42, cal_seed=42,
                                   scenario_seed=2, **common)
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-9), \
        'scenario_seed must not affect q'
