"""Tests for verification_method dispatch.

Each Phase A candidate plugs in via verification_method='<name>'. Default
'scenario' preserves current behavior bit-identically.
"""
import numpy as np
import pytest
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.networks import RotatedBananaNet
from n2v.sets.halfspace import HalfSpace


@pytest.fixture(scope='module')
def banana_net():
    torch.manual_seed(0)
    return RotatedBananaNet().eval()


def _trivial_args(net):
    return dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),
        alpha=0.01, m=200, ell=199, scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
    )


@pytest.mark.slow
def test_default_method_is_scenario(banana_net):
    """verification_method default = 'scenario' produces identical q to current."""
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, **common)
    r2 = run_verification_pipeline(seed=0, verification_method='scenario', **common)
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-9)
    assert r1['verdict'] == r2['verdict']


def test_unknown_method_raises():
    """Unknown verification_method should raise."""
    with pytest.raises(ValueError, match='verification_method'):
        run_verification_pipeline(
            network=RotatedBananaNet().eval(),
            input_lb=np.array([0.0, 0.0]), input_ub=np.array([1.0, 1.0]),
            spec=HalfSpace(np.array([[1.0]]), np.array([[1.0]])),
            verification_method='nonexistent',
            alpha=0.01, m=100, ell=99, scenario_n_samples=100, scenario_beta=0.001,
            n_train=100, flow_epochs=50, seed=0,
        )
