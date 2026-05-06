"""Tests for flow_ensemble_size kwarg."""
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
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]])),  # easy UNSAT
        alpha=0.01, m=200, ell=199, scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
    )


@pytest.mark.slow
def test_ensemble_size_1_matches_default(banana_net):
    """flow_ensemble_size=1 should match no-ensemble behavior bit-identically."""
    common = _trivial_args(banana_net)
    r_default = run_verification_pipeline(seed=0, **common)
    r_size1 = run_verification_pipeline(seed=0, flow_ensemble_size=1, **common)
    assert r_default['q'] == pytest.approx(r_size1['q'], rel=1e-9)
    assert r_default['verdict'] == r_size1['verdict']


@pytest.mark.slow
def test_ensemble_size_3_changes_q(banana_net):
    """ensemble of 3 with max-score should give a different (typically larger) q."""
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, **common)
    r3 = run_verification_pipeline(seed=0, flow_ensemble_size=3, **common)
    assert r1['q'] != r3['q']
    # Conservative score => q should be at least as large as single-flow q
    # (in expectation; small N may break this, but typically holds)
    # Just check it's different and the result is well-formed.
    assert r3.get('flow_ensemble_size') == 3
    assert r3['verdict'] in ('UNSAT', 'SAT', 'UNKNOWN')


@pytest.mark.slow
def test_ensemble_runs_end_to_end(banana_net):
    """Smoke test that ensemble doesn't crash on a real spec."""
    common = _trivial_args(banana_net)
    r = run_verification_pipeline(seed=0, flow_ensemble_size=3, **common)
    assert 'q' in r
    assert 'verdict' in r
    assert r['flow_ensemble_size'] == 3
