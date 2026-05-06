"""Tests for the HalfSpace → scenario-verify bridge helpers."""

import numpy as np
import pytest


def test_spec_summary_single_row_halfspace():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    hs = HalfSpace(np.array([[1.0, 0.0, -1.0]]), np.array([[2.5]]))
    s = spec_summary(hs)
    assert isinstance(s, str)
    assert '1 constraint' in s or '1 halfspace' in s.lower()
    assert 'dim=3' in s


def test_spec_summary_and_halfspace():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    # 3 rows = AND of 3 halfspaces
    hs = HalfSpace(
        np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]]),
        np.array([[1.0], [1.0], [0.0]]),
    )
    s = spec_summary(hs)
    assert '3 constraint' in s or '3 halfspace' in s.lower()


def test_spec_summary_list_of_halfspaces():
    from examples.FlowConformal.benchmarks._spec import spec_summary
    from n2v.sets.halfspace import HalfSpace
    a = HalfSpace(np.array([[1.0, 0.0]]), np.array([[1.0]]))
    b = HalfSpace(np.array([[0.0, 1.0]]), np.array([[1.0]]))
    s = spec_summary([a, b])
    assert 'OR' in s
    assert '2' in s


import math
import torch

from n2v.probabilistic.flow.model import VelocityField
from n2v.probabilistic.flow.ode import FlowODE
from n2v.probabilistic.flow.train import train_flow


def _train_small_flow(dim: int = 2, seed: int = 0):
    """Train a quick flow on a 2D Gaussian for use in spec tests."""
    torch.manual_seed(seed)
    y_train = torch.randn(1000, dim)  # N(0, I)
    vf = VelocityField(dim=dim, hidden=32, n_layers=2, activation='silu')
    vf, _ = train_flow(
        vf, y_train, n_epochs=100, batch_size=128, lr=1e-3,
        coupling='sinkhorn', sinkhorn_reg='auto', sinkhorn_iters=10,
        use_ema=True, standardize_outputs=False,
    )
    return FlowODE(vf.eval())


def test_certify_spec_on_flow_single_halfspace():
    """certify_spec_on_flow on a single unreachable HalfSpace: unsat_certified=True."""
    from examples.FlowConformal.benchmarks._spec import certify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace
    flow = _train_small_flow(dim=2, seed=0)
    hs = HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))
    result = certify_spec_on_flow(
        flow_ode=flow, threshold_q=5.0, spec=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result['unsat_certified'] is True
    assert result['certifying_group_idx'] == 0
    assert result['epsilon_2'] > 0
    assert 0 < result['delta_2'] < 1
    assert 'spec_summary' in result


def test_certify_spec_on_flow_k_row_halfspace_joint_and():
    """A k-row HalfSpace whose AND intersection is empty → unsat_certified=True."""
    from examples.FlowConformal.benchmarks._spec import certify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace
    flow = _train_small_flow(dim=2, seed=0)
    # Contradictory rows: intersection is empty.
    hs = HalfSpace(
        np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0]]),
        np.array([[-100.0], [-100.0], [-100.0]]),
    )
    result = certify_spec_on_flow(
        flow_ode=flow, threshold_q=5.0, spec=hs,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result['unsat_certified'] is True


def test_certify_spec_on_flow_list_of_dicts_now_supported():
    """OR-of-ANDs (list[dict] / list[HalfSpace]) no longer raises
    NotImplementedError — it's dispatched via layer 3."""
    from examples.FlowConformal.benchmarks._spec import certify_spec_on_flow
    from n2v.sets.halfspace import HalfSpace
    flow = _train_small_flow(dim=2, seed=0)
    # Two groups (AND across), each with one HalfSpace.
    # Group 1: y_0 <= -100 (unreachable). Group 2: y_0 <= 100 (reachable).
    # One group is disjoint → UNSAT.
    spec = [
        {'Hg': HalfSpace(np.array([[1.0, 0.0]]), np.array([[-100.0]]))},
        {'Hg': HalfSpace(np.array([[1.0, 0.0]]), np.array([[100.0]]))},
    ]
    result = certify_spec_on_flow(
        flow_ode=flow, threshold_q=5.0, spec=spec,
        n_samples=500, beta_2=0.001, seed=0,
    )
    assert result['unsat_certified'] is True
