"""Tests for the n2v.probabilistic.verify_flow public API.

These tests cover the Plan B integration: the new top-level entry point
``n2v.probabilistic.verify_flow`` and its backward-compat shim at
``examples.FlowConformal.benchmarks._common.run_verification_pipeline``.

Two key invariants:
  - The new library API defaults ``use_falsifier=False`` (Stage 1
    falsifier OFF). Verdict is UNSAT or UNKNOWN — never SAT.
  - The legacy examples shim defaults ``use_falsifier=True`` (Stage 1
    falsifier ON). This preserves the behavior expected by existing
    example scripts (acasxu_sweep.py, phase5c_probe_sweep.py, ...).
"""
import numpy as np
import pytest
import torch

from examples.FlowConformal.networks import RotatedBananaNet
from n2v.probabilistic import verify_flow
from n2v.sets.halfspace import HalfSpace


@pytest.fixture(scope='module')
def banana_net():
    torch.manual_seed(0)
    return RotatedBananaNet().eval()


@pytest.mark.slow
def test_verify_flow_default_no_falsifier(banana_net):
    """Default ``use_falsifier=False`` — Stage 1 is skipped.

    The unsafe halfspace ``y_0 <= 0.5`` is reachable by the banana, so
    with the falsifier ON we'd expect SAT. With it OFF (default), we
    must NOT see SAT — the verdict is whatever the flow + scenario /
    AMLS verifier produces (UNSAT or UNKNOWN).
    """
    result = verify_flow(
        network=banana_net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
        verification_method='amls',
        seed=0,
    )
    # Falsifier disabled => SAT must not appear; only UNSAT or UNKNOWN.
    assert result['verdict'] in ('UNKNOWN', 'UNSAT')
    assert result['counterexample'] is None


@pytest.mark.slow
def test_verify_flow_with_falsifier_opt_in(banana_net):
    """``use_falsifier=True`` enables Stage 1 — SAT is reachable.

    Same banana + spec as above; with the falsifier on, a real
    counterexample exists in the input box, so we expect SAT.
    """
    result = verify_flow(
        network=banana_net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
        verification_method='amls',
        seed=0,
        use_falsifier=True,
    )
    assert result['verdict'] == 'SAT'
    assert result['counterexample'] is not None
    # Flow training should be skipped on the SAT path.
    assert result['flow'] is None
    assert result['q'] is None


@pytest.mark.slow
def test_legacy_examples_shim_default_falsifier_on(banana_net):
    """Legacy import path defaults ``use_falsifier=True``.

    The examples-side shim must keep the falsifier-on-by-default
    behavior expected by existing scripts (acasxu_sweep.py, etc.). We
    deliberately do NOT pass ``use_falsifier`` here so the shim's
    default kicks in.
    """
    from examples.FlowConformal.benchmarks._common import (
        run_verification_pipeline,
    )

    result = run_verification_pipeline(
        network=banana_net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[0.5]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
        verification_method='amls',
        seed=0,
    )
    # Falsifier on by default => SAT for the easy-banana spec.
    assert result['verdict'] == 'SAT'
    assert result['counterexample'] is not None


def test_verify_flow_unknown_method_raises():
    """``ValueError`` on unknown ``verification_method``.

    Validation happens before flow training, so this is fast: no need
    for the ``slow`` marker.
    """
    with pytest.raises(ValueError, match='verification_method'):
        verify_flow(
            network=RotatedBananaNet().eval(),
            input_lb=np.array([0.0, 0.0]),
            input_ub=np.array([1.0, 1.0]),
            spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[1.0]])),
            verification_method='nonexistent',
            alpha=0.01, m=100, ell=99,
            scenario_n_samples=100, scenario_beta=0.001,
            n_train=100, flow_epochs=50, seed=0,
        )
