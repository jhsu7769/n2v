"""Tests for AMLS + flow-training kwarg exposure at the pipeline level.

Default behavior must be bit-identical to before the kwargs were exposed.
Setting the kwargs to non-default values must produce different results
(verifying the kwarg is actually plumbed through, not just accepted-but-ignored).
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


def _trivial_args(net, *, hs_g: float = -100.0):
    """A tiny, fast UNSAT-or-UNKNOWN config for kwarg-plumbing tests.

    Default unsafe region ``y_0 <= -100`` is unreachable for the banana
    (outputs in ~[0, 1]), so the falsifier never finds a counterexample
    and the pipeline always lands in the flow + verify branch where the
    kwargs we're testing are read.

    ``hs_g``: override the halfspace offset. Closer-to-reachset values
    make AMLS terminate at non-trivial levels, which gives the
    "varied-different" tests something to compare.
    """
    return dict(
        network=net,
        input_lb=np.array([0.0, 0.0]),
        input_ub=np.array([1.0, 1.0]),
        spec=HalfSpace(np.array([[1.0, 0.0]]), np.array([[float(hs_g)]])),
        alpha=0.01, m=200, ell=199,
        scenario_n_samples=200, scenario_beta=0.001,
        n_train=500, flow_epochs=100, flow_config='base',
        verification_method='amls',
        # Disable falsifier to keep us on the flow + verify path
        # (the test harness wants kwarg plumbing exercised).
        sat_backend='none',
    )


@pytest.mark.slow
def test_amls_kwargs_default_unchanged(banana_net):
    """Default AMLS kwargs (None) match pre-exposure behavior.

    Bit-identical: passing nothing must equal explicitly passing None for
    each AMLS kwarg.
    """
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, **common)
    r2 = run_verification_pipeline(
        seed=0,
        amls_quantile=None,
        amls_n_mcmc_steps=None,
        amls_mcmc_step_size=None,
        amls_n_samples_per_level=None,
        amls_max_levels=None,
        **common,
    )
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-9)
    assert r1['verdict'] == r2['verdict']
    # AMLS-specific bookkeeping must also be bit-identical.
    assert r1['amls_levels_used'] == r2['amls_levels_used']
    assert r1['amls_detected_unsafe'] == r2['amls_detected_unsafe']


def _amls_final_phi(result: dict) -> float:
    """Pull the AMLS run's final-phi diagnostic from a pipeline result.

    final_phi is the smallest phi (signed distance to U) seen across all
    samples, including MCMC proposals. It is a continuous function of
    quantile / MCMC step count / step size, so two different
    hyperparameter choices on the same seed produce different final_phi
    values — a reliable "the kwarg is plumbed" signal.
    """
    sr = result['scenario_result']
    amls_result = sr['amls_result']
    return min(
        r.final_phi for grp in amls_result.per_hs_results for r in grp
    )


@pytest.mark.slow
def test_amls_quantile_changes_result(banana_net):
    """Varying amls_quantile produces a different AMLS trace.

    Different rho values change the level-cutoff schedule and thus the
    MCMC trajectories at each level. The deterministic ``final_phi``
    diagnostic (min phi seen across the run) reflects this change.
    """
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, amls_quantile=0.05, **common)
    r2 = run_verification_pipeline(seed=0, amls_quantile=0.5, **common)
    assert _amls_final_phi(r1) != _amls_final_phi(r2), (
        'amls_quantile must affect the AMLS trajectory'
    )


@pytest.mark.slow
def test_amls_n_mcmc_steps_changes_trajectory(banana_net):
    """Varying n_mcmc_steps produces a different AMLS run.

    More MCMC steps => more proposals at each level => different best
    final phi seen across the run.
    """
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, amls_n_mcmc_steps=1, **common)
    r2 = run_verification_pipeline(seed=0, amls_n_mcmc_steps=20, **common)
    assert _amls_final_phi(r1) != _amls_final_phi(r2), (
        'amls_n_mcmc_steps must affect the AMLS trajectory'
    )


@pytest.mark.slow
def test_flow_use_ema_kwarg_changes_q(banana_net):
    """Varying flow_use_ema (on/off) produces a different q.

    EMA averages model parameters across training; turning it off changes
    the final flow weights, which changes calibration scores, which
    changes q.
    """
    common = _trivial_args(banana_net)
    r_on = run_verification_pipeline(seed=0, flow_use_ema=True, **common)
    r_off = run_verification_pipeline(seed=0, flow_use_ema=False, **common)
    assert r_on['q'] != r_off['q'], (
        'EMA on/off should affect flow weights -> q'
    )


@pytest.mark.slow
def test_flow_coupling_kwarg_changes_q(banana_net):
    """Varying flow_coupling produces a different q.

    'sinkhorn' (default) vs 'none' (random pairs) trains the flow on a
    different objective; the resulting weights are different, and so is q.
    """
    common = _trivial_args(banana_net)
    r_sinkhorn = run_verification_pipeline(
        seed=0, flow_coupling='sinkhorn', **common,
    )
    r_random = run_verification_pipeline(
        seed=0, flow_coupling='none', **common,
    )
    assert r_sinkhorn['q'] != r_random['q'], (
        'coupling sinkhorn vs none should affect flow weights -> q'
    )


@pytest.mark.slow
def test_flow_training_kwargs_default_unchanged(banana_net):
    """Default flow-training kwargs (None) match pre-exposure behavior.

    Passing nothing == passing None for each flow-training kwarg.
    """
    common = _trivial_args(banana_net)
    r1 = run_verification_pipeline(seed=0, **common)
    r2 = run_verification_pipeline(
        seed=0,
        flow_use_ema=None,
        flow_coupling=None,
        flow_standardize=None,
        **common,
    )
    assert r1['q'] == pytest.approx(r2['q'], rel=1e-9)
    assert r1['verdict'] == r2['verdict']
