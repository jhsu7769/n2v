"""Small helpers for turning VNN-LIB ``HalfSpace`` specs into human-
readable strings and dispatching them to scenario verification.

Not library code — lives alongside the benchmark harness because the
mapping from spec structure → scenario_verify call is benchmark-level
glue, not a reusable abstraction.
"""
from __future__ import annotations

from typing import Union

from n2v.sets.halfspace import HalfSpace

SpecLike = Union[HalfSpace, list]


def spec_summary(spec: SpecLike) -> str:
    """One-line human-readable description of a VNN-LIB spec.

    Supported inputs (matching ``n2v.utils.load_vnnlib``'s ``prop`` field):
      - Single ``HalfSpace`` (may have multiple rows = AND-of-halfspaces).
      - ``list[HalfSpace]`` (= OR-of-ANDs).

    Args:
        spec: The parsed spec.

    Returns:
        A short string like ``"HalfSpace dim=5, 4 constraints (AND)"`` or
        ``"OR of 2 HalfSpace groups"``.
    """
    if isinstance(spec, HalfSpace):
        n_rows = spec.G.shape[0]
        suffix = " (AND)" if n_rows > 1 else ""
        return (
            f"HalfSpace dim={spec.dim}, "
            f"{n_rows} constraint{'s' if n_rows != 1 else ''}{suffix}"
        )
    if isinstance(spec, dict) and 'Hg' in spec:
        hg = spec['Hg']
        if isinstance(hg, HalfSpace):
            n_rows = hg.G.shape[0]
            return f"AND group: HalfSpace dim={hg.dim}, {n_rows} constraint{'s' if n_rows != 1 else ''}"
        if isinstance(hg, list):
            return f"AND group: OR of {len(hg)} HalfSpaces"
    if isinstance(spec, list):
        if len(spec) == 0:
            return "empty spec"
        # list of dicts (load_vnnlib output): each dict is a property group
        if isinstance(spec[0], dict):
            return f"AND of {len(spec)} property groups"
        return f"OR of {len(spec)} HalfSpace groups"
    raise TypeError(f"unsupported spec type: {type(spec).__name__}")


from typing import Callable, Optional

import numpy as np

from n2v.probabilistic.flow.scenario_verify import (
    certify_spec_disjoint,
    scenario_verify_halfspace,
)
from n2v.utils.verify_specification import _parse_property_groups


def certify_spec_on_flow(
    flow_ode,
    threshold_q: float,
    spec,  # HalfSpace, list[HalfSpace], dict with 'Hg', list[dict]
    *,
    n_samples: int,
    beta_2: float,
    t: float = 1.0,
    n_ode_steps: int = 30,
    ode_method: str = 'rk4',
    ode_atol: float = 1e-5,
    ode_rtol: float = 1e-5,
    seed: 'int | None' = None,
    adaptive_threshold: 'float | None' = None,
    adaptive_n_samples: 'int | None' = None,
    sampling_strategy: str = 'uniform',
) -> dict:
    """UNSAT-certification for an arbitrary VNN-LIB-shaped spec on a
    calibrated flow set.

    Normalizes ``spec`` into canonical ``list[list[HalfSpace]]`` via
    ``n2v.utils.verify_specification._parse_property_groups``, then
    delegates to :func:`certify_spec_disjoint` (layer 3 of the scenario-
    verify three-layer dispatcher). UNSAT-only: never returns SAT (SAT
    detection lives in the falsifier lane — separate from this function).

    Supported ``spec`` shapes (all normalized by ``_parse_property_groups``):
      - Single HalfSpace (1 or k rows): single group, single member.
      - list[HalfSpace]: single group with OR semantics.
      - dict with 'Hg' key: single group from load_vnnlib output.
      - list[dict]: AND across groups (each dict is one group).

    Args:
        flow_ode, threshold_q, n_samples, beta_2, t, n_ode_steps,
        ode_method, ode_atol, ode_rtol, seed: passed to layer 3.
        adaptive_threshold, adaptive_n_samples: forwarded to layer 3
            (and ultimately layer 1). See
            ``n2v.probabilistic.flow.scenario_verify.certify_halfspace_disjoint``
            for semantics. Default None disables adaptive escalation.
        sampling_strategy: forwarded to layer 3 (and ultimately layer 1).
            ``'uniform'`` (default) keeps the original truncated-Gaussian-
            on-the-ball latent sampler; ``'qmc'`` switches to Sobol-based
            QMC sampling of N(0, I_d) for variance reduction. See
            ``n2v.probabilistic.flow.scenario_verify.certify_halfspace_disjoint``
            for semantics.

            **Soundness note:** under ``'qmc'`` samples are drawn from
            the full N(0, I_d), not truncated to the conformal level
            set ``||z|| <= threshold_q``. The scenario bound is still
            well-defined, but the joint composition with the conformal
            layer ``epsilon_total = 1 - (1 - epsilon_1)(1 - epsilon_2)``
            may not be tight (it was derived under the 'uniform'
            truncated assumption). QMC is currently experimental for
            variance-reduction ablations; do not rely on the joint
            epsilon for sound certification under QMC.
        spec: the input spec (any of the shapes above).

    Returns:
        dict with keys
          unsat_certified (bool),
          certifying_group_idx (int | None),
          epsilon_2 (float) — Bonferroni bound over all HalfSpace tests,
          delta_2 (float) — 1 - beta_2,
          n_samples_used (int),
          per_group_results (list[GroupDisjointResult]),
          spec_summary (str) — human-readable one-liner for the input spec.
    """
    # Pre-parse to trigger early type errors from the spec (e.g. unknown
    # structure), then pass the normalized groups to layer 3. If the
    # caller has ALREADY pre-parsed (e.g. the pipeline, which whitens
    # each HalfSpace before calling us), we still re-parse; the parser
    # is idempotent on list[list[HalfSpace]] input:
    if (isinstance(spec, list) and len(spec) > 0
            and all(isinstance(g, list) for g in spec)):
        groups = spec  # already canonical
    else:
        groups = _parse_property_groups(spec)

    result = certify_spec_disjoint(
        flow_ode=flow_ode, threshold_q=threshold_q, spec_groups=groups,
        n_samples=n_samples, beta_2=beta_2,
        t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
        ode_atol=ode_atol, ode_rtol=ode_rtol, seed=seed,
        adaptive_threshold=adaptive_threshold,
        adaptive_n_samples=adaptive_n_samples,
        sampling_strategy=sampling_strategy,
    )
    return {
        'unsat_certified': result.unsat_certified,
        'certifying_group_idx': result.certifying_group_idx,
        'epsilon_2': result.epsilon_2,
        'delta_2': 1.0 - beta_2,
        'n_samples_used': result.n_samples_used,
        'per_group_results': result.per_group_results,
        'spec_summary': spec_summary(spec),
    }


def verify_spec_on_flow(
    flow_ode,
    threshold_q: float,
    spec,
    input_lb=None,
    input_ub=None,
    network=None,
    alpha: 'float | None' = None,
    delta_1: 'float | None' = None,
    beta_2: float = 0.001,
    n_samples: int = 10_000,
    t: float = 1.0,
    n_ode_steps: int = 30,
    preimage_n_restarts: int = 10,
    preimage_n_steps: int = 200,
    preimage_lr: float = 0.05,
    preimage_tolerance: float = 0.1,
    output_shift=None,
    ode_method: str = 'rk4',
) -> dict:
    """DEPRECATED: use :func:`certify_spec_on_flow` instead.

    This shim accepts the old (Phase 2/3) parameter list and translates
    the modern UNSAT-only result into the old ``{verdict, ...}`` shape
    so legacy callers continue to work during the Phase 4 migration.

    Under the new two-lane architecture, this function ONLY returns
    UNSAT or UNKNOWN (never SAT). SAT detection is the falsifier lane's
    job.
    """
    import warnings
    warnings.warn(
        "verify_spec_on_flow is deprecated; use certify_spec_on_flow. "
        "The new function handles AND-of-OR-of-AND specs natively and "
        "returns UNSAT-only results (SAT detection is handled by the "
        "falsifier lane).",
        DeprecationWarning, stacklevel=2,
    )
    # Drop kwargs the new API doesn't accept:
    _ = (input_lb, input_ub, network, alpha, delta_1,
         preimage_n_restarts, preimage_n_steps, preimage_lr,
         preimage_tolerance, output_shift)
    res = certify_spec_on_flow(
        flow_ode=flow_ode, threshold_q=threshold_q, spec=spec,
        n_samples=n_samples, beta_2=beta_2,
        t=t, n_ode_steps=n_ode_steps, ode_method=ode_method,
        seed=None,
    )
    verdict = 'UNSAT' if res['unsat_certified'] else 'UNKNOWN'
    return {
        'verdict': verdict,
        'epsilon_2': res['epsilon_2'],
        'delta_2': res['delta_2'],
        'n_samples_used': res['n_samples_used'],
        'counterexample': None,
        'per_constraint_results': [],
    }
