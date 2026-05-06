"""Conformal-parameters ablation row.

Sweeps four conformal knobs while holding everything else at the
Phase 5d locked configuration (verification_method='amls'). Each sweep
varies *one* knob at a time; the others stay at their locked default.

Knobs:
    alpha    in {0.001, 0.01, 0.05, 0.1}   (target miscoverage)
    m        in {500, 2000, 8000}          (calibration size)
    ell off  in {0, 1, 5}                  (offset added to base ell)
                                              base ell = ceil((m+1)*(1-alpha))
                                              actual ell = base - offset
    beta_2   in {0.001, 0.01, 0.1}         (scenario violation budget)

The ``ell offset`` axis trades tightness (small ell -> loose threshold,
fewer calibration samples beat the threshold) for the joint
``epsilon_total`` floor.

This file replaces the old ``ablation_run_calib_size.py`` (which only
swept ``m``) -- the broader knob list aligns with the
2026-04-27 paper-experiments plan.

Usage:

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.\
ablation_run_conformal_params --smoke

Wall-clock estimate (per row, 10-instance probe at AMLS):
    alpha sweep    ~12-20 min/value (4 values  -> ~70 min)
    m sweep        ~12-22 min/value (3 values  -> ~50 min)
    ell-off sweep  ~16-18 min/value (3 values  -> ~50 min)
    beta_2 sweep   ~16-18 min/value (3 values  -> ~50 min)
"""
from __future__ import annotations

import argparse
import math

from examples.FlowConformal.experiments.exp_ablation._common import (
    run_probe_with_overrides,
)


_ALPHA_VALUES = (0.001, 0.01, 0.05, 0.1)
_M_VALUES = (500, 2000, 8000)
_ELL_OFFSETS = (0, 1, 5)
_BETA2_VALUES = (0.001, 0.01, 0.1)

# Locked baseline knob values (per Phase 5d).
_BASE_ALPHA = 0.001
_BASE_M = 8000
_BASE_BETA2 = 0.001


def _ell_for(m: int, alpha: float, offset: int) -> int:
    """Conformal level: base = ceil((m+1)(1-alpha)); subtract offset
    (clamped to [1, m])."""
    base = int(math.ceil((m + 1) * (1.0 - alpha)))
    return max(1, min(m, base - offset))


def _truncated_instances(probe_size):
    """Return the first ``probe_size`` entries of the ACAS Xu probe."""
    from examples.FlowConformal.ablations.phase5c_probe_sweep import _INSTANCES
    if probe_size is None:
        return None
    return list(_INSTANCES[:probe_size])


def _run_alpha_sweep(probe_size: int):
    instances = _truncated_instances(probe_size)
    for alpha in _ALPHA_VALUES:
        ell = _ell_for(_BASE_M, alpha, 0)
        run_probe_with_overrides(
            tag=f'conformal_params_alpha{alpha}',
            verification_method='amls',
            alpha=alpha, m=_BASE_M, ell=ell,
            scenario_beta=_BASE_BETA2,
            instances=instances,
        )


def _run_m_sweep(probe_size: int):
    instances = _truncated_instances(probe_size)
    for m in _M_VALUES:
        ell = _ell_for(m, _BASE_ALPHA, 0)
        run_probe_with_overrides(
            tag=f'conformal_params_m{m}',
            verification_method='amls',
            alpha=_BASE_ALPHA, m=m, ell=ell,
            scenario_beta=_BASE_BETA2,
            instances=instances,
        )


def _run_ell_offset_sweep(probe_size: int):
    instances = _truncated_instances(probe_size)
    for off in _ELL_OFFSETS:
        ell = _ell_for(_BASE_M, _BASE_ALPHA, off)
        run_probe_with_overrides(
            tag=f'conformal_params_elloff{off}',
            verification_method='amls',
            alpha=_BASE_ALPHA, m=_BASE_M, ell=ell,
            scenario_beta=_BASE_BETA2,
            instances=instances,
        )


def _run_beta2_sweep(probe_size: int):
    instances = _truncated_instances(probe_size)
    for beta2 in _BETA2_VALUES:
        ell = _ell_for(_BASE_M, _BASE_ALPHA, 0)
        run_probe_with_overrides(
            tag=f'conformal_params_beta2{beta2}',
            verification_method='amls',
            alpha=_BASE_ALPHA, m=_BASE_M, ell=ell,
            scenario_beta=beta2,
            instances=instances,
        )


_SWEEP_RUNNERS = {
    'alpha': _run_alpha_sweep,
    'm': _run_m_sweep,
    'ell_offset': _run_ell_offset_sweep,
    'beta2': _run_beta2_sweep,
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='one config (alpha=0.01) on first 2 instances')
    p.add_argument('--axis', choices=tuple(_SWEEP_RUNNERS) + ('all',),
                   default='all',
                   help='which knob to sweep (default: all)')
    p.add_argument('--probe-size', type=int, default=10,
                   help='# of probe instances to use (default 10 per the '
                        '2026-04-27 paper-experiments plan; the original '
                        '20-instance probe is preserved in '
                        'phase5c_probe_sweep.py)')
    args = p.parse_args()

    if args.smoke:
        ell = _ell_for(_BASE_M, 0.01, 0)
        run_probe_with_overrides(
            tag='conformal_params_smoke_alpha0.01',
            smoke=True,
            verification_method='amls',
            alpha=0.01, m=_BASE_M, ell=ell,
        )
        return

    axes = list(_SWEEP_RUNNERS) if args.axis == 'all' else [args.axis]
    for axis in axes:
        _SWEEP_RUNNERS[axis](probe_size=args.probe_size)


if __name__ == '__main__':
    main()
