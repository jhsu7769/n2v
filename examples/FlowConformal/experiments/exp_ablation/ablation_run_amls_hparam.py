"""AMLS hyperparameter ablation row.

Two sweeps under verification_method='amls':

    rho (quantile) in {0.05, 0.10, 0.20}
    n_mcmc_steps   in {5, 10, 20, 40}

The pipeline now exposes ``amls_quantile``, ``amls_n_mcmc_steps``, and
``amls_mcmc_step_size`` as direct ``run_verification_pipeline`` kwargs
(see ``examples/FlowConformal/benchmarks/_common.py``). The previous
monkey-patch on ``amls_certify_spec`` has been removed in favour of
straight kwarg forwarding via ``run_probe_with_overrides``.

Usage:

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.\
ablation_run_amls_hparam --smoke

Wall-clock estimate (full 20-instance probe per value):
    rho sweep:       ~18 min/value  (3 values -> ~55 min)
    mcmc-steps:      ~12-30 min/value depending on n_mcmc_steps
                       (4 values -> ~80 min)
"""
from __future__ import annotations

import argparse

from examples.FlowConformal.experiments.exp_ablation._common import (
    run_probe_with_overrides,
)


_RHO_VALUES = (0.05, 0.10, 0.20)
_MCMC_STEP_VALUES = (5, 10, 20, 40)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='single config (rho=0.1, steps=10) on first 2 instances')
    p.add_argument('--axis', choices=('rho', 'mcmc', 'both'), default='both')
    p.add_argument('--probe-size', type=int, default=10,
                   help='# of probe instances to use (default 10 per the '
                        '2026-04-27 paper-experiments plan)')
    args = p.parse_args()

    if args.smoke:
        # Smoke: just one rho value to validate harness end-to-end.
        run_probe_with_overrides(
            tag='amls_hparam_smoke_rho0.1_steps10',
            smoke=True,
            verification_method='amls',
            amls_quantile=0.1,
            amls_n_mcmc_steps=10,
        )
        return

    from examples.FlowConformal.ablations.phase5c_probe_sweep import (
        _INSTANCES,
    )
    instances = list(_INSTANCES[:args.probe_size])

    if args.axis in ('rho', 'both'):
        for rho in _RHO_VALUES:
            run_probe_with_overrides(
                tag=f'amls_hparam_rho{rho}',
                verification_method='amls',
                amls_quantile=rho,
                instances=instances,
            )

    if args.axis in ('mcmc', 'both'):
        for steps in _MCMC_STEP_VALUES:
            run_probe_with_overrides(
                tag=f'amls_hparam_mcmc{steps}',
                verification_method='amls',
                amls_n_mcmc_steps=steps,
                instances=instances,
            )


if __name__ == '__main__':
    main()
