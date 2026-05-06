"""Verification-method ablation row.

Holds everything else fixed (locked Phase 5d flow + calibration) and
varies only ``verification_method``:

    scenario             Phase 5b/5c default with adaptive 2-stage scenario N.
    scenario_v2          C0: K=5 restarts of certify_spec_on_flow under
                         qmc+antithetic, aggregated by min worst-margin.
    amls                 C1: Adaptive Multilevel Splitting (unbounded; legacy).
    is_tilted            C2: importance sampling with flow-tilted proposal.
    derived              C3: Langevin / MALA latent-space sampler.
    amls_bounded         Production: bounded AMLS with single-halfspace chains.
    amls_bounded_union   Production: bounded AMLS with union-mass chain on
                         phi_union(y) = min_j phi_j(y) for K-disjunct OR groups.
    raw_mc_uniform       Brute-force baseline: one pass of uniform MC on
                         ||z|| <= q + per-group Clopper-Pearson upper bound.
                         Same gate as amls_bounded_union, no level splitting.

The script writes per-method CSVs under
``examples/FlowConformal/experiments/exp_ablation/outputs/`` so
``ablation_aggregate.py`` can reference one canonical location.

Usage:

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.\
ablation_run_verify_method --smoke

Wall-clock estimate per method (full 20-instance probe):

    scenario             ~12 min
    scenario_v2          ~25 min  (5x scenario)
    amls                 ~18 min
    is_tilted            ~15 min  (gate-only)
    derived              ~30 min  (gate-only; autograd through ODE)
    amls_bounded         ~18 min  (same chain machinery as amls; ball-truncated)
    amls_bounded_union   ~18 min  (single chain per group; same wall as amls_bounded
                                   on single-halfspace specs like ACAS Xu)
    raw_mc_uniform       ~ 6 min  (single MC pass; one ODE batch per group)
"""
from __future__ import annotations

import argparse

from examples.FlowConformal.experiments.exp_ablation._common import (
    run_probe_with_overrides,
)


_METHODS = (
    'scenario', 'scenario_v2', 'amls', 'is_tilted', 'derived',
    'amls_bounded', 'amls_bounded_union', 'raw_mc_uniform',
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='run only the first 2 instances on the first method')
    p.add_argument('--methods', nargs='+', choices=_METHODS,
                   default=list(_METHODS),
                   help='subset of methods to run')
    p.add_argument('--probe-size', type=int, default=10,
                   help='# of probe instances to use (default 10 per the '
                        '2026-04-27 paper-experiments plan)')
    args = p.parse_args()

    if args.smoke:
        instances = None  # _common.run_probe_with_overrides resolves to 2
    else:
        from examples.FlowConformal.ablations.phase5c_probe_sweep import (
            _INSTANCES,
        )
        instances = list(_INSTANCES[:args.probe_size])

    methods = args.methods[:1] if args.smoke else args.methods
    for method in methods:
        run_probe_with_overrides(
            tag=f'verify_method_{method}',
            smoke=args.smoke,
            verification_method=method,
            instances=instances,
        )


if __name__ == '__main__':
    main()
