"""Flow-training ablation row.

Holds verification_method='amls' fixed and varies the *training-budget*
knobs:

    n_train      in {1K, 2K, 5K, 10K, 20K, 50K}
    flow_epochs  in {500, 1000, 2000, 5000}

Both knobs are forwarded as ``run_verification_pipeline`` kwargs (see
``examples/FlowConformal/benchmarks/_common.py``).

Phase 1-era flow knobs (standardize, OT coupling, EMA) are intentionally
NOT swept by default -- they are settled and live in the
``--legacy-knobs`` axis below for reviewers who want to reproduce them.
The default sweep focuses on training budget, which the
2026-04-27 paper-experiments plan calls out as the "more training -> more
tightness" story.

Usage:

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.\
ablation_run_flow_training --smoke

Wall-clock estimate per (n_train, flow_epochs) pair (10-instance probe):
    n_train=1K  epochs=500    ~3-5 min
    n_train=5K  epochs=2000   ~14 min   (locked baseline)
    n_train=50K epochs=5000   ~80-120 min (gate-only)

Default sweep is a *grid* of (n_train x flow_epochs) but skips the most
expensive corner (50K x 5000) by default; pass ``--full`` to include it.
"""
from __future__ import annotations

import argparse

from examples.FlowConformal.experiments.exp_ablation._common import (
    run_probe_with_overrides,
)


_N_TRAIN_VALUES = (1_000, 2_000, 5_000, 10_000, 20_000, 50_000)
_FLOW_EPOCHS_VALUES = (500, 1_000, 2_000, 5_000)

# Locked Phase 5d baseline.
_BASE_N_TRAIN = 5_000
_BASE_FLOW_EPOCHS = 2_000

# The (50K, 5000) cell costs >2 hours per probe; gate it behind --full.
_EXPENSIVE_PAIRS = {(50_000, 5_000)}

# --- Legacy axes (kept for reproducibility; not in default sweep) ---
_LEGACY_AXES = {
    'standardize': [
        ('standardize_on', dict(flow_standardize=True)),
        ('standardize_off', dict(flow_standardize=False)),
    ],
    'coupling': [
        ('coupling_random', dict(flow_coupling='none')),
        ('coupling_sinkhorn', dict(flow_coupling='sinkhorn')),
    ],
    'ema': [
        ('ema_on', dict(flow_use_ema=True)),
        ('ema_off', dict(flow_use_ema=False)),
    ],
}


def _truncated_instances(probe_size):
    from examples.FlowConformal.ablations.phase5c_probe_sweep import _INSTANCES
    if probe_size is None:
        return None
    return list(_INSTANCES[:probe_size])


def _run_pair(n_train: int, flow_epochs: int, probe_size: int):
    instances = _truncated_instances(probe_size)
    run_probe_with_overrides(
        tag=f'flow_training_n{n_train}_e{flow_epochs}',
        verification_method='amls',
        n_train=n_train,
        flow_epochs=flow_epochs,
        instances=instances,
    )


def _run_n_train_axis(probe_size: int):
    """Sweep n_train at the locked flow_epochs."""
    for n in _N_TRAIN_VALUES:
        _run_pair(n_train=n, flow_epochs=_BASE_FLOW_EPOCHS,
                  probe_size=probe_size)


def _run_flow_epochs_axis(probe_size: int):
    """Sweep flow_epochs at the locked n_train."""
    for e in _FLOW_EPOCHS_VALUES:
        _run_pair(n_train=_BASE_N_TRAIN, flow_epochs=e,
                  probe_size=probe_size)


def _run_grid(probe_size: int, full: bool):
    """Full grid of (n_train, flow_epochs)."""
    for n in _N_TRAIN_VALUES:
        for e in _FLOW_EPOCHS_VALUES:
            if (n, e) in _EXPENSIVE_PAIRS and not full:
                print(f'[flow_training grid] skipping (n_train={n}, '
                      f'flow_epochs={e}); use --full to include',
                      flush=True)
                continue
            _run_pair(n_train=n, flow_epochs=e, probe_size=probe_size)


def _run_legacy_axis(axis_name: str, probe_size: int):
    """Sweep one of the Phase 1-era knobs at the locked baseline."""
    instances = _truncated_instances(probe_size)
    for tag_suffix, overrides in _LEGACY_AXES[axis_name]:
        run_probe_with_overrides(
            tag=f'flow_training_{tag_suffix}',
            verification_method='amls',
            **overrides,
            instances=instances,
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke', action='store_true',
                   help='one cheap pair (1K, 500) on first 2 instances')
    p.add_argument('--axis',
                   choices=('n_train', 'flow_epochs', 'grid',
                            'legacy_standardize', 'legacy_coupling',
                            'legacy_ema'),
                   default='grid',
                   help='which axis or grid to sweep')
    p.add_argument('--probe-size', type=int, default=10,
                   help='# of probe instances to use (default 10)')
    p.add_argument('--full', action='store_true',
                   help='include the expensive (50K, 5000) cell in --axis grid')
    args = p.parse_args()

    if args.smoke:
        run_probe_with_overrides(
            tag='flow_training_smoke_n1000_e500',
            smoke=True,
            verification_method='amls',
            n_train=1_000, flow_epochs=500,
        )
        return

    if args.axis == 'n_train':
        _run_n_train_axis(args.probe_size)
    elif args.axis == 'flow_epochs':
        _run_flow_epochs_axis(args.probe_size)
    elif args.axis == 'grid':
        _run_grid(args.probe_size, full=args.full)
    elif args.axis.startswith('legacy_'):
        _run_legacy_axis(args.axis.removeprefix('legacy_'), args.probe_size)


if __name__ == '__main__':
    main()
