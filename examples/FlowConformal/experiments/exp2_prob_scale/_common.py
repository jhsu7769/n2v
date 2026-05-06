"""Shared utilities for Exp 2 probabilistic-scale benchmark sweeps.

Locked Phase 5d config: verification_method='amls', alpha=0.001, n_train=5000,
flow_epochs=2000, scenario_n_samples=2000, scenario_beta=0.001, flow_config='base'.

Each per-benchmark script accepts:
  --falsify-first / --no-falsify-first    enable / disable Stage-1
                     falsifier. Default ON for parity with VNN-COMP-
                     derived sound-verifier cex extraction (vit_2023,
                     tinyimagenet_2024, cifar100_2024) and adversarial-
                     robustness benchmarks (cifar10_resnet110).
  --instances N      run only first N instances (default: 100)
  --smoke            run 2 instances for smoke test (overrides --instances)
  --timeout S        per-instance soft timeout (default: 600 s)
  --output-csv PATH  override output path
  --seed K           master seed (per-instance hash applied internally)

Probabilistic baselines (Hashemi-clip, RS, SAVER, ProbStar) are NOT run
here — they are invoked separately by the user.
"""
from __future__ import annotations

import csv
import json
import signal
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from examples.FlowConformal.benchmarks._common import run_verification_pipeline


# --- Locked Phase 5d config ---
_FLOW_CONFIG = 'base'
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_SCENARIO_BETA = 0.001
_ALPHA = 0.001
_VERIFICATION_METHOD = 'amls'

_DEFAULT_TIMEOUT_S = 600
_DEFAULT_INSTANCES = 100
_SMOKE_INSTANCES = 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def add_common_args(parser) -> None:
    """Attach the standard Exp 2 flags to ``parser``."""
    parser.add_argument(
        '--falsify-first', dest='falsify_first', action='store_true',
        default=True,
        help='Enable Stage-1 falsifier before flow-conformal (default ON '
             'for parity with VNN-COMP-derived sound-verifier cex '
             'extraction and adversarial-robustness benchmarks).')
    parser.add_argument(
        '--no-falsify-first', dest='falsify_first', action='store_false',
        help='Disable Stage-1 falsifier (overrides default).')
    parser.add_argument('--instances', type=int, default=_DEFAULT_INSTANCES,
                        help='Run first N instances (default 100).')
    parser.add_argument('--smoke', action='store_true',
                        help='Run only 2 instances; overrides --instances.')
    parser.add_argument('--timeout', type=int, default=_DEFAULT_TIMEOUT_S,
                        help='Per-instance soft timeout in seconds.')
    parser.add_argument('--seed', type=int, default=0,
                        help='Master seed; per-instance hash applied.')
    parser.add_argument('--output-csv', type=Path, default=None,
                        help='Override default output path.')


def get_pipeline_kwargs(falsify_first: bool) -> dict:
    """Return kwargs for ``run_verification_pipeline`` matching the Phase 5d
    locked config. ``falsify_first`` opts into Stage-1 falsification per
    Plan B (legacy shim default is True; we set it explicitly here).
    """
    kwargs = dict(
        alpha=_ALPHA,
        n_train=_N_TRAIN,
        flow_epochs=_FLOW_EPOCHS,
        flow_config=_FLOW_CONFIG,
        scenario_n_samples=_SCENARIO_N,
        scenario_beta=_SCENARIO_BETA,
        verification_method=_VERIFICATION_METHOD,
        use_falsifier=falsify_first,
    )
    return kwargs


# ---------------------------------------------------------------------------
# Sweep loop
# ---------------------------------------------------------------------------

def _raise_timeout(signum, frame):
    raise TimeoutError()


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


# An "instance" is a callable that returns ``(network, boxes, spec, name)``
# where ``boxes`` is a list of (lb, ub) flat 1D pairs.
InstanceLoader = Callable[[], tuple]


def run_one_instance(loader: InstanceLoader, *, seed: int,
                     pipeline_kwargs: dict) -> dict:
    """Load and run our pipeline on one instance. Never raises — load /
    run failures become ERROR rows.
    """
    try:
        out = loader()
    except NotImplementedError as e:
        return {'verdict': 'SKIPPED',
                'error': f'{type(e).__name__}: {e}'}
    except TimeoutError:
        raise
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'loadfailed {type(e).__name__}: {e}'}

    if out is None:
        return {'verdict': 'SKIPPED', 'error': 'loader returned None'}
    network, boxes, spec, _ = out

    box_results = []
    for box_idx, (lb, ub) in enumerate(boxes):
        try:
            r = run_verification_pipeline(
                network=network,
                input_lb=np.asarray(lb).flatten(),
                input_ub=np.asarray(ub).flatten(),
                spec=spec,
                seed=seed + 7919 * box_idx,
                **pipeline_kwargs,
            )
        except NotImplementedError as e:
            return {'verdict': 'SKIPPED',
                    'error': f'{type(e).__name__}: {e}'}
        except TimeoutError:
            # Let SIGALRM TimeoutError propagate to the outer sweep loop,
            # which records a TIMEOUT verdict.
            raise
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'runfailed box={box_idx} '
                             f'{type(e).__name__}: {e}'}
        box_results.append(r)
        if r['verdict'] == 'SAT':
            break

    verdicts = [r['verdict'] for r in box_results]
    if 'SAT' in verdicts:
        result = next(r for r in box_results if r['verdict'] == 'SAT')
    elif all(v == 'UNSAT' for v in verdicts):
        result = box_results[0]
        if len(box_results) > 1:
            eps_sum = sum((r.get('epsilon_total') or 0.0) for r in box_results)
            delta_min = min((r.get('delta_total') or 1.0) for r in box_results)
            result = dict(result)
            result['epsilon_total'] = eps_sum
            result['delta_total'] = delta_min
    else:
        result = next(r for r in box_results if r['verdict'] == 'UNKNOWN')

    cex_x, cex_y = '', ''
    if result.get('counterexample') is not None:
        ce = result['counterexample']
        cex_x = json.dumps(np.asarray(ce['x']).tolist())
        cex_y = json.dumps(np.asarray(ce['y']).tolist())
    amls_lvls = result.get('amls_levels_used')
    return {
        'verdict': result['verdict'],
        'coverage': _fmt(result.get('coverage_empirical'), '.4f'),
        'q': _fmt(result.get('q'), '.4f'),
        'epsilon_total': _fmt(result.get('epsilon_total'), '.4f'),
        'delta_total': _fmt(result.get('delta_total'), '.4f'),
        'train_s': _fmt(result.get('flow_train_time_s'), '.1f'),
        'verify_s': _fmt(result.get('verification_time_s'), '.1f'),
        'total_s': _fmt(result.get('total_time_s'), '.1f'),
        'amls_levels_used': str(amls_lvls) if amls_lvls is not None else '',
        'cex_x': cex_x, 'cex_y': cex_y,
        'error': '',
    }


def run_sweep(
    benchmark_name: str,
    instances: list,  # list of (instance_name, InstanceLoader) tuples
    out_csv: Path,
    pipeline_kwargs: dict,
    timeout_s: int = _DEFAULT_TIMEOUT_S,
    seed: int = 0,
):
    """Generic sweep loop. Writes one CSV row per instance. Per-instance
    soft timeout enforced via ``signal.SIGALRM``.
    """
    if not instances:
        print(f'[{benchmark_name}] no instances to run', file=sys.stderr)
        return

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f'[{benchmark_name}] running {len(instances)} instances; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark_name}] config: flow_config={_FLOW_CONFIG} '
          f'n_train={_N_TRAIN} flow_epochs={_FLOW_EPOCHS} alpha={_ALPHA} '
          f'method={_VERIFICATION_METHOD} timeout={timeout_s}s '
          f'falsify_first={pipeline_kwargs.get("use_falsifier", False)}',
          flush=True)

    fields = ['benchmark', 'instance_name', 'seed', 'verdict',
              'coverage', 'q', 'epsilon_total', 'delta_total',
              'train_s', 'verify_s', 'total_s', 'amls_levels_used',
              'cex_x', 'cex_y', 'error']

    signal.signal(signal.SIGALRM, _raise_timeout)
    counts = {}
    t_start = time.time()

    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()
        for k, (name, loader) in enumerate(instances, start=1):
            base = hash((benchmark_name, name)) & 0x7FFFFFFF
            instance_seed = (base + seed * 31337) & 0x7FFFFFFF
            elapsed = time.time() - t_start
            print(f'[{benchmark_name} {k}/{len(instances)} '
                  f't={elapsed:.0f}s] {name}', flush=True)
            t0 = time.time()
            try:
                signal.alarm(timeout_s)
                row = run_one_instance(loader, seed=instance_seed,
                                       pipeline_kwargs=pipeline_kwargs)
            except TimeoutError:
                row = {'verdict': 'TIMEOUT',
                       'error': f'per-instance timeout {timeout_s}s'}
            finally:
                signal.alarm(0)

            out_row = {fld: '' for fld in fields}
            out_row['benchmark'] = benchmark_name
            out_row['instance_name'] = name
            out_row['seed'] = seed
            for k2, v in row.items():
                out_row[k2] = v
            writer.writerow(out_row)
            f.flush()

            v = row['verdict']
            counts[v] = counts.get(v, 0) + 1
            print(f'    verdict={v}  wall={time.time() - t0:.1f}s',
                  flush=True)

    print(f'\n[{benchmark_name}] === sweep complete ===')
    print(f'[{benchmark_name}] wrote {out_csv}')
    print(f'[{benchmark_name}] total wall-clock: '
          f'{(time.time() - t_start) / 60:.1f} min')
    print(f'[{benchmark_name}] counts: {counts}')


# ---------------------------------------------------------------------------
# Helpers shared across image-classification benchmarks
# ---------------------------------------------------------------------------

def make_classification_robustness_spec(num_classes: int, true_class: int):
    """Build a list[HalfSpace] encoding the UNSAFE region for classification
    robustness: for each ``j != true_class`` add ``f_j(x) - f_{true}(x) > 0``,
    i.e. ``HalfSpace`` row ``e_j - e_{true}`` with rhs = 0 (using the
    convention ``G y <= g`` encodes the UNSAFE region — see
    ``project_sat_unsat_convention``).

    The pipeline interprets a list[HalfSpace] as OR-of-ANDs (single-disjunct
    case): the unsafe region is ``UNION_j {y : G_j y <= g_j}``, which is
    "any other class beats the true class". UNSAT certifies "no other class
    can beat ``true_class``" within the input ball.
    """
    from n2v.sets.halfspace import HalfSpace

    halfspaces = []
    for j in range(num_classes):
        if j == true_class:
            continue
        # row: e_j - e_true; we want f_j > f_true, i.e. f_true - f_j < 0,
        # so HalfSpace row (e_true - e_j) y <= 0 (G y <= g, g=0).
        row = np.zeros(num_classes, dtype=np.float64)
        row[true_class] = 1.0
        row[j] = -1.0
        # G y <= 0 means y_true - y_j <= 0, i.e. y_j >= y_true (unsafe).
        G = row.reshape(1, -1)
        g = np.zeros((1, 1), dtype=np.float64)
        halfspaces.append(HalfSpace(G, g))
    return halfspaces


def linf_box_for_image(x: np.ndarray, eps: float,
                       lo: float = 0.0, hi: float = 1.0
                       ) -> tuple[np.ndarray, np.ndarray]:
    """Return the clipped L∞ box ``[x - eps, x + eps] ∩ [lo, hi]^d`` as
    flattened (lb, ub) arrays.
    """
    x_flat = np.asarray(x, dtype=np.float64).flatten()
    lb = np.clip(x_flat - eps, lo, hi)
    ub = np.clip(x_flat + eps, lo, hi)
    return lb, ub
