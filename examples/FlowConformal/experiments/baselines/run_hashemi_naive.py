"""Hashemi-naive probabilistic-verifier runner.

Calls ``n2v.probabilistic.verify(surrogate='naive', ...)`` and converts
the returned ``ProbabilisticBox`` into an UNSAT/UNKNOWN verdict by
checking whether the box is disjoint from the spec's unsafe halfspace
region.

This is a baseline NOT a part of the AMLS pipeline — it uses the legacy
Hashemi center-based surrogate to bound the reachable output set, then
applies a halfspace disjointness check against the unsafe region.

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.baselines.run_hashemi_naive \\
        --benchmark <name> --smoke
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from examples.FlowConformal.experiments.baselines._common import (
    add_common_args, empirical_coverage_for_box, halfspace_disjoint_from_box,
    halfspace_witness_from_samples, load_benchmark_instances,
    resolve_n_instances, resolve_output_csv, run_baseline_sweep, torch_callable,
)


_BASELINE = 'hashemi_naive'
_M = 8000
_ELL = 7999
_EPSILON = 0.001
_N_TEST_COVERAGE = 1000


def _process_factory(seed: int, *, m: int = _M, ell: int | None = None,
                     epsilon: float = _EPSILON):
    """Closure capturing the Hashemi-naive call signature."""
    if ell is None:
        ell = m - 1
    _m = m
    _ell = ell
    _epsilon = epsilon
    from n2v.probabilistic import verify
    from n2v.sets import Box

    def process_one(loader, name):
        try:
            net, boxes, spec, _ = loader()
        except FileNotFoundError as e:
            return {'verdict': 'ERROR', 'error': f'load_missing: {e}'}
        except NotImplementedError as e:
            return {'verdict': 'ERROR', 'error': f'unsupported_spec: {e}'}
        except Exception as e:
            return {'verdict': 'ERROR',
                    'error': f'load {type(e).__name__}: {e}'}

        # Aggregate over OR-of-input-regions: SAT if any box yields a
        # witness, UNSAT if every box's reachset is disjoint from the
        # unsafe region, UNKNOWN otherwise.
        any_unknown = False
        cex_idx = None
        union_lb = None
        union_ub = None
        # Empirical-coverage averages: per-box coverage computed against
        # the per-box reach surrogate, then averaged across boxes
        # (Bonferroni: worst-case is the min, mean is the average across
        # the OR-disjuncts of input regions).
        cov_vals: list[float] = []
        cov_n_total = 0
        for (lb, ub) in boxes:
            input_set = Box(np.asarray(lb).flatten(),
                            np.asarray(ub).flatten())
            model_fn = torch_callable(net)
            try:
                pbox = verify(
                    model=model_fn,
                    input_set=input_set,
                    m=_m, ell=_ell,
                    epsilon=_epsilon,
                    surrogate='naive',
                    seed=seed,
                    verbose=False,
                )
            except Exception as e:
                return {'verdict': 'ERROR',
                        'error': f'verify {type(e).__name__}: {e}'}

            # Empirical coverage on N_TEST_COVERAGE held-out samples
            # drawn uniformly from THIS input box, pushed through the
            # network, and checked for inclusion in pbox = [lb_y, ub_y].
            try:
                cov, _sigma, n_eff = empirical_coverage_for_box(
                    model_fn=model_fn,
                    input_lb=input_set.lb, input_ub=input_set.ub,
                    box_lb=pbox.lb, box_ub=pbox.ub,
                    n_test=_N_TEST_COVERAGE,
                    seed=seed,
                )
                if not np.isnan(cov):
                    cov_vals.append(cov)
                    cov_n_total += n_eff
            except Exception:
                # coverage measurement is diagnostic; never fatal
                pass

            # Quick falsifier: project the surrogate's training samples
            # against the unsafe halfspace. Cheap; if hit -> SAT.
            try:
                lb_samp = input_set.lb.flatten()
                ub_samp = input_set.ub.flatten()
                rng = np.random.default_rng(seed)
                xs = rng.uniform(lb_samp, ub_samp,
                                 size=(min(2048, max(1, _m // 4)),
                                       lb_samp.size)).astype(np.float32)
                ys = model_fn(xs)
                cex_idx = halfspace_witness_from_samples(spec, ys)
                if cex_idx is not None:
                    cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
                    return {
                        'verdict': 'SAT',
                        'm': _m, 'ell': _ell, 'epsilon': _epsilon,
                        'coverage': pbox.coverage,
                        'coverage_empirical': cov_emp,
                        'coverage_n_test': cov_n_total,
                        'confidence': pbox.confidence,
                        'error': '',
                    }
            except Exception:
                # Falsifier failure is non-fatal; continue to bound check.
                pass

            disjoint = halfspace_disjoint_from_box(spec, pbox.lb, pbox.ub)
            if disjoint is True:
                # accumulate union for reporting; verdict UNSAT for this
                # box, but we still need every box to be UNSAT for the
                # overall instance to be UNSAT.
                if union_lb is None:
                    union_lb = np.asarray(pbox.lb, dtype=np.float64)
                    union_ub = np.asarray(pbox.ub, dtype=np.float64)
                else:
                    union_lb = np.minimum(union_lb,
                                          np.asarray(pbox.lb, dtype=np.float64))
                    union_ub = np.maximum(union_ub,
                                          np.asarray(pbox.ub, dtype=np.float64))
            elif disjoint is False:
                any_unknown = True
            else:
                # spec shape unrecognised -> treat as UNKNOWN
                any_unknown = True

        # Final aggregation
        cov_emp = float(np.mean(cov_vals)) if cov_vals else float('nan')
        if any_unknown:
            return {
                'verdict': 'UNKNOWN',
                'm': _m, 'ell': _ell, 'epsilon': _epsilon,
                'coverage': pbox.coverage,
                'coverage_empirical': cov_emp,
                'coverage_n_test': cov_n_total,
                'confidence': pbox.confidence,
                'error': '',
            }
        return {
            'verdict': 'UNSAT',
            'm': _m, 'ell': _ell, 'epsilon': _epsilon,
            'coverage': pbox.coverage,
            'coverage_empirical': cov_emp,
            'coverage_n_test': cov_n_total,
            'confidence': pbox.confidence,
            'error': '',
        }

    return process_one


def main():
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument('--m', type=int, default=_M,
                        help='Calibration set size m (default 8000).')
    parser.add_argument('--epsilon', type=float, default=_EPSILON,
                        help='Miscoverage level (default 1e-3).')
    args = parser.parse_args()

    n = resolve_n_instances(args)
    try:
        instances = load_benchmark_instances(args.benchmark, n)
    except FileNotFoundError as e:
        print(f'[{_BASELINE}] TODO/load failed: {e}', file=sys.stderr)
        sys.exit(0)
    except Exception as e:
        print(f'[{_BASELINE}] load error: {type(e).__name__}: {e}',
              file=sys.stderr)
        sys.exit(2)
    if not instances:
        print(f'[{_BASELINE}] no instances', file=sys.stderr)
        sys.exit(0)

    out_csv = resolve_output_csv(args, _BASELINE)
    extra_fields = ['m', 'ell', 'epsilon', 'coverage',
                    'coverage_empirical', 'coverage_n_test', 'confidence']
    run_baseline_sweep(
        baseline=_BASELINE, benchmark=args.benchmark,
        instances=instances, out_csv=out_csv,
        extra_fields=extra_fields,
        process_one=_process_factory(args.seed, m=args.m, epsilon=args.epsilon),
    )


if __name__ == '__main__':
    main()
