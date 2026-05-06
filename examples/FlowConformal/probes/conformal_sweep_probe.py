"""Conformal (m, ell) sweep probe (read-only diagnostic).

Tests the central architectural claim: in our framework training and
calibration are decoupled, so sweeping the conformal calibration
parameters (m, ell) is essentially free given a single trained flow.
This contrasts with Hashemi/SaVer where ``m`` is the only knob and
controls both compute and tightness — sweeping it requires re-running
the entire benchmark.

What the probe does
-------------------

1. Run the standard pipeline once with ``m = m_max`` to train the flow
   and get the calibrated radius ``q_max``.
2. Reuse the cached trained flow and score function. Generate ``m_max``
   fresh calibration samples (the first ``m`` rows of the ``m_max``
   sample equal a standalone size-``m`` draw with the same seed because
   :func:`sample_box` is a streaming RNG).
3. For each ``(m, ell)`` sweep point: slice the first ``m`` scores,
   take the ``ell``-th order statistic → ``q(m, ell)``. Then re-run
   AMLS verification (``amls_bounded_certify_spec_union``) with the
   new ``q``.
4. Record per-sweep-point: ``q``, ``eps_2_upper``, ``detected_unsafe``,
   ``levels_used``, AMLS wall time. Plus the one-shot training time so
   we can report amortized per-sweep-point cost.

Rationale: if ``q(m, ell)`` is meaningfully smaller for larger ``m`` /
smaller ``ell`` and that smaller ``q`` actually shrinks the lsnc_relu
spurious-witness mass below the AMLS detection floor, this knob is
the missing fix for the over-extrapolation false UNKNOWNs.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.conformal_sweep_probe \\
        --benchmark lsnc_relu --instance-idx 0
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.probabilistic.flow.amls_bounded import (
    amls_bounded_certify_spec_union,
)
from n2v.probabilistic.flow.calibrate import calibrate
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import (
    _forward, _whiten_halfspace, run_verification_pipeline,
)
from n2v.utils.verify_specification import (
    _parse_property_groups, distribute_and_of_or_of_and,
)


_DEFAULT_M_SWEEP = (500, 1000, 2000, 4000, 8000, 16000)
_DEFAULT_ELL_FRACTIONS = (1.0, 0.999, 0.99, 0.95)
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'


def _confidence_delta_2(m: int, ell: int, epsilon: float) -> float:
    """δ_2 = 1 - betacdf_{1-ε}(ell, m+1-ell). Same formula as
    :func:`n2v.probabilistic.conformal.compute_confidence`. Returned
    so we can plot the (m, ell) → δ_2 surface alongside q."""
    from scipy.stats import beta as _beta_dist
    return float(1.0 - _beta_dist.cdf(1.0 - epsilon, ell, m + 1 - ell))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--m-sweep', type=int, nargs='+', default=list(_DEFAULT_M_SWEEP),
                   help='List of m values to sweep (default: 500..16000).')
    p.add_argument('--ell-fractions', type=float, nargs='+',
                   default=list(_DEFAULT_ELL_FRACTIONS),
                   help='List of ell/m ratios to sweep (default: 1.0, 0.999, 0.99, 0.95).')
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    m_max = max(args.m_sweep)
    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[csweep] {args.benchmark} idx={args.instance_idx}: {vnn_rel}')
    print(f'[csweep] m_sweep={args.m_sweep}  ell_fractions={args.ell_fractions}')
    print(f'[csweep] m_max={m_max}')

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    # ---- Step 1: train flow once with m=m_max ----
    print(f'[csweep] training flow once with m={m_max} (this is the dominant '
          f'one-shot cost)...')
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    base_result = run_verification_pipeline(
        network=network, input_lb=lb, input_ub=ub, spec=spec,
        alpha=cfg['alpha'],
        m=m_max, ell=m_max,  # use ell=m for the seed run (reset per sweep)
        n_train=cfg['n_train'],
        flow_epochs=cfg['flow_epochs'],
        flow_config=cfg['flow_config'],
        scenario_n_samples=cfg['scenario_n_samples'],
        scenario_beta=0.001,
        verification_method=cfg['verification_method'],
        amls_max_levels=cfg['amls_max_levels'],
        seed=args.seed,
        use_falsifier=False,
    )
    base_wall = time.time() - t0
    flow = base_result['flow']
    score_fn = base_result['score_fn']
    y_mean_np = base_result['y_mean']
    y_std_np = base_result['y_std']
    q_max = base_result['q']
    print(f'[csweep] base run: wall={base_wall:.1f}s  q_max={q_max:.4f}  '
          f'verdict={base_result["verdict"]}  '
          f'eps_2={base_result.get("amls_bounded_eps_2_upper")}')

    # ---- Step 2: regenerate m_max calibration samples (deterministic) ----
    # The pipeline uses cal_seed=seed and shifts by 1_000_000 for the
    # calibration draw. We replicate that exact RNG path so the first m
    # rows of our m_max sample match what the pipeline used internally.
    cal_seed = args.seed + 1_000_000
    lb_t = torch.as_tensor(lb, dtype=torch.float32)
    ub_t = torch.as_tensor(ub, dtype=torch.float32)
    x_ca = _sample_box(lb_t, ub_t, n_samples=m_max, seed=cal_seed)
    y_ca = _forward(network, x_ca)
    # score_fn is the whitening wrapper; it does (y - y_mean) / y_std
    # internally before calling the underlying FlowScore.
    with torch.no_grad():
        all_scores = score_fn(y_ca).detach().cpu().numpy()
    print(f'[csweep] regenerated {len(all_scores)} calibration scores '
          f'(first 5: {all_scores[:5].round(4)})')

    # ---- Step 3: whitened spec for AMLS ----
    raw_groups = _parse_property_groups(spec)
    raw_groups = distribute_and_of_or_of_and(raw_groups)
    whitened_groups = [
        [_whiten_halfspace(hs, y_mean_np, y_std_np) for hs in group]
        for group in raw_groups
    ]

    # ---- Step 4: sweep (m, ell) → q → AMLS ----
    rows = []
    print(f'[csweep] sweeping {len(args.m_sweep)} m × {len(args.ell_fractions)} '
          f'ell-fractions = {len(args.m_sweep) * len(args.ell_fractions)} points')
    print(f'{"m":>6} {"ell":>6} {"ell/m":>6} {"q":>10} {"d2":>10} {"verdict":>9} '
          f'{"eps_2":>10} {"det":>4} {"K":>3} {"wall_s":>7}')
    for m in args.m_sweep:
        scores_m = all_scores[:m]
        sorted_scores_m = np.sort(scores_m)
        for ell_frac in args.ell_fractions:
            ell = max(1, min(m, int(round(m * ell_frac))))
            q = float(sorted_scores_m[ell - 1])
            d2 = _confidence_delta_2(m, ell, cfg['alpha'])

            # Run AMLS with this q
            t1 = time.time()
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            amls_res = amls_bounded_certify_spec_union(
                flow_ode=flow,
                spec_groups=whitened_groups,
                q=q,
                eps_2_target=cfg['alpha'],
                n_samples_per_level=cfg['scenario_n_samples'],
                quantile=0.1,
                max_levels=cfg['amls_max_levels'],
                n_mcmc_steps=10,
                mcmc_step_size=0.3,
                adaptive_step=False,
                beta=0.001,
                seed=args.seed,
            )
            wall = time.time() - t1

            verdict = ('UNSAT' if amls_res.unsat_certified
                       else 'UNKNOWN')
            eps_2 = float(amls_res.eps_2_upper)
            det = bool(amls_res.detected_any)
            # Find the levels_used max across per-HS results
            K_used = max(
                (r.levels_used for grp in amls_res.per_hs_results for r in grp),
                default=0,
            )
            print(f'{m:>6} {ell:>6} {ell_frac:>6.3f} {q:>10.4f} {d2:>10.4f} '
                  f'{verdict:>9} {eps_2:>10.3e} {str(det):>4} {K_used:>3} '
                  f'{wall:>7.1f}')
            rows.append({
                'benchmark': args.benchmark,
                'instance_idx': args.instance_idx,
                'vnn_rel': vnn_rel,
                'm': m, 'ell': ell, 'ell_frac': ell_frac,
                'q': q, 'delta_2': d2,
                'verdict': verdict, 'eps_2_upper': eps_2,
                'detected_unsafe': det, 'levels_used': K_used,
                'amls_wall_s': wall,
                'one_shot_train_wall_s': base_wall,
            })

    # ---- Output CSV ----
    out_csv = (_OUT_DIR /
               f'conformal_sweep_{args.benchmark}_inst{args.instance_idx}.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        if not rows: return
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f'\n[csweep] wrote {out_csv}')

    # Quick summary
    qs = sorted(set(r['q'] for r in rows))
    eps_2s = [r['eps_2_upper'] for r in rows]
    verdicts = set(r['verdict'] for r in rows)
    dets = set(r['detected_unsafe'] for r in rows)
    print(f'[csweep] q range: [{min(qs):.4f}, {max(qs):.4f}]  '
          f'verdict set: {verdicts}  detection set: {dets}')
    print(f'[csweep] eps_2 range: [{min(eps_2s):.3e}, {max(eps_2s):.3e}]')
    total_amls_wall = sum(r['amls_wall_s'] for r in rows)
    print(f'[csweep] total AMLS wall (sweep): {total_amls_wall:.1f}s '
          f'+ one-shot training: {base_wall:.1f}s')
    hashemi_equivalent_cost = base_wall * len(rows)
    print(f'[csweep] equivalent Hashemi cost (re-run per sweep point): '
          f'~{hashemi_equivalent_cost:.0f}s')


if __name__ == '__main__':
    main()
