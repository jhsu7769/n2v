"""Witness-spuriousness probe (read-only diagnostic).

Tests whether AMLS-found witnesses for ground-truth-UNSAT instances
correspond to real reachable points in the input box, or are pure
flow-extrapolation artifacts.

The hypothesis (from the trajectory probe + lsnc_relu UNKNOWN audit):
on lsnc_relu, AMLS reports ``detected_unsafe=True`` in 1-3 levels
even on instances that αβ-CROWN proved UNSAT. The candidate
explanation is that the flow's distribution Q(y) puts small but
nonzero mass on U because a continuous Gaussian-pushforward smears
beyond the network's true reach manifold. The conformal q-ball
calibration controls coverage of true outputs but NOT over-coverage
beyond the reach set, so AMLS legitimately finds witnesses in the
flow's spurious-mass region.

To confirm: independently run APGD with many restarts on the *actual*
network and input box. If APGD finds nothing on instances where AMLS
found a witness, that is empirical evidence the AMLS witness is
flow-extrapolation, not a real adversarial point.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.witness_spurious_probe \\
        --benchmark lsnc_relu --instance-idx 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG,
    list_instances,
    load_one_instance,
)
from n2v.probabilistic.verify_flow import run_verification_pipeline
from n2v.utils.falsify import falsify


def _phi_for_property(model, x_np, lb_shape, spec):
    """Compute the AND-of-OR-of-AND ``phi(network(x))`` for a single
    flat input ``x_np``. Lower phi = closer to / inside U.
    """
    from n2v.utils.verify_specification import _parse_property_groups
    groups = _parse_property_groups(spec)
    x = torch.from_numpy(np.asarray(x_np, dtype=np.float32)).reshape(1, *lb_shape)
    if hasattr(model, 'parameters'):
        dev = next(model.parameters()).device
        x = x.to(dev)
    with torch.no_grad():
        y = model(x).flatten().detach().cpu().numpy()
    # AND across groups -> max; OR within group -> min; AND within
    # halfspace (rows of G) -> max(G y - g).
    group_phis = []
    for group in groups:
        per_hs = []
        for hs in group:
            G = np.asarray(hs.G, dtype=np.float64)
            g = np.asarray(hs.g, dtype=np.float64).flatten()
            per_hs.append(float((G @ y - g).max()))
        group_phis.append(min(per_hs))
    return float(max(group_phis)), y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--apgd-restarts', type=int, default=50,
                   help='APGD restarts (default 50 — a lot, to stress-test).')
    p.add_argument('--apgd-steps', type=int, default=200,
                   help='APGD steps per restart (default 200).')
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, vnncomp_t = instances[args.instance_idx]
    print(f'[wit] {args.benchmark} idx={args.instance_idx}: {vnn_rel}')

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    # ---- 1. Run standard pipeline (use_falsifier=False to isolate AMLS) ----
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    result = run_verification_pipeline(
        network=network, input_lb=lb, input_ub=ub, spec=spec,
        alpha=cfg['alpha'],
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
    pipeline_wall = time.time() - t0
    print(f'[wit] pipeline: verdict={result["verdict"]}  '
          f'eps_2={result.get("amls_bounded_eps_2_upper")}  '
          f'detected={result.get("amls_bounded_detected_unsafe")}  '
          f'levels={result.get("amls_levels_used")}  '
          f'wall={pipeline_wall:.1f}s')

    # Dig the worst_y out of the per-halfspace AMLSBoundedResult list
    # (the pipeline doesn't surface it at the top level for union mode).
    amls_b_result = result.get('amls_bounded_result')
    worst_y = None
    if amls_b_result is not None:
        for group in (amls_b_result.per_hs_results or []):
            for hs_res in group:
                if (hs_res is not None
                        and getattr(hs_res, 'detected_unsafe', False)
                        and getattr(hs_res, 'worst_y', None) is not None):
                    worst_y = np.asarray(hs_res.worst_y, dtype=np.float64)
                    break
            if worst_y is not None:
                break

    if worst_y is not None:
        from n2v.utils.verify_specification import _parse_property_groups
        groups = _parse_property_groups(spec)
        per_group = []
        for group in groups:
            per_hs = []
            for hs in group:
                G = np.asarray(hs.G, dtype=np.float64)
                g = np.asarray(hs.g, dtype=np.float64).flatten()
                per_hs.append(float((G @ worst_y - g).max()))
            per_group.append(min(per_hs))
        amls_phi = float(max(per_group))
        print(f'[wit] AMLS witness y* phi = {amls_phi:.4e}  '
              f'(<=0 means in U)  ||y*||={float(np.linalg.norm(worst_y)):.4f}')

    # ---- 2. Independent APGD on the actual network/input box ----
    print(f'[wit] running APGD: {args.apgd_restarts} restarts x '
          f'{args.apgd_steps} steps on actual network...')
    t0 = time.time()
    fr_result, fr_cex = falsify(
        network, lb, ub, spec, method='apgd',
        n_restarts=args.apgd_restarts, n_steps=args.apgd_steps,
        seed=args.seed,
    )
    apgd_wall = time.time() - t0
    apgd_verdict = 'SAT' if fr_result == 0 else 'UNKNOWN'
    print(f'[wit] APGD: verdict={apgd_verdict}  wall={apgd_wall:.1f}s')

    if fr_cex is not None:
        x_cex, y_cex = fr_cex
        cex_phi, _ = _phi_for_property(network, x_cex, lb.shape, spec)
        dist_str = ''
        if worst_y is not None:
            dist = float(np.linalg.norm(y_cex.flatten() - worst_y.flatten()))
            dist_str = f'  ||y_cex - y*||={dist:.4f}'
        print(f'[wit] APGD counterexample: phi(network(x))={cex_phi:.4e}'
              f'{dist_str}')

    # ---- 3. Verdict reconciliation ----
    print()
    print('[wit] === reconciliation ===')
    amls_detected = bool(result.get('amls_bounded_detected_unsafe', False))
    apgd_detected = (fr_result == 0)
    if amls_detected and not apgd_detected:
        print('[wit] AMLS witness is SPURIOUS: AMLS found y* ∈ U under flow,')
        print('[wit]   but APGD with many restarts could not find any')
        print(f'[wit]   x ∈ input_box with network(x) ∈ U. The flow puts')
        print(f'[wit]   mass on U via extrapolation beyond the reach set.')
    elif amls_detected and apgd_detected:
        print('[wit] AMLS witness is REAL: APGD also found a counterexample.')
        print('[wit]   The flow correctly identified a true SAT direction.')
    elif not amls_detected and apgd_detected:
        print('[wit] AMLS missed a real counterexample APGD found.')
    else:
        print('[wit] Both AMLS and APGD agree: no detection.')


if __name__ == '__main__':
    main()
