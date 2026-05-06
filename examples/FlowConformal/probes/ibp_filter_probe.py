"""IBP post-hoc filter probe (read-only diagnostic).

Tests whether a sound IBP over-approximation of the network's output
range can be used as a post-hoc filter on existing UNSAT verdicts:

    For each UNSAT verdict from our existing CSV:
      1. Compute IBP output bounds for the input box.
      2. If U ∩ output_box != ∅ (sound over-approx says U reachable),
         flip the verdict to UNKNOWN.
      3. Otherwise leave UNSAT.

Prediction: this should flip the false UNSATs (where U is genuinely
reachable per ground truth) to UNKNOWN while leaving most true UNSATs
intact (where U is provably unreachable). The cost: any true UNSATs
where IBP is too loose to rule out U intersection get demoted to
UNKNOWN — these are the "lost" certifications.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.ibp_filter_probe \\
        --benchmark acasxu_2023
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    list_instances, load_one_instance,
)
from examples.FlowConformal.experiments.baselines._common import (
    halfspace_disjoint_from_box,
)
from n2v.probabilistic.flow.sampling import sample_box as _sample_box
from n2v.probabilistic.verify_flow import _forward


def _output_bounds(network, lb, ub, n_samples: int = 8000,
                   inflation_factor: float = 3.0, seed: int = 47):
    """Empirical output box from random samples + per-dim std inflation.

    Not strictly sound, but mirrors what Hashemi-clipping does in spirit:
    sample uniformly from input box, push through network, take
    per-dim (min, max) of outputs, and inflate by a multiple of the
    per-dim sample std to cover the conformal tail.

    Returns ``(lb_y, ub_y)`` numpy arrays.
    """
    lb_t = torch.as_tensor(np.asarray(lb), dtype=torch.float32)
    ub_t = torch.as_tensor(np.asarray(ub), dtype=torch.float32)
    x = _sample_box(lb_t, ub_t, n_samples=n_samples, seed=seed)
    y = _forward(network, x).detach().cpu().numpy()  # (n_samples, d)
    lb_y = y.min(axis=0)
    ub_y = y.max(axis=0)
    if inflation_factor > 0:
        std = y.std(axis=0)
        lb_y = lb_y - inflation_factor * std
        ub_y = ub_y + inflation_factor * std
    return lb_y, ub_y


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--inflation', type=float, default=3.0,
                   help='Per-dim std inflation factor (default 3.0).')
    p.add_argument('--n-samples', type=int, default=8000,
                   help='Random samples for empirical box (default 8000).')
    p.add_argument('--limit', type=int, default=None,
                   help='Run only the first N instances (default: all).')
    p.add_argument('--gt-csv',
                   default='examples/FlowConformal/experiments/exp1_vnncomp_subset/ground_truth.csv')
    p.add_argument('--ours-csv', default=None,
                   help='Path to ours CSV (default: ./outputs/exp1_<bench>_ours.csv).')
    args = p.parse_args()

    if args.ours_csv is None:
        args.ours_csv = (
            'examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/'
            f'exp1_{args.benchmark}_ours.csv'
        )

    gt = {(r['benchmark'], r['vnnlib_file']): r['ground_truth']
          for r in csv.DictReader(open(args.gt_csv))}
    ours_rows = list(csv.DictReader(open(args.ours_csv)))
    print(f'[ibp] {args.benchmark}: {len(ours_rows)} existing rows', flush=True)

    instances = list_instances(args.benchmark)
    # Key by (onnx, vnnlib) — acasxu has multiple onnx networks paired
    # with the same vnnlib (prop_2 etc.), so vnnlib alone is ambiguous.
    inst_by_pair = {(o, v): (o, v, t) for (o, v, t) in instances}

    transitions = Counter()  # (orig, new, gt) -> count
    by_verdict_gt_count = Counter()
    n_processed = 0
    n_skipped = 0
    t_total = 0.0

    for r in ours_rows:
        v = r.get('vnnlib_file', '')
        orig = r['verdict']
        g = gt.get((args.benchmark, v), '?')
        by_verdict_gt_count[(orig, g)] += 1

        # Only re-evaluate UNSAT verdicts (the ones that could be FN)
        if orig != 'UNSAT':
            transitions[(orig, orig, g)] += 1
            continue

        onnx_rel = r.get('onnx_file', '')
        # The CSV's onnx_file is just the basename; instances.csv may
        # use a relative path. Look up by the (onnx, vnn) pair if both
        # are stored, else fall back to vnn-only match.
        pair_key = (onnx_rel, v)
        if pair_key not in inst_by_pair:
            # Try vnn-suffix match (vnn_rel might be 'vnnlib/foo' vs 'foo')
            pair_key = next(
                ((o, vr) for (o, vr) in inst_by_pair
                 if vr.endswith(v) and o.endswith(onnx_rel)), None)
        if pair_key is None:
            n_skipped += 1
            transitions[(orig, orig, g)] += 1
            continue

        if args.limit is not None and n_processed >= args.limit:
            transitions[(orig, orig, g)] += 1
            continue

        try:
            o_resolved, v_resolved, _ = inst_by_pair[pair_key]
            network, boxes, spec = load_one_instance(args.benchmark,
                                                     o_resolved, v_resolved)
        except Exception as e:
            print(f'  [ibp] LOAD FAIL {v}: {e}', flush=True)
            n_skipped += 1
            transitions[(orig, orig, g)] += 1
            continue

        try:
            lb, ub = boxes[0]
            t0 = time.time()
            lb_y, ub_y = _output_bounds(network, lb, ub,
                                         n_samples=args.n_samples,
                                         inflation_factor=args.inflation)
            t_total += time.time() - t0
            disjoint = halfspace_disjoint_from_box(spec, lb_y, ub_y)
        except Exception as e:
            print(f'  [ibp] IBP FAIL {v}: {type(e).__name__}: {e}', flush=True)
            n_skipped += 1
            transitions[(orig, orig, g)] += 1
            continue

        n_processed += 1

        if disjoint is True:
            new = 'UNSAT'  # sound check confirms safety
        elif disjoint is False:
            new = 'UNKNOWN'  # IBP says U is reachable in over-approx — abstain
        else:
            new = orig  # spec structure unrecognised — keep original

        transitions[(orig, new, g)] += 1
        if n_processed % 25 == 0:
            print(f'  [ibp] processed {n_processed} UNSAT verdicts '
                  f'(avg {t_total/n_processed*1000:.1f}ms/inst)', flush=True)

    # ---- Report ----
    print()
    print(f'[ibp] processed {n_processed} UNSAT verdicts; skipped {n_skipped}')
    print(f'[ibp] avg IBP wall: {t_total/max(n_processed,1)*1000:.1f}ms per instance')
    print()
    print('=== Transitions (orig → new, partitioned by ground truth) ===')
    print(f'{"orig":>9} → {"new":>9}  {"gt":>5}  {"count":>6}')
    for (orig, new, g), c in sorted(transitions.items()):
        marker = ''
        if orig == 'UNSAT' and g == 'sat':
            if new == 'UNKNOWN':
                marker = ' ✓ (FN → UNK, fixed!)'
            elif new == 'UNSAT':
                marker = ' ✗ (FN persists, IBP didn\'t catch)'
        elif orig == 'UNSAT' and g == 'unsat' and new == 'UNKNOWN':
            marker = ' ⚠ (TN → UNK, lost certification)'
        print(f'{orig:>9} → {new:>9}  {g:>5}  {c:>6}{marker}')

    # ---- Bottom-line summary ----
    print()
    print('=== Bottom line ===')
    fn_orig = sum(c for (o, _, g), c in transitions.items()
                  if o == 'UNSAT' and g == 'sat')
    fn_caught = sum(c for (o, n, g), c in transitions.items()
                    if o == 'UNSAT' and n == 'UNKNOWN' and g == 'sat')
    tn_orig = sum(c for (o, _, g), c in transitions.items()
                  if o == 'UNSAT' and g == 'unsat')
    tn_lost = sum(c for (o, n, g), c in transitions.items()
                  if o == 'UNSAT' and n == 'UNKNOWN' and g == 'unsat')
    tn_kept = tn_orig - tn_lost
    print(f'[ibp] FN before: {fn_orig}  →  caught by IBP: {fn_caught}/{fn_orig}')
    print(f'[ibp] TN before: {tn_orig}  →  preserved: {tn_kept}/{tn_orig}  '
          f'(lost: {tn_lost})')
    if tn_orig > 0:
        print(f'[ibp] TN preservation: {100*tn_kept/tn_orig:.1f}%')
    if fn_orig > 0:
        print(f'[ibp] FN elimination:  {100*fn_caught/fn_orig:.1f}%')


if __name__ == '__main__':
    main()
