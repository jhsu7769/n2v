"""Counterexample-in-flow-set probe (read-only diagnostic).

For an instance where our framework reports false UNSAT but a sound
verifier (αβ-CROWN, etc.) found a counterexample, this probe loads
the αβ-CROWN counterexample (input + output), trains our flow as
usual, and asks: is the counterexample's network output INSIDE our
flow's calibrated q-ball or OUTSIDE?

Three outcomes possible:

  1. INSIDE the q-ball: the cex is in our calibrated reach set, so
     AMLS *should* have found it as a witness. Failure to find it is
     an AMLS exploration failure (stuck in wrong region of the ball).

  2. OUTSIDE the q-ball: the cex is OUTSIDE our calibrated reach set,
     meaning the conformal coverage is not capturing this point. The
     cex is one of the (at most α) "missed" outputs the conformal
     guarantee allows. This is a flow-distribution-vs-true-distribution
     mismatch, not an AMLS failure.

  3. The cex's actual network(x) doesn't match the cex's recorded y:
     onnx2torch loading discrepancy. We compute network(x_cex) ourselves.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.cex_in_flow_probe \\
        --benchmark acasxu_2023 --instance-idx 47 \\
        --cex-file ~/v/other/VNNCOMP/vnncomp2025_results/alpha_beta_crown/2025_acasxu_2023/ACASXU_run2a_1_3_batch_2000_prop_2.counterexample.gz
"""
from __future__ import annotations

import argparse
import gzip
import re
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG, list_instances, load_one_instance,
)
from n2v.probabilistic.verify_flow import _forward, run_verification_pipeline


def parse_cex_file(path: Path):
    """Parse VNN-COMP-style counterexample file:

      ((X_0  val) (X_1  val) ... (Y_0  val) ...)

    Returns ``(x_arr, y_arr)`` numpy arrays.
    """
    text = (gzip.open(path, 'rt').read() if str(path).endswith('.gz')
            else open(path).read())
    pat = re.compile(r'\(\s*(X|Y)_(\d+)\s+(-?[\d.eE+-]+)\s*\)')
    xs, ys = {}, {}
    for kind, idx, val in pat.findall(text):
        i = int(idx); v = float(val)
        (xs if kind == 'X' else ys)[i] = v
    if not xs:
        raise ValueError(f'No X variables found in {path}')
    if not ys:
        raise ValueError(f'No Y variables found in {path}')
    n_x = max(xs) + 1
    n_y = max(ys) + 1
    x_arr = np.array([xs[i] for i in range(n_x)], dtype=np.float64)
    y_arr = np.array([ys[i] for i in range(n_y)], dtype=np.float64)
    return x_arr, y_arr


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True)
    p.add_argument('--instance-idx', type=int, required=True)
    p.add_argument('--cex-file', required=True, type=Path)
    p.add_argument('--seed', type=int, default=47)
    args = p.parse_args()

    # 1. Parse the cex file
    x_cex, y_cex_recorded = parse_cex_file(args.cex_file)
    print(f'[cex] loaded cex from {args.cex_file.name}')
    print(f'[cex]   x_cex shape={x_cex.shape}  y_cex_recorded shape={y_cex_recorded.shape}')
    print(f'[cex]   x_cex={x_cex}')
    print(f'[cex]   y_cex_recorded={y_cex_recorded}')

    # 2. Load the instance and run our pipeline
    instances = list_instances(args.benchmark)
    onnx_rel, vnn_rel, _t = instances[args.instance_idx]
    print(f'\n[cex] {args.benchmark} idx={args.instance_idx}: '
          f'{onnx_rel} {vnn_rel}', flush=True)

    network, boxes, spec = load_one_instance(args.benchmark, onnx_rel, vnn_rel)
    if torch.cuda.is_available():
        network = network.cuda()
    cfg = PER_BENCHMARK_CONFIG[args.benchmark]
    lb, ub = boxes[0]

    # 3. Verify x_cex is in the input box (sanity check)
    lb_flat = np.asarray(lb).flatten()
    ub_flat = np.asarray(ub).flatten()
    in_box = bool(((x_cex >= lb_flat - 1e-9) & (x_cex <= ub_flat + 1e-9)).all())
    print(f'\n[cex] x_cex in input box? {in_box}')
    if not in_box:
        print(f'[cex]   lb={lb_flat}')
        print(f'[cex]   ub={ub_flat}')

    # 4. Compute y_cex via OUR network forward
    x_t = torch.tensor(x_cex.astype(np.float32)).reshape(1, *lb.shape)
    if torch.cuda.is_available():
        x_t = x_t.cuda()
    with torch.no_grad():
        y_cex_ours = network(x_t).flatten().detach().cpu().numpy().astype(np.float64)
    print(f'[cex] y_cex via our network forward: {y_cex_ours}')
    print(f'[cex] |y_cex_ours - y_cex_recorded|_inf = '
          f'{np.abs(y_cex_ours - y_cex_recorded).max():.6e}')

    # 5. Check phi(y_cex) ≤ 0 (cex is in U)?
    from n2v.utils.verify_specification import _parse_property_groups
    groups = _parse_property_groups(spec)
    per_group = []
    for grp in groups:
        per_hs = []
        for hs in grp:
            G = np.asarray(hs.G, dtype=np.float64)
            g = np.asarray(hs.g, dtype=np.float64).flatten()
            per_hs.append(float((G @ y_cex_ours - g).max()))
        per_group.append(min(per_hs))
    phi_cex = float(max(per_group))
    print(f'[cex] phi(y_cex_ours) = {phi_cex:.6e}  (≤0 means in U)')

    # 6. Run our pipeline to get the trained flow + q
    print(f'\n[cex] running our pipeline...', flush=True)
    t0 = time.time()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
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
    pipe_wall = time.time() - t0
    print(f'[cex] pipeline: verdict={result["verdict"]}  q={result["q"]:.4f}  '
          f'wall={pipe_wall:.1f}s', flush=True)

    # 7. Score the cex y under the trained flow's score function
    score_fn = result['score_fn']
    q = float(result['q'])
    y_cex_t = torch.as_tensor(y_cex_ours.astype(np.float32)).reshape(1, -1)
    with torch.no_grad():
        cex_score = float(score_fn(y_cex_t).item())
    print(f'\n[cex] y_cex score under our flow: {cex_score:.6f}')
    print(f'[cex] q (calibrated radius):       {q:.6f}')

    # 8. Also score m=2000 calibration outputs to give context
    from n2v.probabilistic.flow.sampling import sample_box as _sample_box
    cal_seed = args.seed + 1_000_000
    lb_t_cpu = torch.as_tensor(lb, dtype=torch.float32)
    ub_t_cpu = torch.as_tensor(ub, dtype=torch.float32)
    x_ca = _sample_box(lb_t_cpu, ub_t_cpu, n_samples=2000, seed=cal_seed)
    y_ca = _forward(network, x_ca).detach().cpu()
    with torch.no_grad():
        cal_scores = score_fn(y_ca).detach().cpu().numpy()
    cal_pcts = [50, 95, 99, 99.5, 99.9, 100]
    print()
    print(f'[cex] Calibration score distribution:')
    for pct in cal_pcts:
        v = float(np.percentile(cal_scores, pct))
        print(f'  pct {pct:>5}: {v:>10.4f}')
    rank = int((cal_scores < cex_score).sum())
    pct = 100 * rank / len(cal_scores)
    print(f'[cex] cex score rank: {rank}/{len(cal_scores)} ({pct:.1f}%)')

    # 9. Verdict
    print()
    print('=== VERDICT ===')
    if cex_score <= q:
        print(f'[cex] ✓ cex is INSIDE the q-ball (score {cex_score:.4f} ≤ q={q:.4f})')
        print(f'[cex]   AMLS should be able to find this. Failure to detect is')
        print(f'[cex]   an AMLS exploration / mixing problem, not a coverage one.')
    else:
        print(f'[cex] ✗ cex is OUTSIDE the q-ball (score {cex_score:.4f} > q={q:.4f})')
        print(f'[cex]   cex is in the (≤ α) tail the conformal guarantee allows')
        print(f'[cex]   to miss. AMLS doesn\'t even consider this point because')
        print(f'[cex]   it\'s outside the calibrated set. The flow distribution')
        print(f'[cex]   doesn\'t cover the SAT direction in this instance.')


if __name__ == '__main__':
    main()
