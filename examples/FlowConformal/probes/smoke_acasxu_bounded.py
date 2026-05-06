"""Focused 10-instance ACAS Xu smoke for amls_bounded vs amls.

CRITICAL: trains the flow + calibrates q ONCE per instance, then
runs both verification methods on the SAME flow + same q. This
isolates the verification-method-only difference; flow training
and calibration variability is not in the comparison.

Tests the exact 4 Phase 5b false-UNSAT instances (αβ-CROWN-confirmed sat)
plus 6 αβ-CROWN-unsat controls.

Pass criteria:
    amls_bounded gt=sat   → must NOT be UNSAT (UNKNOWN/SAT both fine)
    amls_bounded gt=unsat → ideally UNSAT (matches Phase 5d)
    amls           gt=sat → expected NOT UNSAT (Phase 5d eliminates these)
    amls           gt=unsat → expected UNSAT
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PROJ_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_PROJ_ROOT))

INSTANCES = [
    # 4 αβ-CROWN-sat instances that Phase 5b incorrectly UNSATed
    ('ACASXU_run2a_1_2_batch_2000.onnx', 'prop_2.vnnlib', 'sat'),
    ('ACASXU_run2a_1_5_batch_2000.onnx', 'prop_2.vnnlib', 'sat'),
    ('ACASXU_run2a_1_6_batch_2000.onnx', 'prop_2.vnnlib', 'sat'),
    ('ACASXU_run2a_1_9_batch_2000.onnx', 'prop_7.vnnlib', 'sat'),
    # 6 αβ-CROWN-unsat controls
    ('ACASXU_run2a_1_1_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
    ('ACASXU_run2a_1_2_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
    ('ACASXU_run2a_1_3_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
    ('ACASXU_run2a_1_4_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
    ('ACASXU_run2a_1_5_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
    ('ACASXU_run2a_1_6_batch_2000.onnx', 'prop_1.vnnlib', 'unsat'),
]

BENCH_ROOT = Path(os.path.expanduser(
    '~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023'
))


def whiten_halfspace(hs, mu, sigma):
    """Whiten a HalfSpace G y <= g into G_w y_w <= g_w under
    y_w = (y - mu) / sigma.
    """
    from n2v.sets.halfspace import HalfSpace
    G = np.asarray(hs.G, dtype=np.float64)
    g = np.asarray(hs.g, dtype=np.float64).flatten()
    G_w = (G * sigma).astype(np.float64)
    g_w = (g - (G @ mu).astype(np.float64)).astype(np.float64)
    return HalfSpace(G_w, g_w.reshape(-1, 1))


def normalize_spec_for_amls(spec, mu, sigma):
    """Pull out an AND-of-OR-of-AND list[list[HalfSpace]] in whitened
    coords from the loader's spec object.
    """
    from n2v.sets.halfspace import HalfSpace
    if isinstance(spec, list):
        groups = []
        for g in spec:
            if isinstance(g, dict):
                hs = g.get('Hg')
                if isinstance(hs, list):
                    groups.append([whiten_halfspace(h, mu, sigma) for h in hs])
                else:
                    groups.append([whiten_halfspace(hs, mu, sigma)])
            elif isinstance(g, HalfSpace):
                groups.append([whiten_halfspace(g, mu, sigma)])
        return groups
    if isinstance(spec, HalfSpace):
        return [[whiten_halfspace(spec, mu, sigma)]]
    raise ValueError(f'unsupported spec type: {type(spec)}')


def main():
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._common import (
        load_instance,
    )
    from n2v.probabilistic.verify_flow import _train_flow
    from n2v.probabilistic.flow.scores import FlowScore
    from n2v.probabilistic.flow.calibrate import calibrate
    from n2v.probabilistic.flow.amls import amls_certify_spec
    from n2v.probabilistic.flow.amls_bounded import (
        amls_bounded_certify_spec,
    )

    out_csv = (Path(__file__).parent / 'outputs'
               / 'smoke_acasxu_bounded.csv')
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_csv, 'w', newline='')
    fields = [
        'onnx', 'vnnlib', 'gt',
        'q', 'flow_train_loss_final', 'flow_train_s',
        'method', 'verdict', 'detected_unsafe', 'levels_used',
        'pi_hat', 'pi_upper', 'wall_s', 'soundness_flag',
    ]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    f.flush()

    print(f'[smoke] {len(INSTANCES)} instances; training flow ONCE per '
          f'instance, running both AMLS methods on same flow + q')
    print(f'[smoke] writing to {out_csv}')
    print()

    n_train = 10000
    m_calib = 8000
    flow_epochs = 2000
    alpha = 0.001
    n_amls = 2000

    t_start = time.time()
    for inst_idx, (onnx_name, vnn_name, gt) in enumerate(INSTANCES):
        seed = hash((onnx_name, vnn_name)) & 0x7FFFFFFF
        torch.manual_seed(seed)
        np.random.seed(seed)

        try:
            net, boxes, spec = load_instance(
                BENCH_ROOT, f'onnx/{onnx_name}', f'vnnlib/{vnn_name}',
            )
        except Exception as e:
            print(f'  [{onnx_name}+{vnn_name}] LOAD FAILED: {e}')
            continue
        lb, ub = boxes[0]
        lb_t = torch.tensor(np.asarray(lb).flatten(), dtype=torch.float32)
        ub_t = torch.tensor(np.asarray(ub).flatten(), dtype=torch.float32)

        # --- 1. Train flow ONCE ---
        torch.manual_seed(seed)
        x_tr = lb_t + torch.rand(n_train, lb_t.shape[0]) * (ub_t - lb_t)
        x_ca = lb_t + torch.rand(m_calib, lb_t.shape[0]) * (ub_t - lb_t)
        with torch.no_grad():
            y_tr = net(x_tr).detach()
            y_ca = net(x_ca).detach()
        y_mean = y_tr.mean(dim=0)
        y_std = y_tr.std(dim=0).clamp_min(1e-8)
        y_tr_w = (y_tr - y_mean) / y_std
        y_ca_w = (y_ca - y_mean) / y_std

        t_train = time.time()
        flow, losses = _train_flow(
            y_tr_w, dim=y_tr_w.shape[1],
            n_epochs=flow_epochs, seed=seed,
            return_losses=True,
        )
        flow = flow.to('cpu').eval()
        train_s = time.time() - t_train

        # --- 2. Calibrate q ONCE ---
        score_fn = FlowScore(flow, t=1.0, n_steps=30, method='rk4',
                             batch_size=65536)
        with torch.no_grad():
            calib_scores = score_fn(y_ca_w)
        ell = max(1, min(m_calib,
                          int(math.ceil((m_calib + 1) * (1 - alpha)))))
        q = calibrate(calib_scores, ell).item()

        # --- 3. Whiten the spec for AMLS ---
        whitened_groups = normalize_spec_for_amls(
            spec, y_mean.numpy(), y_std.numpy(),
        )

        loss0 = float(losses[0]) if losses else float('nan')
        lossN = float(losses[-1]) if losses else float('nan')
        print(f'[smoke {inst_idx+1:>2}/{len(INSTANCES)}] '
              f'{onnx_name}+{vnn_name} (gt={gt})  '
              f'q={q:.4f}  loss {loss0:.4f}->{lossN:.4f}  '
              f'train={train_s:.1f}s', flush=True)

        # --- 4a. Run amls (unbounded) ---
        t0 = time.time()
        amls_r = amls_certify_spec(
            flow_ode=flow, spec_groups=whitened_groups,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, beta=0.001,
            seed=seed,
        )
        amls_wall = time.time() - t0
        first_amls = amls_r.per_hs_results[0][0]
        amls_verdict = ('UNSAT' if amls_r.unsat_certified
                          else 'UNKNOWN')

        # --- 4b. Run amls_bounded ---
        t0 = time.time()
        amls_b_r = amls_bounded_certify_spec(
            flow_ode=flow, spec_groups=whitened_groups,
            q=q, eps_2_target=alpha,
            n_samples_per_level=n_amls, quantile=0.1,
            n_mcmc_steps=10, mcmc_step_size=0.3,
            adaptive_step=False, beta=0.001,
            seed=seed,
        )
        amls_b_wall = time.time() - t0
        first_amls_b = amls_b_r.per_hs_results[0][0]
        amls_b_verdict = ('UNSAT' if amls_b_r.unsat_certified
                          else 'UNKNOWN')

        def soundness(verdict, gt):
            if verdict == 'UNSAT' and gt == 'sat':
                return 'FALSE_UNSAT'
            if verdict == 'UNSAT' and gt == 'unsat':
                return 'ok'
            return 'unk'

        for method, verdict, first, wall in [
            ('amls', amls_verdict, first_amls, amls_wall),
            ('amls_bounded', amls_b_verdict, first_amls_b, amls_b_wall),
        ]:
            flag = soundness(verdict, gt)
            row = {
                'onnx': onnx_name,
                'vnnlib': vnn_name,
                'gt': gt,
                'q': f'{q:.4f}',
                'flow_train_loss_final': f'{lossN:.6f}',
                'flow_train_s': f'{train_s:.1f}',
                'method': method,
                'verdict': verdict,
                'detected_unsafe': str(first.detected_unsafe),
                'levels_used': str(first.levels_used),
                'pi_hat': f'{first.pi_hat:.3e}',
                'pi_upper': f'{first.pi_upper:.3e}',
                'wall_s': f'{wall:.2f}',
                'soundness_flag': flag,
            }
            w.writerow(row)
            f.flush()
            print(f'    {method:14s}  verdict={verdict:8s}  flag={flag:11s}  '
                  f'detected={str(first.detected_unsafe):5s}  '
                  f'levels={first.levels_used:>3d}  '
                  f'pi_hat={first.pi_hat:.2e}  pi_upper={first.pi_upper:.2e}  '
                  f'wall={wall:.2f}s', flush=True)

    f.close()
    print(f'\n[smoke] done in {(time.time()-t_start)/60:.1f} min')
    print(f'[smoke] wrote {out_csv}')

    print()
    print('=== SUMMARY ===')
    rows = list(csv.DictReader(open(out_csv)))
    for method in ('amls', 'amls_bounded'):
        method_rows = [r for r in rows if r['method'] == method]
        n_false = sum(1 for r in method_rows
                       if r['soundness_flag'] == 'FALSE_UNSAT')
        n_ok = sum(1 for r in method_rows if r['soundness_flag'] == 'ok')
        n_unk = sum(1 for r in method_rows if r['soundness_flag'] == 'unk')
        gt_sat = sum(1 for r in method_rows if r['gt'] == 'sat')
        gt_unsat = sum(1 for r in method_rows if r['gt'] == 'unsat')
        unk_on_sat = sum(1 for r in method_rows
                         if r['gt'] == 'sat' and r['soundness_flag'] == 'unk')
        print(f'  {method:14s}: gt-sat ({gt_sat}): '
              f'FALSE_UNSAT={n_false}, abstained={unk_on_sat}; '
              f'gt-unsat ({gt_unsat}): UNSAT={n_ok}, UNKNOWN={n_unk - unk_on_sat}')


if __name__ == '__main__':
    main()
