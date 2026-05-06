"""Hyperparameter sweep probe for ACAS Xu false UNSATs.

Tries multiple (n_train, flow_epochs, scenario_n_samples) configs on
the 5 ACAS Xu false UNSAT instances (all prop_2 on networks 1_1, 1_3,
1_7, 1_8, 1_9) and reports which combinations flip the verdict away
from false UNSAT.

We already know one config that worked: n_train=20K, flow_epochs=5K
flipped 1_3 prop_2 from UNSAT to UNKNOWN with detected=True. The
question: do other configs also work, and is there a config that
flips ALL 5 false UNSATs?

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.probes.hparam_sweep_probe
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    PER_BENCHMARK_CONFIG, list_instances, load_one_instance,
)
from n2v.probabilistic.verify_flow import run_verification_pipeline


# 5 ACAS Xu false UNSAT instances (idx in the canonical instances list)
_FN_INSTANCES = [
    (45, 'ACASXU_run2a_1_1', 'prop_2'),
    (47, 'ACASXU_run2a_1_3', 'prop_2'),
    (51, 'ACASXU_run2a_1_7', 'prop_2'),
    (52, 'ACASXU_run2a_1_8', 'prop_2'),
    (53, 'ACASXU_run2a_1_9', 'prop_2'),
]


# Hyperparameter configs to try
_CONFIGS = [
    # name, n_train, flow_epochs, scenario_n_samples, mcmc_step_size
    ('baseline',    10_000, 2_000,   2_000, 0.3),
    ('bumped_2x',   20_000, 5_000,   2_000, 0.3),  # known to work on idx 47
    ('big_n_train', 50_000, 5_000,   4_000, 0.3),
    ('long_train',  10_000, 20_000,  2_000, 0.3),
    ('small_step',  10_000, 5_000,   4_000, 0.05),
    ('big_amls_N',  10_000, 2_000,   8_000, 0.3),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--instances', type=int, nargs='+', default=None,
                   help='Instance idx subset (default: all 5).')
    p.add_argument('--configs', type=str, nargs='+', default=None,
                   help='Config name subset (default: all).')
    p.add_argument('--seed', type=int, default=47)
    p.add_argument('--output-csv', type=Path,
                   default=Path('examples/FlowConformal/probes/outputs/'
                                'hparam_sweep_acasxu_fns.csv'))
    args = p.parse_args()

    target_insts = (args.instances if args.instances is not None
                    else [idx for idx, _, _ in _FN_INSTANCES])
    target_configs = [c for c in _CONFIGS
                      if args.configs is None or c[0] in args.configs]
    print(f'[sweep] targets: {len(target_insts)} instances × '
          f'{len(target_configs)} configs = {len(target_insts) * len(target_configs)} runs',
          flush=True)

    instances = list_instances('acasxu_2023')
    rows = []

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output_csv, 'w', newline='')
    fieldnames = ['instance_idx', 'instance_name', 'config_name',
                  'n_train', 'flow_epochs', 'scenario_n_samples',
                  'mcmc_step_size', 'verdict', 'eps_2', 'detected_unsafe',
                  'levels_used', 'wall_s', 'flipped_to']
    writer = csv.DictWriter(out_f, fieldnames=fieldnames)
    writer.writeheader(); out_f.flush()

    cfg_default = PER_BENCHMARK_CONFIG['acasxu_2023']

    for inst_idx in target_insts:
        onnx_rel, vnn_rel, vnncomp_t = instances[inst_idx]
        print(f'\n[sweep] === instance idx={inst_idx}: {onnx_rel} {vnn_rel} ===',
              flush=True)
        for (name, n_train, flow_epochs, scen_n, mcmc_step) in target_configs:
            print(f'  [{name}] n_train={n_train} epochs={flow_epochs} '
                  f'scen_n={scen_n} step={mcmc_step}', flush=True)

            # Load fresh per run (network state irrelevant for our flow)
            network, boxes, spec = load_one_instance('acasxu_2023', onnx_rel, vnn_rel)
            if torch.cuda.is_available():
                network = network.cuda()
            lb, ub = boxes[0]

            torch.manual_seed(args.seed); np.random.seed(args.seed)
            t0 = time.time()
            try:
                result = run_verification_pipeline(
                    network=network, input_lb=lb, input_ub=ub, spec=spec,
                    alpha=cfg_default['alpha'],
                    n_train=n_train,
                    flow_epochs=flow_epochs,
                    flow_config=cfg_default['flow_config'],
                    scenario_n_samples=scen_n,
                    scenario_beta=0.001,
                    verification_method=cfg_default['verification_method'],
                    amls_max_levels=cfg_default['amls_max_levels'],
                    amls_n_samples_per_level=scen_n,
                    amls_mcmc_step_size=mcmc_step,
                    seed=args.seed,
                    use_falsifier=False,
                )
                wall = time.time() - t0
                verdict = result['verdict']
                eps_2 = result.get('amls_bounded_eps_2_upper', '')
                det = result.get('amls_bounded_detected_unsafe', '')
                lev = result.get('amls_levels_used', '')
                flipped = ('FIXED' if verdict != 'UNSAT' else 'still UNSAT (FN)')
                print(f'    → verdict={verdict}  eps_2={eps_2}  '
                      f'detected={det}  levels={lev}  wall={wall:.1f}s  [{flipped}]',
                      flush=True)
            except Exception as e:
                wall = time.time() - t0
                verdict = 'ERROR'
                eps_2 = ''
                det = ''
                lev = ''
                flipped = f'ERROR: {type(e).__name__}'
                print(f'    → ERROR: {type(e).__name__}: {e}', flush=True)

            row = {
                'instance_idx': inst_idx,
                'instance_name': f'{onnx_rel}|{vnn_rel}',
                'config_name': name,
                'n_train': n_train, 'flow_epochs': flow_epochs,
                'scenario_n_samples': scen_n, 'mcmc_step_size': mcmc_step,
                'verdict': verdict, 'eps_2': eps_2,
                'detected_unsafe': det, 'levels_used': lev,
                'wall_s': f'{wall:.1f}', 'flipped_to': flipped,
            }
            writer.writerow(row); out_f.flush()
            rows.append(row)

    out_f.close()
    print(f'\n[sweep] wrote {args.output_csv}')
    print()
    print('=== Summary: which configs flip which instances ===')
    for inst_idx in target_insts:
        print(f'\nidx={inst_idx}:')
        for r in rows:
            if r['instance_idx'] != inst_idx: continue
            mark = '✓' if r['verdict'] != 'UNSAT' and r['verdict'] != 'ERROR' else '✗'
            print(f'  {mark} {r["config_name"]:15s} → {r["verdict"]:>9}  '
                  f'eps_2={r["eps_2"][:10] if r["eps_2"] else "?"}  '
                  f'det={r["detected_unsafe"]}  wall={r["wall_s"]}s')


if __name__ == '__main__':
    main()
