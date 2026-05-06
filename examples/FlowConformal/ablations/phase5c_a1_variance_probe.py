"""A1 diagnostic probe: decompose q-variance into flow / cal / scenario.

8 instances x 3 trial sets x 5 trials = 120 runs. Per trial: record q,
worst_max_margin, verdict, flow_train_loss_final.

Trial sets:
  F: vary flow_seed in {1..5}, cal_seed=0, scenario_seed=0
  C: vary cal_seed in {1..5},  flow_seed=0, scenario_seed=0
  S: vary scenario_seed in {1..5}, flow_seed=0, cal_seed=0

Output: outputs/phase5c_a1_variance_probe.csv

Usage:
    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.ablations.phase5c_a1_variance_probe \\
        > /tmp/phase5c_a1.log 2>&1
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from examples.FlowConformal.benchmarks._common import run_verification_pipeline
from examples.FlowConformal.benchmarks.test_acasxu_single import (
    _ACASXuWrapper, _extract_spec,
)
from n2v.utils import load_vnnlib
from n2v.utils.model_loader import load_onnx


_ACASXU_ROOT = Path(__file__).resolve().parents[2] / 'ACASXu'
_OUT_DIR = Path(__file__).parent / 'outputs'
_OUT_CSV = _OUT_DIR / 'phase5c_a1_variance_probe.csv'

# Mirror acasxu_sweep.py exactly.
_FLOW_CONFIG = 'base'
_N_TRAIN = 5_000
_FLOW_EPOCHS = 2_000
_SCENARIO_N = 2_000
_ALPHA = 0.001

_TRIALS_PER_SET = 5

_INSTANCES = [
    # (onnx_rel, vnn_rel, label)
    ('onnx/ACASXU_run2a_1_2_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'calib_miss'),
    ('onnx/ACASXU_run2a_1_5_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'samp_miss_persistent'),
    ('onnx/ACASXU_run2a_1_6_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'regression'),
    ('onnx/ACASXU_run2a_1_9_batch_2000.onnx', 'vnnlib/prop_7.vnnlib', 'samp_miss_persistent'),
    ('onnx/ACASXU_run2a_1_4_batch_2000.onnx', 'vnnlib/prop_2.vnnlib', 'closed_to_unknown'),
    ('onnx/ACASXU_run2a_2_9_batch_2000.onnx', 'vnnlib/prop_8.vnnlib', 'unknown_to_sat'),
    ('onnx/ACASXU_run2a_1_1_batch_2000.onnx', 'vnnlib/prop_1.vnnlib', 'control_typical_unsat'),
    ('onnx/ACASXU_run2a_4_5_batch_2000.onnx', 'vnnlib/prop_3.vnnlib', 'control_other_prop'),
]

# Smoke flag: if True, reduce instances to 1 and trials per set to 1.
_SMOKE = False


def _load_instance(onnx_rel: str, vnn_rel: str):
    onnx_path = _ACASXU_ROOT / onnx_rel.removeprefix('./')
    vnn_path = _ACASXU_ROOT / vnn_rel.removeprefix('./')
    network = _ACASXuWrapper(load_onnx(str(onnx_path)).eval())
    prop = load_vnnlib(str(vnn_path))
    if isinstance(prop['lb'], list) or isinstance(prop['ub'], list):
        raise ValueError('OR-of-input-regions not supported')
    input_lb = np.asarray(prop['lb']).flatten()
    input_ub = np.asarray(prop['ub']).flatten()
    spec = _extract_spec(prop['prop'])
    return network, input_lb, input_ub, spec


def _extract_worst_max_margin(result: dict) -> 'float | None':
    """Pull the worst (min) max-row-margin across all per-group / per-hs
    results in the scenario_result. Positive means certified disjoint;
    negative means a flow sample fell inside the unsafe region.

    Returns None when the scenario step did not run (e.g. SAT verdict
    short-circuits before scenario-verify) or the structure is empty.
    """
    sr = result.get('scenario_result')
    if sr is None:
        return None
    per_group = sr.get('per_group_results') or []
    margins = []
    for group_res in per_group:
        per_hs = getattr(group_res, 'per_hs_results', None) or []
        for hs_res in per_hs:
            wm = getattr(hs_res, 'worst_max_margin', None)
            if wm is not None:
                margins.append(float(wm))
    if not margins:
        return None
    return min(margins)


def _run_trial(onnx_rel: str, vnn_rel: str, *,
               flow_seed: int, cal_seed: int, scenario_seed: int) -> dict:
    """Run one trial. Returns a partial row dict (no CSV-frame fields)."""
    try:
        network, lb, ub, spec = _load_instance(onnx_rel, vnn_rel)
    except (ValueError, NotImplementedError) as e:
        return {'verdict': 'ERROR', 'error': f'load skip {type(e).__name__}: {e}'}
    except Exception as e:
        return {'verdict': 'ERROR', 'error': f'loadfailed {type(e).__name__}: {e}'}

    try:
        result = run_verification_pipeline(
            network=network,
            input_lb=lb, input_ub=ub, spec=spec,
            alpha=_ALPHA,
            n_train=_N_TRAIN, flow_epochs=_FLOW_EPOCHS,
            flow_config=_FLOW_CONFIG,
            scenario_n_samples=_SCENARIO_N, scenario_beta=0.001,
            seed=0,
            flow_seed=flow_seed, cal_seed=cal_seed, scenario_seed=scenario_seed,
        )
    except Exception as e:
        return {'verdict': 'ERROR',
                'error': f'runfailed {type(e).__name__}: {e}'}

    return {
        'verdict': result['verdict'],
        'q': result.get('q'),
        'worst_max_margin': _extract_worst_max_margin(result),
        'flow_train_loss_final': result.get('flow_train_loss_final'),
        'error': '',
    }


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def main():
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    instances = _INSTANCES
    trials_per_set = _TRIALS_PER_SET
    if _SMOKE:
        instances = _INSTANCES[:1]
        trials_per_set = 1

    fields = [
        'instance', 'label', 'trial_set', 'trial_idx',
        'flow_seed', 'cal_seed', 'scenario_seed',
        'q', 'worst_max_margin', 'verdict',
        'flow_train_loss_final', 'wall_s', 'error',
    ]

    n_total = len(instances) * 3 * trials_per_set
    print(f'Phase 5c A1 variance probe', flush=True)
    print(f'  instances={len(instances)}  trial_sets=3  '
          f'trials_per_set={trials_per_set}  total_runs={n_total}', flush=True)
    print(f'  flow_config={_FLOW_CONFIG}  n_train={_N_TRAIN}  '
          f'flow_epochs={_FLOW_EPOCHS}  scenario_n={_SCENARIO_N}  '
          f'alpha={_ALPHA}', flush=True)

    t_start = time.time()
    run_idx = 0

    with open(_OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        f.flush()

        for onnx_rel, vnn_rel, label in instances:
            inst_name = f'{Path(onnx_rel).name}+{Path(vnn_rel).name}'
            for trial_set in ('F', 'C', 'S'):
                for trial_idx in range(1, trials_per_set + 1):
                    run_idx += 1
                    if trial_set == 'F':
                        flow_seed, cal_seed, scenario_seed = trial_idx, 0, 0
                    elif trial_set == 'C':
                        flow_seed, cal_seed, scenario_seed = 0, trial_idx, 0
                    else:  # 'S'
                        flow_seed, cal_seed, scenario_seed = 0, 0, trial_idx

                    elapsed = time.time() - t_start
                    print(f'[{run_idx}/{n_total}  t={elapsed:.0f}s] '
                          f'{inst_name}  set={trial_set}  trial={trial_idx}  '
                          f'(flow={flow_seed},cal={cal_seed},scen={scenario_seed})',
                          flush=True)
                    t0 = time.time()
                    row = _run_trial(
                        onnx_rel, vnn_rel,
                        flow_seed=flow_seed, cal_seed=cal_seed,
                        scenario_seed=scenario_seed,
                    )
                    wall_s = time.time() - t0

                    out_row = {_f: '' for _f in fields}
                    out_row['instance'] = inst_name
                    out_row['label'] = label
                    out_row['trial_set'] = trial_set
                    out_row['trial_idx'] = trial_idx
                    out_row['flow_seed'] = flow_seed
                    out_row['cal_seed'] = cal_seed
                    out_row['scenario_seed'] = scenario_seed
                    out_row['verdict'] = row['verdict']
                    out_row['wall_s'] = f'{wall_s:.1f}'
                    if row['verdict'] == 'ERROR':
                        out_row['error'] = row.get('error', '')
                    else:
                        out_row['q'] = _fmt(row.get('q'), '.6f')
                        out_row['worst_max_margin'] = _fmt(
                            row.get('worst_max_margin'), '.6f')
                        out_row['flow_train_loss_final'] = _fmt(
                            row.get('flow_train_loss_final'), '.6f')
                        out_row['error'] = row.get('error', '')

                    writer.writerow(out_row)
                    f.flush()
                    print(f'    verdict={out_row["verdict"]}  '
                          f'q={out_row["q"]}  '
                          f'wmm={out_row["worst_max_margin"]}  '
                          f'wall={wall_s:.1f}s', flush=True)

    print(f'\n=== Probe complete ===')
    print(f'Wrote {_OUT_CSV}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
