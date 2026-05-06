"""Per-benchmark APGD falsifier budget probe.

For each benchmark with falsifier ON in ``PER_BENCHMARK_CONFIG``, runs
APGD at an escalating ``(n_restarts, n_steps)`` grid against ``K``
SAT-by-VNN-COMP-consensus instances and records ``(sat_found, wall_s)``.
The goal is to pick the smallest budget per benchmark that reliably
finds a counterexample on SAT-consensus instances, without paying the
30x200 cascade cost of the legacy default.

Output: ``probes/outputs/probe_falsifier_budget.csv`` with one row per
``(benchmark, idx, n_restarts, n_steps)``. The aggregator at the
bottom prints a per-benchmark recommendation table.

Usage:

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.probes.probe_falsifier_budget

cifar10_resnet110 has no VNN-COMP consensus (it isn't a VNN-COMP
benchmark) so it inherits the budget chosen for cifar100_2024.
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from n2v.utils.falsify import falsify

_SEED = 47
_OUT = (Path(__file__).resolve().parent / 'outputs'
        / 'probe_falsifier_budget.csv')
_BUDGETS: List[Tuple[int, int]] = [(3, 25), (5, 50), (10, 100), (20, 200)]
_K_SAT_INSTANCES = 3
_FIELDS = [
    'experiment', 'benchmark', 'idx', 'onnx_file', 'vnnlib_file',
    'n_restarts', 'n_steps', 'sat_found', 'wall_s',
]


def _load_ground_truth(gt_path: Path) -> Dict[Tuple[str, str, str], str]:
    out: Dict[Tuple[str, str, str], str] = {}
    if not gt_path.exists():
        return out
    with open(gt_path, newline='') as f:
        for row in csv.DictReader(f):
            bench = row.get('benchmark', '').strip()
            onnx = row.get('onnx_file', '').strip()
            vnn = row.get('vnnlib_file', '').strip()
            gt = row.get('ground_truth', '').strip().lower()
            out[(bench, onnx, vnn)] = gt
    return out


def _exp1_sat_indices(benchmark: str, k: int) -> List[Tuple[int, str, str]]:
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        list_instances,
    )
    inst = list_instances(benchmark)
    gt_path = (Path(__file__).resolve().parents[1]
               / 'experiments' / 'exp1_vnncomp_subset' / 'ground_truth.csv')
    gt = _load_ground_truth(gt_path)
    out = []
    for idx, (onnx_rel, vnn_rel, _t) in enumerate(inst):
        onnx_name = Path(onnx_rel).name
        vnn_name = Path(vnn_rel).name
        if gt.get((benchmark, onnx_name, vnn_name)) == 'sat':
            out.append((idx, onnx_name, vnn_name))
            if len(out) >= k:
                break
    return out


def _exp2_sat_indices(benchmark: str, k: int) -> List[Tuple[int, str, str]]:
    from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
        list_vnncomp_format_instances,
    )
    instances = list_vnncomp_format_instances(benchmark, n=200)
    gt_path = (Path(__file__).resolve().parents[1]
               / 'experiments' / 'exp2_prob_scale' / 'ground_truth.csv')
    gt = _load_ground_truth(gt_path)
    out = []
    for idx, (onnx_path, vnnlib_path, _t) in enumerate(instances):
        onnx_name = Path(onnx_path).name
        vnn_name = Path(vnnlib_path).name
        if gt.get((benchmark, onnx_name, vnn_name)) == 'sat':
            out.append((idx, onnx_name, vnn_name))
            if len(out) >= k:
                break
    return out


def _load_exp1_instance(benchmark: str, onnx_rel: str, vnn_rel: str):
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        load_one_instance,
    )
    return load_one_instance(benchmark, onnx_rel, vnn_rel)


def _load_exp2_instance(benchmark: str, idx: int):
    from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
        list_instances, load_one_instance,
    )
    instances_meta = list_instances(benchmark)
    name, loader, _t = instances_meta[idx]
    return load_one_instance(benchmark, loader)


def _run_one(network, lb, ub, spec, n_restarts: int,
             n_steps: int) -> Tuple[bool, float]:
    """Time a single APGD call with the given budget."""
    t0 = time.time()
    try:
        result, _cex = falsify(
            model=network, lb=np.asarray(lb), ub=np.asarray(ub),
            property=spec, method='apgd', seed=_SEED,
            n_restarts=n_restarts, n_steps=n_steps,
        )
        sat_found = (result == 0)
    except Exception:
        sat_found = False
    return sat_found, time.time() - t0


def _probe_benchmark(experiment: str, benchmark: str,
                     writer: csv.DictWriter,
                     out_file) -> None:
    """Probe APGD budgets on K SAT-by-consensus instances of one benchmark."""
    print(f'\n=== {experiment}: {benchmark} ===', flush=True)
    if experiment == 'exp1':
        sat = _exp1_sat_indices(benchmark, _K_SAT_INSTANCES)
    else:
        sat = _exp2_sat_indices(benchmark, _K_SAT_INSTANCES)
    if not sat:
        print(f'  no SAT-by-consensus instances; skipping', flush=True)
        return

    for (idx, onnx_name, vnn_name) in sat:
        print(f'  [idx={idx}] {onnx_name} + {vnn_name}', flush=True)
        try:
            if experiment == 'exp1':
                from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
                    list_instances,
                )
                inst_meta = list_instances(benchmark)
                onnx_rel, vnn_rel, _t = inst_meta[idx]
                network, boxes, spec = _load_exp1_instance(
                    benchmark, onnx_rel, vnn_rel,
                )
            else:
                network, boxes, spec, _name = _load_exp2_instance(
                    benchmark, idx,
                )
        except Exception as e:
            print(f'    load failed: {type(e).__name__}: {e}', flush=True)
            continue

        try:
            network = network.cuda()
        except Exception:
            pass

        for (n_restarts, n_steps) in _BUDGETS:
            lb, ub = boxes[0]
            sat_found, wall_s = _run_one(
                network, lb, ub, spec, n_restarts, n_steps,
            )
            print(f'    ({n_restarts:2d}, {n_steps:3d}): '
                  f'sat={sat_found}  wall={wall_s:.2f}s', flush=True)
            writer.writerow({
                'experiment': experiment,
                'benchmark': benchmark,
                'idx': idx,
                'onnx_file': onnx_name,
                'vnnlib_file': vnn_name,
                'n_restarts': n_restarts,
                'n_steps': n_steps,
                'sat_found': int(sat_found),
                'wall_s': f'{wall_s:.2f}',
            })
            out_file.flush()
            if sat_found:
                # Stop escalating on this instance once a budget works.
                break


def _aggregate_recommendations() -> None:
    if not _OUT.exists():
        print('no probe output found', file=sys.stderr)
        return
    print('\n\n=== Per-benchmark budget recommendations ===\n')
    by_bench: Dict[str, List[Dict[str, str]]] = {}
    with open(_OUT) as f:
        for r in csv.DictReader(f):
            by_bench.setdefault(r['benchmark'], []).append(r)
    for bench, rows in sorted(by_bench.items()):
        # For each instance, find the smallest (n_restarts*n_steps) budget
        # that succeeded.
        per_inst: Dict[int, Tuple[int, int, float]] = {}
        for r in rows:
            if int(r['sat_found']) != 1:
                continue
            idx = int(r['idx'])
            nr = int(r['n_restarts'])
            ns = int(r['n_steps'])
            wall = float(r['wall_s'])
            if (idx not in per_inst) or (nr * ns < per_inst[idx][0] * per_inst[idx][1]):
                per_inst[idx] = (nr, ns, wall)
        if not per_inst:
            print(f'{bench:25s}  no SAT found at any budget — recommend (20, 200)')
            continue
        # The smallest budget that works for ALL probed instances:
        smallest_works = max(
            (nr * ns, nr, ns) for (nr, ns, _w) in per_inst.values()
        )
        budget_product, nr_rec, ns_rec = smallest_works
        avg_wall = sum(w for (_n, _s, w) in per_inst.values()) / len(per_inst)
        print(f'{bench:25s}  recommend (n_restarts={nr_rec:2d}, '
              f'n_steps={ns_rec:3d})  '
              f'covers {len(per_inst)} SAT inst, avg wall {avg_wall:.1f}s')


def main() -> None:
    _OUT.parent.mkdir(parents=True, exist_ok=True)

    # Decide which benchmarks to probe based on PER_BENCHMARK_CONFIG.
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        EXP1_BENCHMARKS,
        PER_BENCHMARK_CONFIG as EXP1_CFG,
    )
    from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
        VNNCOMP_BENCHMARKS as EXP2_VNNCOMP,
        PER_BENCHMARK_CONFIG as EXP2_CFG,
    )
    exp1_targets = [b for b in EXP1_BENCHMARKS
                    if EXP1_CFG[b].get('use_falsifier', False)]
    exp2_targets = [b for b in EXP2_VNNCOMP
                    if EXP2_CFG[b].get('use_falsifier', False)]

    print(f'Probing Exp 1: {exp1_targets}')
    print(f'Probing Exp 2: {exp2_targets}  '
          f'(cifar10_resnet110 has no consensus; inherits from cifar100_2024)')
    print(f'Budgets: {_BUDGETS}, K={_K_SAT_INSTANCES} SAT instances per benchmark')
    print(f'Output CSV: {_OUT}\n')

    with open(_OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        writer.writeheader(); f.flush()
        for bench in exp1_targets:
            _probe_benchmark('exp1', bench, writer, f)
        for bench in exp2_targets:
            _probe_benchmark('exp2', bench, writer, f)

    _aggregate_recommendations()


if __name__ == '__main__':
    main()
