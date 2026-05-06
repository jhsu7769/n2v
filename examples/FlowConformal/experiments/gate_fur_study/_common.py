"""Shared helpers for the gate-only FUR study.

Two responsibilities:

1. **Benchmark dispatch.** Each benchmark lives in either Exp 1 or
   Exp 2; this module normalises that so the runners take a single
   ``--benchmark`` arg without caring which experiment owns it.
2. **SAT-instance filter.** Reads each experiment's ``ground_truth.csv``
   (the VNN-COMP 2025 SAT-wins consensus) and returns only those
   ``(onnx_rel, vnn_rel, vnncomp_t)`` tuples whose ground truth is SAT.

vit_2023 has zero SAT instances under consensus and is therefore
excluded from the dispatch table. Calling ``list_sat_instances('vit_2023')``
returns an empty list rather than raising.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
    EXP1_BENCHMARKS,
    PER_BENCHMARK_CONFIG as EXP1_CONFIG,
    list_instances as exp1_list_instances,
    load_one_instance as exp1_load_one_instance,
)
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_BENCHMARKS,
    PER_BENCHMARK_CONFIG as EXP2_CONFIG,
    list_instances as exp2_list_instances,
    load_one_instance as exp2_load_one_instance,
)

_HERE = Path(__file__).resolve().parent
_EXPERIMENTS_DIR = _HERE.parent
_GT_CSV_BY_EXPERIMENT = {
    'exp1': _EXPERIMENTS_DIR / 'exp1_vnncomp_subset' / 'ground_truth.csv',
    'exp2': _EXPERIMENTS_DIR / 'exp2_prob_scale' / 'ground_truth.csv',
}

# All FUR-eligible benchmarks: every Exp 1 benchmark plus the two Exp 2
# VNN-COMP benchmarks with at least one SAT-ground-truth instance.
# vit_2023 has zero SAT instances and cifar10_resnet110 lacks a VNN-COMP
# ground truth (locally generated), so neither participates.
_BENCH_DISPATCH: Dict[str, Tuple[str, dict, Callable, Callable]] = {}
for _b in EXP1_BENCHMARKS:
    _BENCH_DISPATCH[_b] = (
        'exp1', EXP1_CONFIG[_b], exp1_list_instances, exp1_load_one_instance,
    )
for _b in ('tinyimagenet_2024', 'cifar100_2024'):
    _BENCH_DISPATCH[_b] = (
        'exp2', EXP2_CONFIG[_b], exp2_list_instances, exp2_load_one_instance,
    )

GATE_FUR_BENCHMARKS: Tuple[str, ...] = tuple(sorted(_BENCH_DISPATCH.keys()))


# Per-benchmark Hashemi-clipping config used by the gate-FUR study so the
# stress-test mirrors the production run's Hashemi configuration on each
# benchmark. ``pca_components=None`` means raw clipping_block (small
# output dim, no PCA needed); ``pca_components=K`` means PCA-augmented
# clipping with K components (K matches the value used by the production
# Phase 6 / Phase 2-prime runner for that benchmark).
HASHEMI_CONFIG_BY_BENCHMARK: Dict[str, Dict[str, object]] = {
    # Exp 1 small-output benches: raw clipping_block
    'acasxu_2023':            {'pca_components': None},
    'collins_rul_cnn_2022':   {'pca_components': None},
    'dist_shift_2023':        {'pca_components': None},
    'linearizenn_2024':       {'pca_components': None},
    'tllverify_2023':         {'pca_components': None},
    'malbeware':              {'pca_components': None},
    # Exp 1 medium-output bench: PCA per Phase 6 production run
    'metaroom_2023':          {'pca_components': 10},
    # New Exp 1 benches (smoke-decided): raw clipping (small output dim)
    'lsnc_relu':              {'pca_components': None},
    'relusplitter':           {'pca_components': None},
    # Exp 2 high-output benches: PCA per Phase 2-prime production run
    'tinyimagenet_2024':      {'pca_components': 32},
    'cifar100_2024':          {'pca_components': 32},
    # vit_2023 has zero SAT instances so it isn't in GATE_FUR_BENCHMARKS;
    # listed here for completeness if it's ever added back.
    'vit_2023':               {'pca_components': None},
}


def hashemi_config(benchmark: str) -> Dict[str, object]:
    """Return the Hashemi-clipping config for a gate-FUR run on ``benchmark``.

    The returned dict is passed as kwargs to ``n2v.probabilistic.verify``
    (specifically ``pca_components`` is forwarded). Default for any
    benchmark not in the table is ``{'pca_components': None}`` (raw
    clipping_block).
    """
    return HASHEMI_CONFIG_BY_BENCHMARK.get(benchmark, {'pca_components': None})


def benchmark_dispatch(
    benchmark: str,
) -> Tuple[str, dict, Callable, Callable]:
    """Return ``(experiment, cfg, list_instances_fn, load_one_instance_fn)``.

    ``cfg`` is the per-benchmark config from the owning experiment's
    ``PER_BENCHMARK_CONFIG``; the gate-only runners override
    ``use_falsifier`` to ``False`` regardless of its value here.
    """
    if benchmark not in _BENCH_DISPATCH:
        raise KeyError(
            f'unknown gate-FUR benchmark: {benchmark}; '
            f'expected one of {GATE_FUR_BENCHMARKS}')
    return _BENCH_DISPATCH[benchmark]


def _read_ground_truth(experiment: str, benchmark: str) -> Dict[Tuple[str, str], str]:
    """Return ``{(onnx_basename, vnn_basename): verdict_upper}`` for one bench."""
    csv_path = _GT_CSV_BY_EXPERIMENT[experiment]
    if not csv_path.exists():
        raise FileNotFoundError(f'ground_truth.csv missing at {csv_path}')
    out: Dict[Tuple[str, str], str] = {}
    with open(csv_path, newline='') as f:
        for r in csv.DictReader(f):
            if r['benchmark'].strip() != benchmark:
                continue
            key = (r['onnx_file'].strip(), r['vnnlib_file'].strip())
            out[key] = r['ground_truth'].strip().upper()
    return out


def list_sat_instances(benchmark: str) -> List[Tuple[str, str, int]]:
    """Return only those instances whose VNN-COMP consensus verdict is SAT.

    Each tuple is ``(onnx_rel, vnn_rel, vnncomp_timeout_s)`` matching the
    raw output of the experiment's own ``list_instances``. The filter
    matches on basename (the same key shape ``ground_truth.csv`` uses).
    """
    experiment, _cfg, list_fn, _load_fn = benchmark_dispatch(benchmark)
    raw = list_fn(benchmark)
    gt = _read_ground_truth(experiment, benchmark)

    sat: List[Tuple[str, str, int]] = []
    for onnx_rel, vnn_rel, vnncomp_t in raw:
        key = (Path(onnx_rel).name, Path(vnn_rel).name)
        if gt.get(key) == 'SAT':
            sat.append((onnx_rel, vnn_rel, vnncomp_t))
    return sat
