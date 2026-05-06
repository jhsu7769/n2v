"""Cross-experiment soundness audit runner.

Reads any per-instance CSV (Exp 1 / Exp 2 / Exp 4 ours or other
methods), filters to UNSAT rows, and runs an aggressive cex-search
pipeline (AutoAttack + 5K-restart PGD) directly on the original
network. A real counterexample on a UNSAT row flags a false UNSAT —
the very signal Exp 1 measures, double-checked here.

Wraps the existing attack logic from
:mod:`examples.FlowConformal.experiments.exp2_prob_scale.exp2_soundness_audit`
(``autoattack_audit``, ``pgd_5k_audit``, ``audit_instance``) and
extends benchmark dispatch to cover Exp 1 (acasxu, collins, dist_shift,
linearizenn, tllverify, vit_2023) and Exp 4 (synthetic networks at
each depth).

SEED=47 is reset per instance so the audit's PGD random restarts are
reproducible.

Usage::

    cd /home/sasakis/v/tools/n2v

    # Audit all UNSAT rows of Exp 1 ACAS Xu ours:
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.run_soundness_audit \\
        --benchmark acasxu_2023 --input-csv \\
        examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_acasxu_2023_ours.csv

    # Audit a single Exp 4 cell:
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.run_soundness_audit \\
        --benchmark exp4_d16 --input-csv \\
        examples/FlowConformal/experiments/exp4_scaling/outputs/exp4_d16_ours.csv

    # Smoke (1 instance, AutoAttack only — fast):
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.run_soundness_audit \\
        --benchmark vit_2023 --input-csv \\
        examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs/exp1_vit_2023_ours.csv \\
        --max-instances 1 --skip-pgd
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

# Reuse the existing attack helpers — these are well-tested.
from examples.FlowConformal.experiments.exp2_prob_scale.exp2_soundness_audit import (
    audit_instance,
    autoattack_audit as _autoattack_audit,  # noqa: F401 — re-export for completeness
    pgd_5k_audit as _pgd_5k_audit,           # noqa: F401
)

_SEED = 47

_EXP1_BENCHMARKS = (
    'acasxu_2023', 'collins_rul_cnn_2022', 'dist_shift_2023',
    'linearizenn_2024', 'tllverify_2023', 'malbeware',
    'metaroom_2023',
)
_EXP2_BENCHMARKS = (
    'vit_2023', 'tinyimagenet_2024', 'cifar100_2024', 'cifar10_resnet110',
)
_EXP4_DEPTHS = (2, 4, 8, 16, 24, 32, 40)
_EXP4_BENCHMARK_TAGS = tuple(f'exp4_d{d}' for d in _EXP4_DEPTHS)
_ALL_BENCHMARKS = sorted(set(_EXP1_BENCHMARKS) | set(_EXP2_BENCHMARKS)
                          | set(_EXP4_BENCHMARK_TAGS))


def _exp1_loader_factory(benchmark: str) -> Callable[[str], tuple]:
    """Loader for Exp 1 benchmarks. Returns ``loader(instance_name)`` →
    ``(network, x_clean, y_clean, eps_inf)`` per the audit contract.

    ``instance_name`` is ``onnx_basename + '+' + vnnlib_basename`` since
    that's what the new Exp 1 CSVs write (or column-pair onnx_file +
    vnnlib_file). The loader uses :mod:`exp1_vnncomp_subset._benchmarks`
    to materialise the network, then derives ``(x_clean, y_clean, eps)``
    from the input box's centre and the spec's true class.
    """
    from examples.FlowConformal.experiments.exp1_vnncomp_subset._benchmarks import (
        list_instances,
        load_one_instance,
    )
    from examples.FlowConformal.experiments.baselines.run_rs import (
        _extract_true_class_from_spec,
    )

    rows = list_instances(benchmark)
    by_name = {f'{Path(o).name}+{Path(v).name}': (o, v, t) for (o, v, t) in rows}

    def _load(instance_name: str):
        if instance_name not in by_name:
            raise KeyError(
                f'instance not found in {benchmark}: {instance_name}')
        onnx_rel, vnn_rel, _ = by_name[instance_name]
        net, boxes, spec = load_one_instance(benchmark, onnx_rel, vnn_rel)
        if len(boxes) != 1:
            raise NotImplementedError(
                'soundness audit on OR-of-input-regions not supported')
        lb, ub = boxes[0]
        lb = np.asarray(lb).flatten().astype(np.float32)
        ub = np.asarray(ub).flatten().astype(np.float32)
        x_clean = torch.as_tensor(0.5 * (lb + ub), dtype=torch.float32)
        eps_inf = float(np.max(0.5 * (ub - lb)))
        # Try to extract a true class from a classification spec; for
        # ACAS Xu / dist_shift / linearizenn / tllverify the spec is a
        # halfspace not a classifier, so y_clean is irrelevant for the
        # audit's verdict.
        y_clean = _extract_true_class_from_spec(spec)
        if y_clean is None:
            y_clean = -1
        # Stash spec + box bounds on the network so ``_check_cex`` can
        # do spec-aware cex verification (the same convention used by
        # the Exp 2 audit's ``_vnncomp_loader_factory``).
        net._audit_spec = spec
        net._audit_lb = torch.as_tensor(lb, dtype=torch.float32)
        net._audit_ub = torch.as_tensor(ub, dtype=torch.float32)
        return net, x_clean, int(y_clean), eps_inf

    return _load


def _exp4_loader_factory(depth: int) -> Callable[[str], tuple]:
    """Loader for Exp 4 synthetic networks. ``instance_name`` is
    ``f'exp4_d{depth}_i{idx}'`` (matching the runner's naming).
    """
    from examples.FlowConformal.experiments.exp4_scaling._benchmarks import (
        get_instance, get_network,
    )

    net, _ = get_network(depth)

    def _load(instance_name: str):
        # Parse "exp4_d<D>_i<idx>" or just "<idx>".
        if '_i' in instance_name:
            idx = int(instance_name.split('_i')[-1])
        else:
            idx = int(instance_name)
        inst = get_instance(depth, idx, net=net)
        lb = np.asarray(inst['lb']).flatten().astype(np.float32)
        ub = np.asarray(inst['ub']).flatten().astype(np.float32)
        x_clean = torch.as_tensor(0.5 * (lb + ub), dtype=torch.float32)
        eps_inf = float(np.max(0.5 * (ub - lb)))
        net._audit_spec = inst['spec_halfspace']
        net._audit_lb = torch.as_tensor(lb, dtype=torch.float32)
        net._audit_ub = torch.as_tensor(ub, dtype=torch.float32)
        return net, x_clean, -1, eps_inf

    return _load


def get_loader(benchmark: str) -> Callable[[str], tuple]:
    if benchmark in _EXP1_BENCHMARKS:
        return _exp1_loader_factory(benchmark)
    if benchmark in _EXP2_BENCHMARKS:
        # Reuse the Exp 2 dispatch from the legacy audit module.
        from examples.FlowConformal.experiments.exp2_prob_scale.exp2_soundness_audit import (
            get_loader as _exp2_get_loader,
        )
        # Build a fake args namespace for the Exp 2 loader's signature.
        ns = argparse.Namespace(benchmark=benchmark)
        return _exp2_get_loader(benchmark, ns)
    if benchmark.startswith('exp4_d'):
        depth = int(benchmark[len('exp4_d'):])
        return _exp4_loader_factory(depth)
    raise KeyError(f'unknown benchmark: {benchmark}')


_FIELDS = [
    'benchmark', 'instance_name', 'original_verdict',
    'audit_attack', 'found_cex', 'cex_pred',
    'audit_wall_s', 'error',
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=_ALL_BENCHMARKS)
    p.add_argument('--input-csv', type=Path, required=True,
                   help='Per-instance CSV from a runner (must include '
                        'verdict + onnx_file/vnnlib_file or instance).')
    p.add_argument('--output-csv', type=Path, default=None)
    p.add_argument('--filter-verdict', default='UNSAT',
                   help='Audit only rows with this verdict (default UNSAT).')
    p.add_argument('--max-instances', type=int, default=None)
    p.add_argument('--skip-aa', action='store_true',
                   help='Skip AutoAttack pass.')
    p.add_argument('--skip-pgd', action='store_true',
                   help='Skip 5K-restart PGD pass.')
    p.add_argument('--pgd-restarts', type=int, default=5000)
    p.add_argument('--pgd-steps', type=int, default=100)
    p.add_argument('--device', default='cuda',
                   help='cuda or cpu (default cuda; falls back automatically '
                        'if CUDA missing).')
    args = p.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        print('[audit] CUDA unavailable; falling back to CPU', flush=True)
        args.device = 'cpu'

    if not args.input_csv.exists():
        print(f'input CSV not found: {args.input_csv}', file=sys.stderr)
        sys.exit(2)

    out_csv = (args.output_csv if args.output_csv is not None
               else args.input_csv.with_name(
                   args.input_csv.stem + '_audit.csv'))

    # Read input rows; filter to verdict.
    with open(args.input_csv, newline='') as f:
        rows = list(csv.DictReader(f))
    filt = [r for r in rows if r.get('verdict', '').strip()
            == args.filter_verdict]
    if args.max_instances is not None:
        filt = filt[:args.max_instances]
    print(f'[audit] benchmark={args.benchmark}  input={args.input_csv}  '
          f'verdict_filter={args.filter_verdict}  rows_to_audit={len(filt)}',
          flush=True)

    loader = get_loader(args.benchmark)
    counts = {'cex_found': 0, 'no_cex': 0, 'error': 0}
    t_start = time.time()

    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=_FIELDS)
        w.writeheader()
        f.flush()

        for k, r in enumerate(filt, start=1):
            instance = r.get('instance', '').strip()
            if not instance:
                onnx = r.get('onnx_file', '').strip()
                vnn = r.get('vnnlib_file', '').strip()
                instance = f'{onnx}+{vnn}' if onnx and vnn else ''
            if not instance:
                # Exp 4 row: instance_idx column.
                idx = r.get('instance_idx', '').strip()
                instance = idx if idx else ''
            if not instance:
                continue

            elapsed = time.time() - t_start
            print(f'[audit {k}/{len(filt)}  t={elapsed:.0f}s] '
                  f'{instance}', flush=True)
            torch.manual_seed(_SEED)
            np.random.seed(_SEED)
            t0 = time.time()
            try:
                row = audit_instance(
                    loader, instance,
                    skip_aa=args.skip_aa, skip_pgd=args.skip_pgd,
                    pgd_restarts=args.pgd_restarts,
                    pgd_steps=args.pgd_steps,
                    device=args.device,
                )
            except Exception as e:
                row = {
                    'audit_attack': 'none',
                    'found_cex': 0, 'cex_pred': '',
                    'error': f'audit_failed {type(e).__name__}: {e}',
                }
            audit_wall_s = time.time() - t0
            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': args.benchmark,
                'instance_name': instance,
                'original_verdict': r.get('verdict', ''),
                'audit_wall_s': f'{audit_wall_s:.2f}',
            })
            out_row.update(row)
            w.writerow(out_row)
            f.flush()

            if row.get('found_cex'):
                counts['cex_found'] += 1
            elif row.get('error'):
                counts['error'] += 1
            else:
                counts['no_cex'] += 1
            print(f'    found_cex={row.get("found_cex", 0)}  '
                  f'wall={audit_wall_s:.1f}s', flush=True)

    print(f'\n=== Audit complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')
    if counts['cex_found'] > 0:
        print(f'⚠️  FOUND {counts["cex_found"]} false UNSATs '
              f'on {args.benchmark} — soundness violation')


if __name__ == '__main__':
    main()
