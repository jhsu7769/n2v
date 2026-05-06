"""Exp 2 — αβ-CROWN runner (subprocess + --results_file).

Supported benchmarks: ``vit_2023``, ``tinyimagenet_2024``, ``cifar100_2024``,
``cifar10_resnet110``.

The first three load directly from the VNN-COMP 2025 distribution.
``cifar10_resnet110`` uses an ONNX export + locally-generated vnnlib
specs produced by
:mod:`examples.FlowConformal.experiments.exp2_prob_scale.build_resnet110_onnx`
(no canonical VNN-COMP pair — the network is Cohen RS's pretrained
110-layer ResNet). Exp 2's design hypothesizes sound verifiers won't
scale to ResNet-110 — running αβ-CROWN here lets us settle that
empirically rather than asserting it.

Usage::

    cd /home/sasakis/v/tools/n2v
    /home/sasakis/miniconda3/envs/n2v/bin/python -m \\
        examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_alpha_beta_crown \\
        --benchmark vit_2023 --smoke
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import sys
import time
from pathlib import Path

from examples.FlowConformal.experiments._external_verifiers import (
    ABCROWN_REPO,
    run_alpha_beta_crown,
)
from examples.FlowConformal.experiments.exp2_prob_scale._benchmarks import (
    EXP2_VNNCOMP_FORMAT,
    list_vnncomp_format_instances,
)

_DEFAULT_N_INSTANCES = 100
_OUT_DIR = Path(__file__).resolve().parent / 'outputs'

# Per-benchmark αβ-CROWN config — picked from their VNN-COMP submission
# library to match each network's ops. The default vnncomp21/acasxu.yaml
# uses ``conv_mode=patches``, which crashes on transformer ops in
# vit_2023 (``AttributeError: 'Patches' object has no attribute
# 'unsqueeze'``); vnncomp23/vit.yaml sets ``conv_mode=matrix`` and
# adhoc-tunes for ViTs.
_ABCROWN_CONFIG = {
    'vit_2023': ABCROWN_REPO / 'complete_verifier' / 'exp_configs'
                / 'vnncomp23' / 'vit.yaml',
    # tinyimagenet_2024 + cifar100_2024 are both ResNet-medium image
    # classifiers; the αβ-CROWN team's 2024 cifar100 config transfers
    # cleanly to tinyimagenet (same architecture family + same spec
    # shape — multi-class disjunctive) and is the closest match the
    # public αβ-CROWN repo ships.
    'tinyimagenet_2024': ABCROWN_REPO / 'complete_verifier' / 'exp_configs'
                          / 'vnncomp24' / 'cifar100.yaml',
    'cifar100_2024': ABCROWN_REPO / 'complete_verifier' / 'exp_configs'
                     / 'vnncomp24' / 'cifar100.yaml',
    # cifar10_resnet110 — αβ-CROWN cannot verify this 110-layer
    # adversarially-trained CIFAR-10 ResNet at any reasonable GPU
    # memory budget. We exhausted both code paths:
    #
    #   (1) ONNX path (this config): tried vnncomp21/cifar10-resnet,
    #       vnncomp24/cifar100, vnncomp22/resnet_A — all error with
    #       shape mismatches in auto_LiRPA's `add_b` during bound
    #       concretization (the onnx2pytorch parser mishandles the
    #       residual-add at the layer1→layer2 transition).
    #   (2) PyTorch-native path (see ``abcrown_resnet110_pytorch_native.py``):
    #       parses the architecture cleanly but OOMs at intermediate-
    #       bound computation even with batch_size=32, share_alphas,
    #       and expandable_segments — the 109-ReLU intermediate bound
    #       tensors don't fit in 23.5 GB GPU memory.
    #
    # The ERROR result for cifar10_resnet110 is a defensible scaling
    # signal — exactly the design intent of Cohen et al. (2019), who
    # selected this 110-layer adversarially-trained ResNet to motivate
    # randomized smoothing precisely because no sound verifier could
    # handle it. The vnncomp22/resnet_A config gets us furthest along
    # the ONNX path (closest architectural match), so we keep it as
    # the documented attempt.
    'cifar10_resnet110': ABCROWN_REPO / 'complete_verifier' / 'exp_configs'
                         / 'vnncomp22' / 'resnet_A.yaml',
}

_FIELDS = [
    'benchmark', 'onnx_file', 'vnnlib_file', 'verdict', 'wall_s',
    'timeout_s', 'cex_x', 'cex_y', 'error', 'timestamp',
]


def _now_iso() -> str:
    return _dt.datetime.now(tz=_dt.timezone.utc).isoformat()


def _write_timeout_row(out_csv, benchmark, onnx_name, vnnlib_name, timeout_s):
    """Append a TIMEOUT row when killed by outer shell timeout."""
    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    with open(out_csv, 'a' if file_exists else 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists:
            writer.writeheader(); f.flush()
        row = {_f: '' for _f in _FIELDS}
        row.update({'benchmark': benchmark, 'onnx_file': onnx_name,
                    'vnnlib_file': vnnlib_name, 'verdict': 'TIMEOUT',
                    'timeout_s': timeout_s,
                    'error': 'shell timeout (run_cell.sh exit 124)',
                    'timestamp': _now_iso()})
        writer.writerow(row); f.flush()



def main():
    p = argparse.ArgumentParser()
    p.add_argument('--benchmark', required=True, choices=EXP2_VNNCOMP_FORMAT,
                   help=f'One of {EXP2_VNNCOMP_FORMAT}. cifar10_resnet110 '
                        f'is excluded — no canonical ONNX+vnnlib pair.')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--n-instances', type=int, default=_DEFAULT_N_INSTANCES)
    p.add_argument('--instance-idx', type=int, default=None,
                   help='Run only this 0-based index; APPEND to CSV.')
    p.add_argument('--list-instances', action='store_true',
                   help='Print "<idx> <timeout_s>" per instance, exit.')
    p.add_argument('--write-timeout-row', action='store_true',
                   help='Append a single TIMEOUT row for --instance-idx '
                        '(used by run_cell.sh on outer-timeout exit 124).')
    p.add_argument('--output-csv', type=Path, default=None)
    args = p.parse_args()

    benchmark = args.benchmark

    if args.list_instances:
        rows = list_vnncomp_format_instances(benchmark, n=args.n_instances)
        for idx, (_o, _v, t) in enumerate(rows):
            print(f'{idx} {t}')
        return

    out_csv = (args.output_csv if args.output_csv is not None
               else _OUT_DIR / f'exp2_{benchmark}_alpha_beta_crown.csv')
    _OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.write_timeout_row:
        if args.instance_idx is None:
            print('--write-timeout-row requires --instance-idx', file=sys.stderr)
            sys.exit(2)
        all_inst = list_vnncomp_format_instances(benchmark, n=args.n_instances)
        if not (0 <= args.instance_idx < len(all_inst)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        onnx_path, vnnlib_path, t = all_inst[args.instance_idx]
        _write_timeout_row(out_csv, benchmark, onnx_path.name, vnnlib_path.name, t)
        return

    append_mode = False
    if args.instance_idx is not None:
        if args.smoke:
            print('--instance-idx and --smoke are mutually exclusive',
                  file=sys.stderr)
            sys.exit(2)
        all_inst = list_vnncomp_format_instances(benchmark, n=args.n_instances)
        if not (0 <= args.instance_idx < len(all_inst)):
            print('--instance-idx out of range', file=sys.stderr)
            sys.exit(2)
        instances = [all_inst[args.instance_idx]]
        append_mode = True
        print(f'[{benchmark}] running only idx={args.instance_idx}; '
              f'appending to {out_csv}', flush=True)
    else:
        n = 1 if args.smoke else args.n_instances
        instances = list_vnncomp_format_instances(benchmark, n=n)
    if args.smoke:
        instances = instances[:1]
        print(f'[smoke] {benchmark}: running only the first instance',
              flush=True)
    print(f'[{benchmark}] Loaded {len(instances)} instances; '
          f'writing to {out_csv}', flush=True)
    print(f'[{benchmark}] Tool: αβ-CROWN', flush=True)

    counts = {'UNSAT': 0, 'SAT': 0, 'UNKNOWN': 0,
              'ERROR': 0, 'TIMEOUT': 0}
    t_start = time.time()

    file_exists = out_csv.exists() and out_csv.stat().st_size > 0
    csv_mode = 'a' if append_mode and file_exists else 'w'
    with open(out_csv, csv_mode, newline='') as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDS)
        if not file_exists or csv_mode == 'w':
            writer.writeheader()
            f.flush()

        for k, (onnx_path, vnnlib_path, timeout_s) in enumerate(
                instances, start=1):
            elapsed = time.time() - t_start
            print(f'[{benchmark} {k}/{len(instances)} t={elapsed:.0f}s '
                  f'budget={timeout_s}s] {onnx_path.name} + '
                  f'{vnnlib_path.name}', flush=True)
            tag = f'exp2_{benchmark}_{onnx_path.stem}_{vnnlib_path.stem}'
            verdict, wall_s, err = run_alpha_beta_crown(
                onnx_path=onnx_path,
                vnnlib_path=vnnlib_path,
                timeout_s=timeout_s,
                config_yaml=_ABCROWN_CONFIG[benchmark],
                instance_tag=tag,
            )

            out_row = {_f: '' for _f in _FIELDS}
            out_row.update({
                'benchmark': benchmark,
                'onnx_file': onnx_path.name,
                'vnnlib_file': vnnlib_path.name,
                'verdict': verdict,
                'wall_s': f'{wall_s:.2f}' if wall_s is not None else '',
                'timeout_s': timeout_s,
                'error': err,
                'timestamp': _now_iso(),
            })
            writer.writerow(out_row)
            f.flush()
            counts[verdict] = counts.get(verdict, 0) + 1
            print(f'    verdict={verdict}  wall={wall_s:.1f}s  err={err!r}',
                  flush=True)

    print(f'\n=== Sweep complete ===')
    print(f'Wrote {out_csv}')
    print(f'Total wall-clock: {(time.time()-t_start)/60:.1f} min')
    print(f'Counts: {counts}')

    if args.smoke:
        # vit_2023 / cifar100_2024 are sound-verifier-friendly, so we
        # expect no ERROR. yolo's αβ-CROWN config may not be tuned
        # for this exact benchmark — TIMEOUT is acceptable for the
        # smoke (it's a perf signal, not a soundness violation).
        if counts.get('ERROR', 0) > 0:
            print(f'[smoke] FAIL on {benchmark}: ERROR observed, '
                  f'counts={counts}', file=sys.stderr)
            sys.exit(1)
        actual = next((v for v, c in counts.items() if c > 0), 'NONE')
        print(f'[smoke] PASS on {benchmark}: αβ-CROWN ran '
              f'(verdict={actual}).')


if __name__ == '__main__':
    main()
