"""Common utilities for ablation experiments.

Each ablation row varies one design choice on the 20-instance ACAS Xu
probe set defined in
``examples.FlowConformal.ablations.phase5c_probe_sweep``. Output schema
mirrors that script:

    onnx_file, vnnlib_file, verdict, q, worst_max_margin, wall_s, error

Per-row scripts call :func:`run_probe_with_overrides` with a tag and a
set of ``run_verification_pipeline`` overrides; results are written to
``outputs/ablation_<tag>.csv``.

A ``--smoke`` flag is honored uniformly: passes ``smoke=True`` to
:func:`run_probe_with_overrides`, which restricts the probe to the first
two instances. Smoke runs are intended only to verify the harness wires
correctly end-to-end on a representative pair (one false-UNSAT-prone
instance + one easy UNSAT).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from examples.FlowConformal.ablations.phase5c_probe_sweep import (
    _BASE_KWARGS, _INSTANCES, _extract_worst_max_margin, _load_instance,
)
from examples.FlowConformal.benchmarks._common import (
    run_verification_pipeline,
)


_OUT_DIR = Path(__file__).parent / 'outputs'


def _fmt(v, spec):
    return f'{v:{spec}}' if v is not None else ''


def run_probe_with_overrides(
    tag: str,
    out_dir: Path | None = None,
    *,
    smoke: bool = False,
    instances=None,
    **overrides,
) -> Path:
    """Run the 20-instance probe with kwargs overrides.

    ``overrides`` are merged into ``_BASE_KWARGS`` and forwarded to
    :func:`run_verification_pipeline` per instance. The output CSV is
    written incrementally (flushed per row) so a partial run leaves a
    usable artifact.

    Args:
        tag: Label for the output filename (``ablation_<tag>.csv``).
        out_dir: Output directory. Defaults to ``./outputs/`` next to
            this module.
        smoke: When True, run only the first two probe instances.
            Intended for harness verification (~1-2 min total).
        instances: Optional explicit list of (onnx_rel, vnn_rel) pairs.
            Overrides both the default 20-instance probe and ``smoke``.
        **overrides: Forwarded to ``run_verification_pipeline`` after
            merging onto ``_BASE_KWARGS``. Common knobs:
            ``verification_method``, ``flow_config``, ``alpha``, ``m``,
            ``ell``, ``flow_epochs``, ``sampling_strategy``,
            ``adaptive_threshold``, ``adaptive_n_samples``, ...

    Returns:
        Path to the written CSV.
    """
    out_dir = out_dir or _OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f'ablation_{tag}.csv'

    if instances is None:
        instances = _INSTANCES[:2] if smoke else _INSTANCES

    kwargs = dict(_BASE_KWARGS, **overrides)
    fields = ['onnx_file', 'vnnlib_file', 'verdict', 'q',
              'worst_max_margin', 'amls_levels_used', 'wall_s', 'error']

    print(f'[ablation] tag={tag}  smoke={smoke}  n_instances={len(instances)}',
          flush=True)
    print(f'[ablation] kwargs={kwargs}', flush=True)
    print(f'[ablation] out={out_csv}', flush=True)

    t0 = time.time()
    with open(out_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        f.flush()
        for k, (onnx_rel, vnn_rel) in enumerate(instances, start=1):
            inst_seed = hash((onnx_rel, vnn_rel)) & 0x7FFFFFFF
            print(f'[{k}/{len(instances)}] {Path(onnx_rel).name}'
                  f'+{Path(vnn_rel).name}', flush=True)
            t_inst = time.time()
            try:
                network, lb, ub, spec = _load_instance(onnx_rel, vnn_rel)
                res = run_verification_pipeline(
                    network=network, input_lb=lb, input_ub=ub, spec=spec,
                    seed=inst_seed, **kwargs,
                )
                wmm = _extract_worst_max_margin(res)
                amls_lvls = res.get('amls_levels_used')
                row = {
                    'onnx_file': Path(onnx_rel).name,
                    'vnnlib_file': Path(vnn_rel).name,
                    'verdict': res['verdict'],
                    'q': _fmt(res.get('q'), '.6f'),
                    'worst_max_margin': _fmt(wmm, '.6f'),
                    'amls_levels_used': (
                        str(amls_lvls) if amls_lvls is not None else ''
                    ),
                    'wall_s': f'{time.time() - t_inst:.1f}',
                    'error': '',
                }
            except Exception as e:
                row = {
                    'onnx_file': Path(onnx_rel).name,
                    'vnnlib_file': Path(vnn_rel).name,
                    'verdict': 'ERROR', 'q': '', 'worst_max_margin': '',
                    'amls_levels_used': '',
                    'wall_s': f'{time.time() - t_inst:.1f}',
                    'error': f'{type(e).__name__}: {e}',
                }
            w.writerow(row)
            f.flush()
            print(f'    verdict={row["verdict"]}  q={row["q"]}  '
                  f'wmm={row["worst_max_margin"]}  wall={row["wall_s"]}s'
                  + (f'  err={row["error"]}' if row['error'] else ''),
                  flush=True)
    print(f'[ablation] wrote {out_csv}  total wall '
          f'{(time.time() - t0) / 60:.1f} min', flush=True)
    return out_csv
