"""Aggregate per-row ablation CSVs into a single summary table.

Reads ONLY from ``exp_ablation/outputs/``. There are no fallback paths
into legacy / archive directories — any missing Phase 7 cell shows
"(missing)" in the report instead of silently substituting older
methodology data.

Produces a markdown table with columns:

    | row (axis : value) | n_inst | unsat | unknown | error | mean_wall_s |

For the 3D-banana score-function row, additionally reports:

    | score | mean_volume | volume_ratio_vs_exact |

Usage:

    /home/sasakis/miniconda3/envs/n2v/bin/python -u -m \\
        examples.FlowConformal.experiments.exp_ablation.ablation_aggregate \\
        [--out path/to/markdown.md]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean


_HERE = Path(__file__).parent
_OUR_OUT = _HERE / 'outputs'


# Mapping: aggregated row label -> csv path under _OUR_OUT.
# Prints "(missing)" rather than crashing if the CSV doesn't exist.
_VERIFY_METHOD_ROWS = [
    ('verify_method : scenario',
     _OUR_OUT / 'ablation_verify_method_scenario.csv'),
    ('verify_method : scenario_v2',
     _OUR_OUT / 'ablation_verify_method_scenario_v2.csv'),
    ('verify_method : amls (locked)',
     _OUR_OUT / 'ablation_verify_method_amls.csv'),
    ('verify_method : is_tilted',
     _OUR_OUT / 'ablation_verify_method_is_tilted.csv'),
    ('verify_method : derived',
     _OUR_OUT / 'ablation_verify_method_derived.csv'),
]

_AMLS_HPARAM_ROWS = [
    ('amls_rho : 0.05', _OUR_OUT / 'ablation_amls_hparam_rho0.05.csv'),
    ('amls_rho : 0.1',  _OUR_OUT / 'ablation_amls_hparam_rho0.1.csv'),
    ('amls_rho : 0.2',  _OUR_OUT / 'ablation_amls_hparam_rho0.2.csv'),
    ('amls_mcmc : 5',   _OUR_OUT / 'ablation_amls_hparam_mcmc5.csv'),
    ('amls_mcmc : 10',  _OUR_OUT / 'ablation_amls_hparam_mcmc10.csv'),
    ('amls_mcmc : 20',  _OUR_OUT / 'ablation_amls_hparam_mcmc20.csv'),
    ('amls_mcmc : 40',  _OUR_OUT / 'ablation_amls_hparam_mcmc40.csv'),
]

# Conformal-parameters ablation rows (alpha / m / ell-offset / beta_2).
_CONFORMAL_PARAMS_ROWS = [
    # alpha sweep
    ('conformal_params : alpha=0.001',
     _OUR_OUT / 'ablation_conformal_params_alpha0.001.csv'),
    ('conformal_params : alpha=0.01',
     _OUR_OUT / 'ablation_conformal_params_alpha0.01.csv'),
    ('conformal_params : alpha=0.05',
     _OUR_OUT / 'ablation_conformal_params_alpha0.05.csv'),
    ('conformal_params : alpha=0.1',
     _OUR_OUT / 'ablation_conformal_params_alpha0.1.csv'),
    # m sweep
    ('conformal_params : m=500',
     _OUR_OUT / 'ablation_conformal_params_m500.csv'),
    ('conformal_params : m=2000',
     _OUR_OUT / 'ablation_conformal_params_m2000.csv'),
    ('conformal_params : m=8000',
     _OUR_OUT / 'ablation_conformal_params_m8000.csv'),
    # ell-offset sweep
    ('conformal_params : ell-off=0',
     _OUR_OUT / 'ablation_conformal_params_elloff0.csv'),
    ('conformal_params : ell-off=1',
     _OUR_OUT / 'ablation_conformal_params_elloff1.csv'),
    ('conformal_params : ell-off=5',
     _OUR_OUT / 'ablation_conformal_params_elloff5.csv'),
    # beta_2 sweep
    ('conformal_params : beta2=0.001',
     _OUR_OUT / 'ablation_conformal_params_beta20.001.csv'),
    ('conformal_params : beta2=0.01',
     _OUR_OUT / 'ablation_conformal_params_beta20.01.csv'),
    ('conformal_params : beta2=0.1',
     _OUR_OUT / 'ablation_conformal_params_beta20.1.csv'),
]

# Flow-training ablation rows: n_train x flow_epochs grid.
_FLOW_TRAIN_ROWS = [
    # n_train sweep at locked flow_epochs=2000
    ('flow_train : n_train=1000',
     _OUR_OUT / 'ablation_flow_training_n1000_e2000.csv'),
    ('flow_train : n_train=2000',
     _OUR_OUT / 'ablation_flow_training_n2000_e2000.csv'),
    ('flow_train : n_train=5000 (locked)',
     _OUR_OUT / 'ablation_flow_training_n5000_e2000.csv'),
    ('flow_train : n_train=10000',
     _OUR_OUT / 'ablation_flow_training_n10000_e2000.csv'),
    ('flow_train : n_train=20000',
     _OUR_OUT / 'ablation_flow_training_n20000_e2000.csv'),
    ('flow_train : n_train=50000',
     _OUR_OUT / 'ablation_flow_training_n50000_e2000.csv'),
    # flow_epochs sweep at locked n_train=5000
    ('flow_train : flow_epochs=500',
     _OUR_OUT / 'ablation_flow_training_n5000_e500.csv'),
    ('flow_train : flow_epochs=1000',
     _OUR_OUT / 'ablation_flow_training_n5000_e1000.csv'),
    ('flow_train : flow_epochs=2000 (locked)',
     _OUR_OUT / 'ablation_flow_training_n5000_e2000.csv'),
    ('flow_train : flow_epochs=5000',
     _OUR_OUT / 'ablation_flow_training_n5000_e5000.csv'),
]


def _resolve(path: Path) -> Path | None:
    return path if path.exists() else None


def _summarize_acasxu_csv(path: Path) -> dict:
    n_unsat = n_unknown = n_error = 0
    walls = []
    with open(path) as f:
        for row in csv.DictReader(f):
            v = row.get('verdict')
            if v == 'UNSAT':
                n_unsat += 1
            elif v == 'UNKNOWN':
                n_unknown += 1
            else:
                n_error += 1
            try:
                walls.append(float(row.get('wall_s', '') or 'nan'))
            except ValueError:
                pass
    walls = [w for w in walls if w == w]
    return {
        'n_inst': n_unsat + n_unknown + n_error,
        'unsat': n_unsat,
        'unknown': n_unknown,
        'error': n_error,
        'mean_wall_s': mean(walls) if walls else float('nan'),
        'source': str(path),
    }


def _summarize_score_csv(path: Path) -> list[dict]:
    """Group ablation_score.csv rows by ``(network, score)`` and average.

    Backward compat: legacy CSVs without the ``network`` column are
    treated as if every row had ``network='3d_banana'`` (the original
    score-ablation benchmark).
    """
    by_key: dict[tuple, list[dict]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            net = row.get('network') or '3d_banana'
            by_key.setdefault((net, row['score']), []).append(row)
    out = []
    for (net, score), rows in by_key.items():
        try:
            vols = [float(r['volume']) for r in rows
                    if r['volume'] not in ('', 'nan')]
            ratios = [float(r['volume_ratio']) for r in rows
                      if r['volume_ratio'] not in ('', 'nan')]
        except ValueError:
            vols, ratios = [], []
        out.append({
            'network': net,
            'score': score,
            'n_seeds': len(rows),
            'mean_volume': mean(vols) if vols else float('nan'),
            'mean_volume_ratio': mean(ratios) if ratios else float('nan'),
        })
    return out


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    cols = list(zip(headers, *rows))
    widths = [max(len(str(x)) for x in col) for col in cols]
    fmt = ' | '.join(f'{{:<{w}}}' for w in widths)
    sep = ' | '.join('-' * w for w in widths)
    out = ['| ' + fmt.format(*headers) + ' |',
           '| ' + sep + ' |']
    for r in rows:
        out.append('| ' + fmt.format(*r) + ' |')
    return '\n'.join(out)


def _row_for(label: str, path: Path) -> list[str]:
    p = _resolve(path)
    if p is None:
        return [label, '(missing)', '', '', '', '']
    s = _summarize_acasxu_csv(p)
    return [label, str(s['n_inst']), str(s['unsat']), str(s['unknown']),
            str(s['error']), f'{s["mean_wall_s"]:.1f}']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=None,
                   help='write markdown to this path (default: stdout)')
    args = p.parse_args()

    sections = []

    # ACAS-Xu probe rows.
    headers = ['row', 'n_inst', 'unsat', 'unknown', 'error', 'mean_wall_s']

    def _section(title: str, row_specs):
        rows = [_row_for(label, path) for (label, path) in row_specs]
        return f'## {title}\n\n' + _md_table(headers, rows) + '\n'

    sections.append(_section('Verification method', _VERIFY_METHOD_ROWS))
    sections.append(_section('AMLS hyperparameters', _AMLS_HPARAM_ROWS))
    sections.append(_section('Conformal parameters', _CONFORMAL_PARAMS_ROWS))
    sections.append(_section('Flow training', _FLOW_TRAIN_ROWS))

    # Score-function ablation: 3D banana. CSV layout differs.
    score_csv = _OUR_OUT / 'ablation_score.csv'
    sec = ['## Score function (3D banana, volume tightness)\n']
    if score_csv.exists():
        rows = _summarize_score_csv(score_csv)
        sec.append(_md_table(
            ['network', 'score', 'n_seeds', 'mean_volume', 'mean_volume_ratio'],
            [[r.get('network', '3d_banana'),
              r['score'], str(r['n_seeds']),
              f'{r["mean_volume"]:.4f}',
              f'{r["mean_volume_ratio"]:.3f}']
             for r in rows],
        ))
    else:
        sec.append('(missing — run `ablation_run_score.py`)')
    sections.append('\n'.join(sec) + '\n')

    md = ('# Ablation results (aggregated)\n\n' + '\n'.join(sections))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f'Wrote {args.out}')
    else:
        print(md)


if __name__ == '__main__':
    main()
