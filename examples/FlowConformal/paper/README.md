# Paper figures and tables

This directory holds every figure and table that ships with the
flow-conformal probabilistic verifier paper. Each artefact is generated
by a single Python script that reads CSV inputs from a configurable
directory and writes a paper-ready output (`.png` for figures, `.tex`
for tables).

## Layout

```
paper/
├── _common.py                       # shared CSV / styling helpers + sound-verifier lists
├── figures/
│   ├── _common.py                   # matplotlib styling helpers
│   ├── flow_matching_training/      # pre-rendered overlay + heavyweight training script
│   │   ├── fig_training_progression.py
│   │   ├── fig_training_progression_checkpoints.pt
│   │   └── overlay.png
│   ├── fig1_flow_training_progression.py
│   ├── fig2_exp1_runtime.py
│   ├── fig3_exp2_runtime.py
│   ├── fig4_score_vs_dim.py            # original log-y line plot
│   ├── fig4a_score_vs_dim_linear.py    # linear-y, value-clipped variant
│   ├── fig5_scaling.py                 # original log-log
│   ├── fig5a_scaling_linear.py         # linear-y variant
│   ├── fig5b_scaling_semilog.py        # log-log variant (renamed from old fig5)
│   ├── fig6_ablation_grid.py           # legacy combined 3-panel
│   ├── fig6a_ablation_amls_hparam.py   # split panel a
│   ├── fig6b_ablation_conformal_params.py   # split panel b
│   ├── fig6c_ablation_flow_training.py # split panel c
│   ├── fig7_banana_score_geometries.py # 2D banana score-overlay (standalone, trains a flow)
│   └── fig*.png                     # generated outputs (committed for convenience)
├── tables/
│   ├── _common.py
│   ├── tab1_exp1_verdict_matrix.py     # 4 sound verifiers + 4 prob baselines + ours
│   ├── tab2_exp2_verdict_matrix.py     # only αβ-CROWN among sound + 4 prob + ours
│   ├── tab3_verify_method_ablation.py
│   ├── tab_score_vs_dim.py             # alternative to fig4
│   ├── tab_amls_hparam.py              # ρ × MCMC false-UNSAT counts
│   ├── tab_conformal_params.py         # one row per (knob, value)
│   ├── tab_flow_training.py            # n_train × flow_epochs grid
│   ├── tab_exp1_runtime.py             # alternative to fig2
│   ├── tab_exp2_runtime.py             # alternative to fig3
│   └── tab*.tex                     # generated LaTeX fragments
├── regenerate_all.py                # one-shot regenerator (auto-discovers scripts)
└── README.md                        # this file
```

## Methodology change (2026-04 paper revision)

- **Marabou is dropped from all references.** Sound verifiers for
  Exp 1 are now: αβ-CROWN, NeuralSAT, PyRAT, CORA (4 total). For
  Exp 2 only αβ-CROWN is shown (NeuralSAT, PyRAT, NNV, Rover are
  dropped, because they are TIMEOUT-heavy at Exp 2 scale and add no
  signal).
- These canonical lists live in `_common.py` as
  `EXP1_SOUND_VERIFIERS` and `EXP2_SOUND_VERIFIERS`.

## Regenerating a single artefact

Each script accepts the same two flags:

```
--csv-dir <path>   directory containing the input CSVs (REQUIRED — no default)
--output  <path>   output file (default: <script_name>.{png,tex})
```

There is **no fake-data fallback**. Every invocation must point
`--csv-dir` at a real experiment outputs directory; running a script
without `--csv-dir` raises a hard error to make sure paper artifacts
can never be silently rendered from synthetic / stale data.

Examples:

```bash
# Render a figure from real ablation results:
python examples/FlowConformal/paper/figures/fig4a_score_vs_dim_linear.py \
    --csv-dir examples/FlowConformal/experiments/exp_ablation/outputs

# Override the output path:
python examples/FlowConformal/paper/tables/tab1_exp1_verdict_matrix.py \
    --csv-dir examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs \
    --output /tmp/tab1.tex
```

The scripts can also be invoked as modules:

```bash
python -m examples.FlowConformal.paper.figures.fig4a_score_vs_dim_linear \
    --csv-dir examples/FlowConformal/experiments/exp_ablation/outputs
```

## Regenerating everything

```bash
# Pass over a real-results directory (--csv-dir required):
python examples/FlowConformal/paper/regenerate_all.py --csv-dir <path>

# List discovered scripts (no execution):
python examples/FlowConformal/paper/regenerate_all.py --list

# Skip the heavyweight banana visualization (trains a flow):
python examples/FlowConformal/paper/regenerate_all.py --csv-dir <path> \
    --skip fig7_banana_score_geometries.py
```

The runner auto-discovers any new ``tab*.py`` / ``fig*.py`` placed in
``tables/`` / ``figures/`` and aborts if any one of them fails.

## Real-data paths

When the real experiments land, point each script at the canonical
output directory documented in [`../CSV_SCHEMAS.md`](../CSV_SCHEMAS.md):

| Script                                 | Real `--csv-dir`                                                |
|----------------------------------------|-----------------------------------------------------------------|
| tab1, fig2, tab_exp1_runtime (Exp 1)   | `examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs`|
| tab2, fig3, tab_exp2_runtime (Exp 2)   | `examples/FlowConformal/experiments/exp2_prob_scale/outputs`    |
| fig4(a), fig5(a/b), fig6(a/b/c), tab3, | `examples/FlowConformal/experiments/exp_ablation/outputs`       |
| tab_score_vs_dim, tab_amls_hparam,     |                                                                 |
| tab_conformal_params, tab_flow_training|                                                                 |

`fig1_flow_training_progression.py` is a thin wrapper around the
existing `flow_matching_training/fig_training_progression.py` script.
By default it copies the cached `overlay.png`; pass `--regenerate` to
re-train the snapshot flows from scratch (slow; needs torch).

`fig7_banana_score_geometries.py` is standalone — no CSV input. It
trains a small flow on `RotatedBananaNet` outputs every run (~30 s on
CPU). Pass `--epochs N` to override the default 100.

## LaTeX integration

The `.tex` outputs are self-contained `\begin{table}...\end{table}`
fragments. They depend on a small set of LaTeX packages:

- ``booktabs`` — toprule / midrule / bottomrule (every table)
- ``multirow`` — `\multirow{N}{*}{...}` cells (tab1, tab2,
  tab_conformal_params)
- ``rotating`` — `\rotatebox{60}{...}` column headers (tab_exp1_runtime,
  tab_exp2_runtime)

Either:

- `\input{tables/tab1_exp1_verdict_matrix.tex}` directly into your
  paper, or
- compile a quick standalone preview:

  ```latex
  \documentclass{article}
  \usepackage{booktabs}
  \usepackage{multirow}
  \usepackage{rotating}
  \begin{document}
  \input{tab1_exp1_verdict_matrix.tex}
  \end{document}
  ```

  All tables compile cleanly with `pdflatex` (one harmless "overfull
  hbox" warning on Table 1 — fix later by tightening column widths).

## Saved-data audit

CSV outputs of `run_verification_pipeline` now include the
``amls_levels_used`` column (number of adaptive AMLS levels actually
run; blank when verification_method != 'amls'). Updated:

- `examples/FlowConformal/ablations/acasxu_sweep.py`
- `examples/FlowConformal/experiments/exp1_vnncomp_subset/_common.py`
- `examples/FlowConformal/experiments/exp2_prob_scale/_common.py`
- `examples/FlowConformal/experiments/exp3_synthetic/exp3_run_*.py`
- `examples/FlowConformal/experiments/exp_ablation/_common.py`
- `examples/FlowConformal/CSV_SCHEMAS.md`

The underlying value comes from `AMLSResult.levels_used` — already
exposed in the `run_verification_pipeline` result dict at key
``amls_levels_used``.

## Style conventions

- **Colour scheme** (`_common.METHOD_COLORS`):
  ours = green (`#1b9e3a`), αβ-CROWN = orange-red (`#e6550d`),
  NeuralSAT = red, PyRAT = dark red-brown, CORA = light orange,
  Hashemi = mid-blue, RS = light blue, SaVer = purple,
  ProbStar = teal-blue.
- **Bold rows** in LaTeX tables denote *ours*; **italic rows** denote
  sound (read-only) verifiers.
- **Em-dashes (`---`)** in tables mark tool/benchmark combinations
  that are NOT_APPLICABLE (e.g. RS / SaVer / Hashemi-clipping on
  ACAS Xu, where the spec is not classification-robustness).
- **Indeterminate column** in Tables 1/2 aggregates UNKNOWN, TIMEOUT,
  ERROR, NOT_APPLICABLE, SKIPPED.
- **Log-y** is used only when dynamic range > 50× (figs 2, 3, 5b).
- **Linear-y** for verdict counts, percentages, and figs 5a, 4a.
- **Sans-serif disabled** — figures use Computer-Modern serif via
  `apply_paper_style()`.

## Pick-one decisions for the user

The following pairs are produced as alternatives so the user can pick
one before submission:

| What                | Figure version                       | Table version              |
|---------------------|--------------------------------------|----------------------------|
| Score × dim         | `fig4_score_vs_dim` (log-y), `fig4a_score_vs_dim_linear` | `tab_score_vs_dim` |
| Scaling             | `fig5a_scaling_linear`, `fig5b_scaling_semilog` (replaces `fig5_scaling`) | — |
| Exp 1 runtime       | `fig2_exp1_runtime`                  | `tab_exp1_runtime`         |
| Exp 2 runtime       | `fig3_exp2_runtime`                  | `tab_exp2_runtime`         |
| Ablations (3 panels)| `fig6a/b/c_ablation_*` (split)       | `tab_amls_hparam`, `tab_conformal_params`, `tab_flow_training` |
