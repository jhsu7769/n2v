# FlowConformal — flow-matching-based probabilistic reachability

Runnable entry points for the flow-matching probabilistic-reachability project. The library code (flow training, calibration, AMLS / scenario / IS / Langevin spec verification, falsifier ensemble) lives in `n2v/probabilistic/flow/`, `n2v/probabilistic/verify_flow.py`, and `n2v/utils/falsify.py`. This directory holds the demos, benchmarks, ablations, paper-experiment scripts, smokes, and visualization tooling.

For research context see [`.claude/research/flow-matching-probabilistic-reach/`](../../.claude/research/flow-matching-probabilistic-reach/) (project description, post-Phase-5d reference, paper plan pointer). For the current paper-experiment design see [`docs/plans/2026-04-27-paper-experiments-design.md`](../../docs/plans/2026-04-27-paper-experiments-design.md).

## Public library API

```python
from n2v.probabilistic import verify_flow

result = verify_flow(
    network=net, input_lb=lb, input_ub=ub, spec=spec,
    alpha=0.001, m=8000, ell=7999,
    scenario_n_samples=2000, scenario_beta=0.001,
    flow_config='base',
    verification_method='amls_bounded',  # Phase 5e production default; library default still 'scenario' for back-compat
    use_falsifier=False,                 # opt-in Stage 1 falsifier
    seed=0,
)
```

`use_falsifier=False` is the library default: the pipeline runs flow + conformal + spec verification only and returns UNSAT or UNKNOWN. With `use_falsifier=True` the pipeline first runs an ensemble of random + PGD + APGD attacks; if any restart finds an `x` with `f(x) ∈ Unsafe`, the verdict is SAT with that concrete counterexample.

The ACAS Xu sweep harness (`examples/FlowConformal/benchmarks/_common.py`) wraps `verify_flow` and defaults `use_falsifier=True` for backward compatibility with the legacy benchmark scripts.

A minimal demo of the public API: [`use_n2v_verify_flow.py`](use_n2v_verify_flow.py).

## Directory layout

```
FlowConformal/
├── networks.py                     toy networks (RotatedBananaNet, ThreeBlob{2D,3D})
├── use_n2v_verify_flow.py          public-API demo (Plan B entry point)
│
├── benchmarks/                     small benchmarks + the ACAS Xu test harness
│   ├── _common.py                  back-compat wrapper around verify_flow (use_falsifier=True default)
│   ├── _common_analytical.py       analytical-ground-truth runner (rotated cube)
│   ├── _spec.py                    spec helpers shared with experiments/
│   ├── test_acasxu_single.py       single-instance ACAS Xu runner
│   ├── test_banana.py              2D curved strip
│   ├── test_three_blob_3d.py       3D multimodal classifier
│   ├── test_rotated_linear.py      2D + 3D rotated cube (analytical)
│   ├── test_rotated_linear_production.py
│   └── test_identity_network.py    cube sanity check
│
├── ablations/                      sweep runners + per-phase outputs
│   ├── acasxu_sweep.py             canonical full-sweep driver (Phase 5d: AMLS; Phase 5e: bounded AMLS + falsifier)
│   ├── acasxu_*_diag.py            per-instance diagnostic probes (false-UNSAT, scaling, output dist)
│   ├── acasxu_volume_validation.py
│   ├── phase5c_a1_variance_probe.py + phase5c_a1_analyze.py
│   ├── phase5c_probe_sweep.py      20-instance probe used to gate AMLS
│   ├── multi_seed_three_blob.py    PoC 6-config × 3-seed flow training sweep
│   ├── sweep_logdensity_vs_naive.py
│   ├── sweep_three_blob_{enhanced,training}.py
│   └── outputs/                    CSVs (per-phase: _phase4, _phase5b, _phase5c, _phase5c probe)
│
├── experiments/                    paper-quality runs
│   ├── README.md                   master design doc + execution order
│   ├── run_all_sweeps.sh           priority-ordered phase launcher
│   ├── run_cell.sh                 VNN-COMP-style per-instance shell-timeout wrapper
│   ├── run_soundness_audit.py      AutoAttack + 5K-restart PGD post-hoc audit
│   ├── build_ground_truth.py       one-shot SAT-wins consensus generator (Exp 1 + Exp 2)
│   ├── _external_verifiers.py      αβ-CROWN / NeuralSAT subprocess wrappers
│   ├── baselines/                  shared probabilistic-baseline helpers
│   │   ├── run_hashemi_naive.py / run_hashemi_clipping.py
│   │   ├── run_rs.py               Cohen et al. randomized smoothing
│   │   ├── run_saver.py            SAVER (Convertino et al.)
│   │   ├── run_probstar.py         ProbStar / StarV (Tran et al.)
│   │   └── _common.py / README.md
│   ├── exp1_vnncomp_subset/        Exp 1 — sound-verifier comparison (7 benchmarks)
│   │   ├── exp1_run_ours.py                  ours (bounded AMLS), --benchmark X
│   │   ├── exp1_run_hashemi_clipping.py      Hashemi-clipping baseline
│   │   ├── exp1_aggregate.py                 builds exp1_comparison_table.csv
│   │   ├── ground_truth.csv                  pre-computed SAT-wins consensus
│   │   ├── _benchmarks.py / _common.py / outputs/ / README.md
│   ├── exp2_prob_scale/            Exp 2 — probabilistic-scale comparison
│   │   ├── exp2_run_ours.py                  ours, --benchmark X
│   │   ├── exp2_run_hashemi_clipping.py
│   │   ├── exp2_run_alpha_beta_crown.py      αβ-CROWN via subprocess
│   │   ├── exp2_run_rs.py                    Cohen RS (image classification only)
│   │   ├── exp2_run_cifar10_resnet110.py     Python-side ResNet-110 loader
│   │   ├── build_resnet110_onnx.py           ONNX + 73 vnnlib spec builder for cifar10_resnet110
│   │   ├── cifar10_resnet110_vnncomp/        locally-generated benchmark dir
│   │   ├── exp2_aggregate.py / exp2_soundness_audit.py
│   │   ├── _benchmarks.py / ground_truth.csv / outputs/ / README.md
│   ├── exp3_synthetic/             Exp 3 — synthetic geometric validation (4 score families)
│   │   ├── exp3_run_ours.py                  ours, --benchmark/--score/--spec
│   │   ├── exp3_run_hashemi_clipping.py
│   │   ├── _score_pipeline.py                hyperrect + ellipsoid + naive-GMM + flow scores
│   │   ├── exp3_aggregate.py / _benchmarks.py / outputs/ / README.md
│   ├── exp4_scaling/               Exp 4 — controlled depth-scaling on 1-Lipschitz family
│   │   ├── exp4_run_{ours,hashemi_clipping,alpha_beta_crown,neuralsat}.py
│   │   ├── networks.py / instance_generator.py / _benchmarks.py
│   │   ├── exp4_aggregate.py / outputs/ / README.md
│   ├── exp_ablation/               9 ablation rows on ACAS Xu
│   │   ├── ablation_run_{score,verify_method,amls_hparam,calib_size,flow_training}.py
│   │   ├── ablation_aggregate.py / _common.py / outputs/ / README.md
│   └── legacy/                     superseded per-benchmark thin wrappers and v1 aggregators
│

├── smokes/                         fast correctness checks
│   ├── smoke_logdensity_gaussian.py         sign-error canary (PoC era)
│   ├── verify_exact_caches.py               cross-check cached Star-union volumes
│   └── fm_validation/                       two_moons, 8_gaussians, checkerboard
│
├── viz/                            visualization demos + figure HTMLs
│   ├── star_viz_demo.py
│   ├── flow_viz_demo_3d.py                  flow reachset vs Star-union overlay
│   ├── figures_star_union/                  Plotly HTMLs for Star union
│   └── figures_flow_reachset/               Plotly HTMLs for flow reachset
│
├── figures/                        paper-figure generators + their outputs
│   └── flow_matching_training/              training-progression overlay
│
└── utils/                          shared benchmark helpers (not library)
    └── reach.py                    compute_exact_reach wrapper around n2v.nn.NeuralNetwork
```

The `_archive/` subtree under [`.claude/research/flow-matching-probabilistic-reach/`](../../.claude/research/flow-matching-probabilistic-reach/) holds superseded designs, older audits, and 12+ exploratory experiment attempts — useful for design rationale, not for current behavior.

## How to run

All commands assume the project's conda env:

```bash
CONDA=/home/sasakis/miniconda3/envs/n2v/bin/python
```

A single benchmark (~3 min):

```bash
$CONDA -m examples.FlowConformal.benchmarks.test_banana
$CONDA -m examples.FlowConformal.benchmarks.test_three_blob_3d
$CONDA -m examples.FlowConformal.benchmarks.test_acasxu_single
```

The full ACAS Xu sweep (~95-200 min depending on AMLS settings; best run with `nohup`):

```bash
cd /home/sasakis/v/tools/n2v
nohup $CONDA -u -m examples.FlowConformal.ablations.acasxu_sweep \
    > /tmp/acasxu_sweep.log 2>&1 &
disown
tail -f /tmp/acasxu_sweep.log
```

A 20-instance probe (faster signal, ~12-25 min):

```bash
$CONDA -m examples.FlowConformal.ablations.phase5c_probe_sweep
```

A paper experiment, e.g.:

```bash
$CONDA -m examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_collins_rul_cnn
$CONDA -m examples.FlowConformal.experiments.exp_ablation.ablation_run_verify_method
$CONDA -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_3d_banana
```

A baseline (probabilistic comparison):

```bash
$CONDA -m examples.FlowConformal.experiments.baselines.run_hashemi_naive
$CONDA -m examples.FlowConformal.experiments.baselines.run_rs
$CONDA -m examples.FlowConformal.experiments.baselines.run_saver
$CONDA -m examples.FlowConformal.experiments.baselines.run_probstar
```

A smoke (~30 s):

```bash
$CONDA -m examples.FlowConformal.smokes.smoke_logdensity_gaussian
```

A viz demo (~3-5 min, writes HTMLs to `viz/figures_*/`):

```bash
$CONDA -m examples.FlowConformal.viz.flow_viz_demo_3d
```

## Test suite

Fast per-module subset while iterating (~20 s):

```bash
$CONDA -m pytest \
    tests/unit/probabilistic/flow/test_amls.py \
    tests/unit/probabilistic/flow/test_scores.py \
    tests/unit/probabilistic/flow/test_sets.py \
    tests/unit/probabilistic/flow/test_calibrate.py \
    tests/unit/probabilistic/flow/test_scenario_verify.py \
    -q --tb=short
```

Full fast suite (~1 min, skips training-heavy slow-marked tests):

```bash
$CONDA -m pytest tests/unit/probabilistic/flow/ -m "not slow" -q
```

Full suite including slow tests (~5 min; trains flows on Gaussian targets):

```bash
$CONDA -m pytest tests/unit/probabilistic/flow/ -q
```

## Current results headline

Full ACAS Xu sweep (186 instances, VNN-COMP 2025, Phase 5d):

| metric | Phase 4 (single-lane) | Phase 5b (two-lane scenario) | **Phase 5d (two-lane AMLS)** |
|---|---:|---:|---:|
| false UNSATs | 5 | 4 | **0** |
| UNSAT / SAT / UNKNOWN | — | — | 132 / 38 / 16 |
| agreement with α,β-CROWN | — | — | 91.4% |
| wall-clock (full sweep) | ~80 min | ~94 min | 164 min |

Historical sweep CSVs (Phase 5d/5e) archived to
`.claude/research/flow-matching-probabilistic-reach/_archive/acasxu_phase5_sweeps/`.
Current canonical ACAS Xu numbers come from the Phase 1 cell
`exp1_acasxu_2023_ours.csv` produced by `run_all_sweeps.sh`.

## Setup blocked on user

Some baselines and Exp 2 paths require licenses or pre-trained weights:

- ✅ Gurobi WLS academic license + `gurobipy` 13.0.1 — installed in `n2v` conda env (2026-04-28); `run_probstar.py` ready.
- ✅ Cohen RS pretrained ResNet-110 weights — downloaded.
- `pip install autoattack timm` — still needed for Exp 2 soundness audit and ViT-Small CIFAR-10 baselines.

## Default config knobs

The Phase 5e ACAS Xu defaults live in `examples/FlowConformal/ablations/acasxu_sweep.py` and the per-experiment `_common.py` files. Key knobs:

```python
# Stage-2 verification (Phase 5e)
verification_method        = 'amls_bounded'   # replaces unbounded 'amls' from Phase 5d
amls_bounded_eps_2_target  = 0.001            # default = alpha; joint mult. bound 1-(1-α)(1-ε_2) ≈ 2α
amls_bounded_adaptive_step = False            # optional q/sqrt(d) MCMC step scaling for high-dim outputs
m, ell, alpha              = 8000, 7999, 0.001
scenario_n_samples         = 2000             # AMLS samples per level
scenario_beta              = 0.001            # AMLS asymptotic-CI failure prob

# Flow training (current ACAS Xu defaults)
flow_config = 'base'     # see verify_flow for the named-config table
n_train     = 10000      # Phase 5e mega tier; was 5000 in Phase 5d
flow_epochs = 2000

# Falsifier (Stage 1, opt-in)
use_falsifier = True     # set by the benchmarks shim; verify_flow itself defaults to False
```

For Phase 5e per-benchmark hparam recommendations from the lock-in probe see [`probes/outputs/probe_amls_bounded_lock.csv`](probes/outputs/probe_amls_bounded_lock.csv) once the run completes.

The PoC-era "tight" 3D config (h256/L6, 5000 ep, sinusoidal time, OT-CFM+Sinkhorn) is preserved in `ablations/sweep_three_blob_enhanced.py` and `multi_seed_three_blob.py` for reproducing the 4-benchmark synthetic table; it is not the production config for ACAS Xu.

## Phase 5e diagnostic + design (post-Phase 5d work)

Empirical probes (probe v2 = 363 cells, AMLS-budget probe = 75 cells, ACAS Xu bounded smoke = 10 instances) found that unbounded AMLS over-rejects on benchmarks where the conformal reach set is genuinely disjoint from unsafe but the flow assigns small tail mass outside the calibrated ball. **Bounded AMLS** restricts the rare-event search to `||z|| <= q`, giving the right verdict on tllverify (UNSAT, margin +20) where unbounded AMLS gave UNKNOWN. Implementation in `n2v/probabilistic/flow/amls_bounded.py`; design and soundness argument at [`docs/research/2026-04-28-bounded-amls-design.md`](../../docs/research/2026-04-28-bounded-amls-design.md). Diagnostic scripts that motivated the change: `probes/diag_tllverify_*.py`.

The Phase 5d unbounded-AMLS code path remains intact (`verification_method='amls'`) for the Phase 5d-vs-5e ablation row in the paper.
