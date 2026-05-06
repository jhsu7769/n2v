# Exp 2: Probabilistic-Scale Comparison

Run flow-conformal AMLS on four large image-classification benchmarks
where sound verifiers are expected to TIMEOUT, and compare against
αβ-CROWN (computed by us, not VNN-COMP results since some configs
differ), Hashemi-clipping, and randomized smoothing (RS).

## Roster (4 benchmarks)

All four are multi-class image classification with cora-style nested
OR specs (`list[1 dict]` + `Hg = list[K HalfSpaces]`). Ours uses a
**uniform config** across all four — `mega + amls_bounded_union +
amls_max_levels=30` — so any cross-benchmark differences come from
the network/spec, not the verifier setup.

| short name           | spec disjuncts | ONNX size | Output dim | Per-row budget | Source |
|----------------------|----------------|-----------|------------|----------------|--------|
| `vit_2023`           | 9 (10-class)   | 0.3 MB    | 10         | 100 s (VNN-COMP) | VNN-COMP 2023 |
| `cifar100_2024`      | 99 (100-class) | 9.7 MB    | 100        | 100 s (VNN-COMP) | VNN-COMP 2024 |
| `tinyimagenet_2024`  | 199 (200-class)| 13.8 MB   | 200        | 100 s (VNN-COMP) | VNN-COMP 2024 |
| `cifar10_resnet110`  | 9 (10-class)   | 7 MB      | 10         | 300 s (no VNN-COMP equivalent) | Cohen-RS pretrained ResNet-110 |

`yolo_2023` was previously here but dropped — special-track YOLO
benchmark with object-detection spec, only 39 instances, and old
probe data showed multiple false-UNSAT rows. Replaced by
`tinyimagenet_2024`: 200 instances, regular VNN-COMP track,
ResNet-medium architecture (matches cifar100), same 100s budget.

## Files

```
_benchmarks.py                    PER_BENCHMARK_CONFIG, deferred loaders, VNN-COMP path lookup
exp2_run_ours.py                  ours runner, --benchmark X
exp2_run_hashemi_clipping.py      Hashemi-clipping baseline (m=8000)
exp2_run_alpha_beta_crown.py      αβ-CROWN via subprocess
exp2_run_rs.py                    Cohen RS (image classification only)
exp2_run_cifar10_resnet110.py     Python-side ResNet-110 loader (used by ours/Hashemi/RS)
build_resnet110_onnx.py           one-shot ONNX export + 73 vnnlib specs for αβ-CROWN
cifar10_resnet110_vnncomp/        locally-built benchmark dir for αβ-CROWN
exp2_aggregate.py                 reads ground_truth.csv + per-(benchmark, method) CSVs
exp2_soundness_audit.py           AutoAttack + 5K-restart PGD on UNSAT verdicts
ground_truth.csv                  pre-computed SAT-wins consensus from VNN-COMP 2025 (8 tools)
outputs/                          per-(benchmark, method) CSVs land here
```

## Smoke

```bash
PY=/home/sasakis/miniconda3/envs/n2v/bin/python
cd /home/sasakis/v/tools/n2v
for bench in vit_2023 tinyimagenet_2024 cifar100_2024 cifar10_resnet110; do
  for tool in ours hashemi_clipping; do
    $PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_$tool \
        --benchmark $bench --smoke
  done
done
# αβ-CROWN smokes (subprocess to alpha-beta-crown conda env)
for bench in vit_2023 tinyimagenet_2024 cifar100_2024 cifar10_resnet110; do
  $PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_alpha_beta_crown \
      --benchmark $bench --smoke
done
# RS smokes (image classification only)
for bench in cifar10_resnet110 cifar100_2024; do
  $PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_rs \
      --benchmark $bench --smoke
done
```

## Full sweep

Use the project-level launcher (per-instance shell timeouts via
`run_cell.sh`):

```bash
bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 2
```

## Per-benchmark prerequisites

### `cifar10_resnet110`
The Cohen-RS pretrained ResNet-110 expected at:
```
~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/checkpoint.pth.tar
```
where `<sigma>` defaults to `0.25`. The script gracefully exits with a
TODO message if the checkpoint is absent. CIFAR-10 test data downloads
to `~/.cache/n2v_exp2_cifar10/` on first run.

For αβ-CROWN to ingest cifar10_resnet110, you need to run
`build_resnet110_onnx.py` once to produce
`cifar10_resnet110_vnncomp/onnx/cifar10_resnet110.onnx` and 73 vnnlib
specs (instances where the model misclassifies the clean input are
skipped — RS robustness undefined). Validation: ONNX↔PyTorch
numerical equivalence (atol=1e-5) and vnnlib round-trip parsing.

## Aggregation

```bash
$PY -m examples.FlowConformal.experiments.exp2_prob_scale.exp2_aggregate
```

Reads every `outputs/exp2_<bench>_<method>.csv`, joins against
[`ground_truth.csv`](ground_truth.csv) (which uses the SAT-wins rule
across 8 VNN-COMP 2025 sound verifiers), and writes
`outputs/exp2_comparison_table.csv` with verdict counts, p10/median/p90
wall-clock, false-UNSAT/false-SAT counts, and percentages. The
`cifar10_resnet110` rows emit `ground_truth_source='not_applicable_resnet110'`
and skip false-UNSAT/SAT counting (no external reference).

## Soundness audit (Phase 6)

The soundness audit is scoped to **only `cifar10_resnet110`** —
the single Exp 2 benchmark with no external sound-verifier reference.
For the other three benchmarks, the VNN-COMP 8-tool consensus is the
reference and an additional cex search would only confirm what
ground_truth.csv already says.

```bash
$PY -m examples.FlowConformal.experiments.run_soundness_audit \
    --benchmark cifar10_resnet110 \
    --input-csv outputs/exp2_cifar10_resnet110_ours.csv \
    --output-csv outputs/exp2_cifar10_resnet110_audit_ours.csv
```

## Sound-verifier expectations

For the paper's "no sound verifier scales here" claim:

| benchmark | sound-verifier expectation |
|---|---|
| `vit_2023` | αβ-CROWN solves a small fraction; most UNKNOWN/TIMEOUT in VNN-COMP 2023 |
| `cifar100_2024` | αβ-CROWN solves some at 100 s budget; ResNet-medium pushes against the budget |
| `tinyimagenet_2024` | αβ-CROWN solves few; 200-class disjunctive spec amplifies search cost |
| `cifar10_resnet110` | αβ-CROWN errors on every instance — see special-case note below |

## Special case: `cifar10_resnet110` αβ-CROWN

αβ-CROWN cannot verify the 110-layer Cohen-RS ResNet on standard
verification hardware (24 GB-class GPU). We exhausted both code paths:

* **ONNX path** (4 configs tried: vnncomp21 cifar10-resnet, vnncomp24
  cifar100, vnncomp22 resnet_A, custom): all error with shape
  mismatches in `auto_LiRPA`'s `add_b()` bound concretization step.
* **PyTorch-native path** (registered `cifar_resnet110_cohen_rs` in
  αβ-CROWN's `model_defs.py`, ran with batch_size=32 +
  `share_alphas: true` + `expandable_segments`): parses the
  architecture cleanly but exhausts GPU memory at intermediate-bound
  computation, **before any branch-and-bound starts**.

αβ-CROWN successfully verifies smaller ResNets in the same setup, so
the issue is depth-driven memory growth, not a tooling defect. The
production sweep keeps the ONNX path with `vnncomp22/resnet_A.yaml`
(the closest architectural match) as the best-effort attempt. The
ERROR result for all 100 instances is the defensible "doesn't scale"
data point — exactly the design intent of Cohen et al. (2019).

Full audit trail: [`.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md`](../../../../.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md).
Reproducer: [`abcrown_resnet110_pytorch_native.py`](abcrown_resnet110_pytorch_native.py).
