# Reproducing the FlowConformal experiments

End-to-end recipe to reproduce every paper experiment from a clean
checkout, including conda-env creation, pretrained-weight setup,
smoke validation, full sweep execution, and figure generation.

**Audience:** anyone re-running the experiments — the paper authors on
a fresh machine, reviewers, or future maintainers. Assume a Linux box
with an NVIDIA GPU (CUDA 12.x driver) and conda installed.

**Total wall time:** roughly **75-95 hours of GPU compute** for the
full sweep (Exp 1 → Exp 4), plus ~3-5 hours for the soundness audit,
~7-8 hours for ablation studies, and ~8-12 hours for Tier-B (Exp 2
ProbStar + SaVer in Phase 8). Sequential; multi-GPU parallelisation
across phases gets it down to ~3 calendar days.

---

## 0. Prerequisites

```bash
# clone + cd
git clone <repo-url> n2v && cd n2v

# system libs (Ubuntu/Debian); root only.
sudo apt install libglpk-dev libgmp3-dev    # for ProbStar/StarV (Tier-B)
```

GPU driver must be **≥ 525.60** (CUDA 12.1+). Verify with `nvidia-smi`.

---

## 1. Conda environments

We use one env per tool to avoid dep conflicts (each verifier has
incompatible torch/numpy pins).

| env | python | torch | role |
|---|---|---|---|
| `n2v` | 3.12 | 2.7.x+cu118 | our core method, Hashemi, RS, AutoAttack, **SaVer** (imported in-process from `~/v/other/SaVer-Toolbox`) |
| `alpha-beta-crown` | 3.11 | 2.7.1+cu118 | αβ-CROWN |
| `neuralsat` | 3.10 | 2.1.2+cu118 | NeuralSAT |
| `starv` | 3.10 | 2.7.1+cu118 (numpy<=1.26.4) | ProbStar (Tier-B); subprocess-dispatched because StarV's numpy<=1.26.4 pin is incompatible with n2v |

### 1a. n2v env

```bash
# Existing env that the project uses; create or activate it.
# (If creating fresh, install n2v's own requirements first; see repo root.)
~/miniconda3/envs/n2v/bin/pip install git+https://github.com/fra31/auto-attack.git
~/miniconda3/envs/n2v/bin/pip install timm   # if not already
```

### 1b. αβ-CROWN

```bash
git clone https://github.com/Verified-Intelligence/alpha-beta-CROWN.git \
    ~/v/other/alpha-beta-CROWN
cd ~/v/other/alpha-beta-CROWN
conda env create -f complete_verifier/environment.yaml --name alpha-beta-crown
# Their pinned torch is cu128; downgrade to cu118 for older drivers:
~/miniconda3/envs/alpha-beta-crown/bin/pip install --force-reinstall \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118
cd -
```

Quick smoke (should print `Result: unsat` in <30s):
```bash
~/miniconda3/envs/alpha-beta-crown/bin/python \
    ~/v/other/alpha-beta-CROWN/complete_verifier/abcrown.py \
    --config ~/v/other/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp21/acasxu.yaml \
    --onnx_path ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
    --vnnlib_path ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib
```

### 1c. NeuralSAT

```bash
git clone https://github.com/dynaroars/neuralsat.git ~/v/other/neuralsat
conda create -n neuralsat python=3.10 -c conda-forge -y
~/miniconda3/envs/neuralsat/bin/pip install \
    torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
conda install -n neuralsat -c gurobi gurobi -y
~/miniconda3/envs/neuralsat/bin/pip install -r ~/v/other/neuralsat/requirements.txt
~/miniconda3/envs/neuralsat/bin/pip install "setuptools<80"   # so torch.utils.cpp_extension imports
```

Quick smoke (should print `unsat,<small>` on the last line):
```bash
cd ~/v/other/neuralsat && \
~/miniconda3/envs/neuralsat/bin/python src/main.py \
    --net ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/onnx/ACASXU_run2a_1_1_batch_2000.onnx \
    --spec ~/v/other/VNNCOMP/vnncomp2025_benchmarks/benchmarks/acasxu_2023/vnnlib/prop_1.vnnlib \
    --device cuda --timeout 60
```

### 1d. ProbStar / SaVer (Tier-B; defer if not running Phase 1 SaVer cells or Phase 8)

```bash
# StarV / ProbStar (subprocess-dispatched into its own env)
conda create -n starv python=3.10 -c conda-forge -y
~/miniconda3/envs/starv/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118
conda install -n starv -c gurobi gurobi -y
~/miniconda3/envs/starv/bin/pip install -e ~/v/other/StarV
~/miniconda3/envs/starv/bin/pip install "numpy<=1.26.4"

# SaVer (no separate env — runs in-process from n2v env via a sys.path
# insert). Just clone SaVer-Toolbox to ~/v/other/SaVer-Toolbox; the
# baseline runner picks it up automatically.
git clone https://github.com/<saver-toolbox-fork> ~/v/other/SaVer-Toolbox
```

### 1e. Cohen RS pretrained weights (for Exp 2 cifar10_resnet110)

```bash
# Follow Cohen et al. 2019 instructions to download checkpoint.pth.tar
# into ~/v/other/smoothing/models/cifar10/resnet110/noise_<sigma>/
# See https://github.com/locuslab/smoothing for the gdrive link.
```

---

## 2. VNN-COMP benchmark + results data

Clone the VNN-COMP 2025 repos under `~/v/other/VNNCOMP/`:

```bash
mkdir -p ~/v/other/VNNCOMP
cd ~/v/other/VNNCOMP
git clone https://github.com/stanleybak/vnncomp2025_benchmarks.git
git clone https://github.com/<vnncomp25-results>.git vnncomp2025_results
```

The benchmark dir layout we expect:
* `vnncomp2025_benchmarks/benchmarks/<bench>/onnx/*.onnx`
* `vnncomp2025_benchmarks/benchmarks/<bench>/vnnlib/*.vnnlib`
* `vnncomp2025_benchmarks/benchmarks/<bench>/instances.csv`
* `vnncomp2025_results/<tool>/results.csv`  (consensus ground truth)

---

## 3. Pre-flight smoke

Before committing ~80 GPU-hours to the full sweep, run every per-tool
smoke to catch broken installs / config mismatches:

```bash
cd /home/sasakis/v/tools/n2v
bash examples/FlowConformal/experiments/run_all_sweeps.sh --smoke --dry-run
```

The `--dry-run` just prints the full command list without executing.
Drop `--dry-run` to actually run the smokes (~30-60 min total, 1
instance per cell).

Each cell's smoke writes an `outputs/exp<N>_<bench>_<tool>.csv` with
exactly 1 row, and prints `[smoke] PASS` or `[smoke] FAIL`. Failures
need investigation before the full sweep.

---

## 4. Pre-sweep config-validation probes (recommended)

The two new Exp 1 additions (`malbeware`, `metaroom_2023`) and
Exp 2's disjunctive cifar100 spec are validated by the lock probe +
a one-shot runner smoke; rerun them if the underlying VNN-COMP
results or the flow-config defaults change:

```bash
# Lock probe for the new Exp 1 additions (5 instances each)
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.probes.probe_amls_bounded_lock \
    --benchmarks malbeware,metaroom_2023 \
    --output-csv /tmp/probe_lock_new_exp1.csv \
    --instances-per-cell 5

# cifar100_2024 union-AMLS smoke (1 instance via the runner --smoke)
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_ours \
    --benchmark cifar100_2024 --smoke

# Exp 4 graceful TIMEOUT at D=24
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp4_scaling.exp4_run_ours \
    --depth 24 --smoke
```

---

## 5. Run experiments via the master launcher

Sequential execution of all 8 phases (~75-95 hr total):

```bash
bash examples/FlowConformal/experiments/run_all_sweeps.sh
```

Phase-by-phase (recommended for parallelisation across GPUs / nights):

```bash
# Phase 1: Exp 1 — 21 cells = (ours + Hashemi-clipping + SaVer) × 7 benchmarks.
# ~10-12 hours. SaVer (Tier-B) emits UNKNOWN on most instances (its DKW
# bound is loose at the 1e-3 budget) but fills out the comparison row.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 1 \
    > /tmp/exp1.log 2>&1 &

# Phase 2: Exp 2 Tier-A — probabilistic-scale (ours + Hashemi + αβ-CROWN × 4 benchmarks).
# RS is moved to phase 5 (it only applies to image-classification, and
# is logically a smoothing-only baseline). ~17 hours.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 2 \
    > /tmp/exp2.log 2>&1 &

# Phase 3: Exp 3 — synthetic geometry validation (4 score families × 2 spec types × 4 benchmarks).
# ~8 hours.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 3 \
    > /tmp/exp3.log 2>&1 &

# Phase 4: Exp 4 — controlled scaling (4 methods × 7 depths).
# ~14 hours.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 4 \
    > /tmp/exp4.log 2>&1 &

# Phase 5: Exp 2 RS — Cohen smoothing on cifar10_resnet110 + cifar100_2024.
# ~5 hours.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 5 \
    > /tmp/exp2_rs.log 2>&1 &

# Phase 6: Soundness audit (AutoAttack + 5K-PGD on every UNSAT row).
# ~3-5 hours.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 6 \
    > /tmp/audit.log 2>&1 &

# Phase 7: Ablations (5 axes × ~7-8 hr total).
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 7 \
    > /tmp/ablations.log 2>&1 &

# Phase 8: Exp 2 Tier-B (ProbStar + SaVer × 4 benchmarks). ~8-12 hours.
# ProbStar uniformly emits NOT_APPLICABLE on Exp 2's networks (StarV's
# loader doesn't support transformer attention, residual Add, or Gemm
# nodes — vit_2023, cifar100_2024, tinyimagenet_2024, cifar10_resnet110
# all hit one of these). The runner records the NOT_APPLICABLE rows
# either way to substantiate the gap claim in the paper.
nohup bash examples/FlowConformal/experiments/run_all_sweeps.sh --phase 8 \
    > /tmp/exp2_tier_b.log 2>&1 &
```

---

## 6. Aggregate per-instance CSVs into comparison tables

After each phase finishes, run the matching aggregator to produce the
`*_comparison_table.csv` / `*_scaling_summary.csv` that figure scripts
consume:

```bash
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_aggregate
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp2_prob_scale.exp2_aggregate
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_aggregate
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.exp4_scaling.exp4_aggregate
```

The launcher's `aggregate_all` step does this automatically for the
all-phases run.

---

## 7. Soundness audit

The soundness audit is **scoped to `cifar10_resnet110` only** — the
single Exp 2 benchmark with no external sound-verifier reference (it
isn't a VNN-COMP benchmark). For all other benchmarks, the
ground-truth file (built from VNN-COMP 2025's 8-tool consensus, see
`build_ground_truth.py`) is the reference and an additional cex
search would only confirm what `ground_truth.csv` already says.

```bash
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.experiments.run_soundness_audit \
    --benchmark cifar10_resnet110 \
    --input-csv examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_cifar10_resnet110_ours.csv \
    --output-csv examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_cifar10_resnet110_audit_ours.csv
```

`run_all_sweeps.sh --phase 5` runs this against ours, hashemi_clipping,
and αβ-CROWN's CSVs (no point auditing αβ-CROWN since it produces
all-ERROR — see special-case note below — but the run is harmless).
Any `found_cex=1` row in an audit CSV is a soundness violation that
needs paper-level attention.

### Special-case sound-verifier behaviors

These outcomes are **expected** (they correspond to design intent of
the experiments) but require careful framing in the paper. Full audit
trail and reproducible scripts at
[`.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md`](../../.claude/research/flow-matching-probabilistic-reach/sound-verifier-limitations.md):

* **Exp 4 αβ-CROWN at d ≥ 24:** real exponential BaB-tree blowup
  (worst-case bound stalls, open-domain count never drops, more time
  wouldn't help) → TIMEOUT.
* **Exp 4 NeuralSAT at d ≥ 16:** abstractor too loose, gives up
  before BaB → UNKNOWN with constant ~17 s wall regardless of depth.
* **Exp 2 cifar10_resnet110 αβ-CROWN:** GPU-memory-exhausted bound
  propagation regardless of config (4 ONNX configs + 1
  PyTorch-native config tried) → ERROR for all 100 instances.

---

## 8. Figures

Every figure / table script requires an explicit `--csv-dir` pointing
at the real experiment-outputs directory. There is no fake-data
fallback; running a script without `--csv-dir` hard-errors.

```bash
# Single figure (ours, Hashemi, SaVer + sound-verifier consensus):
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.figures.fig2_exp1_runtime \
    --csv-dir examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs

# Bulk regenerator over all figures + tables (forwards --csv-dir verbatim):
~/miniconda3/envs/n2v/bin/python -m \
    examples.FlowConformal.paper.regenerate_all \
    --csv-dir examples/FlowConformal/experiments/exp1_vnncomp_subset/outputs
```

The per-script `--csv-dir → real outputs dir` mapping is documented
in [`examples/FlowConformal/paper/README.md`](paper/README.md).

Figures read directly from per-instance CSVs (column convention:
`wall_s` for wall-clock, plus method-specific columns). See
[`CSV_SCHEMAS.md`](CSV_SCHEMAS.md) for the canonical
column list.

---

## 9. Reproducibility guarantees

* **Single global seed:** `SEED = 47`. Every per-(benchmark, tool)
  runner resets `torch.manual_seed(47)` and `np.random.seed(47)` at
  the start of each instance's pipeline. Re-running any sweep
  produces bit-identical CSVs (modulo MCMC numerical reproducibility,
  which torch generators guarantee on a fixed CUDA device).

* **Order-independent:** the per-instance reset means reordering
  instances within a benchmark doesn't change any individual row.

* **Cross-tool deterministic:** the same instance's input box,
  calibration data, and spec are bit-identical regardless of which
  tool runs (the seed is set at the runner level, not by per-tool
  sub-seeding).

* **Synthetic instance generation** (Exp 3, Exp 4) uses
  `_stable_hash` (hashlib SHA-256) for cross-process determinism;
  Python's built-in `hash()` is randomised per-process and would
  break reproducibility.

---

## 10. Files of interest

* `examples/FlowConformal/experiments/README.md` — paper-experiment
  spec (benchmark choice, hparam overrides per benchmark, expected
  CSV outputs, execution order).
* `examples/FlowConformal/experiments/AUDIT_FINDINGS.md` — pre-run
  audit checklist with what was verified and what's pending.
* `examples/FlowConformal/experiments/run_all_sweeps.sh` — master
  launcher with `--smoke`, `--phase`, `--dry-run` flags.
* `examples/FlowConformal/experiments/_external_verifiers.py` —
  subprocess wrappers for αβ-CROWN, NeuralSAT, and ProbStar
  (each dispatches into its own conda env to avoid dep conflicts).
* `examples/FlowConformal/experiments/baselines/run_probstar.py` —
  zero-n2v-dep ProbStar standalone executed inside the `starv` env.
* `examples/FlowConformal/experiments/run_soundness_audit.py` —
  cross-experiment soundness audit (AutoAttack + 5K-PGD).
* `examples/FlowConformal/CSV_SCHEMAS.md` — canonical CSV schema for every per-instance
  and per-aggregate output.
