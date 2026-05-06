#!/usr/bin/env bash
# Companion to run_all_sweeps.sh — runs ONLY the remaining experiments
# the user explicitly enumerated on 2026-05-04 to complete the
# benchmark × method matrix for the FlowConformal paper.
#
# Final benchmark set (9): acasxu_2023, collins_rul_cnn_2022,
#   dist_shift_2023, linearizenn_2024, metaroom_2023, tllverify_2023,
#   cifar100_2024, tinyimagenet_2024, vit_2023
#   (lsnc_relu and relusplitter dropped from the paper.)
#
# Methods covered by this script:
#   ours              — extend Exp 2 (cifar100/tinyimagenet/vit) to 201 inst.
#   hashemi_clipping_pca (m=2500)
#                     — Exp 2 (cifar100/tinyimagenet/vit) at full 201.
#                       (basic Hashemi-clipping omitted on Exp 2: it
#                        TIMEOUTed on every UNSAT instance previously,
#                        so the comparison is uninformative.)
#   saver             — Exp 2 (cifar100/tinyimagenet/vit) at 201 +
#                       extend Exp 1 acasxu to 186 (currently only 100).
#   probstar          — Exp 1 core 6 benchmarks + Exp 2 (3 benchmarks).
#                       Returns NOT_APPLICABLE per-instance when StarV's
#                       loader can't handle the network.
#   rs                — Exp 2 only (classifier-only certifier). Extend
#                       cifar100 to 201; attempt vit + tinyimagenet
#                       (will be rejected by the runner if not in
#                        EXP2_RS_APPLICABLE — leave the failure in the
#                        log; cifar10_resnet110 is dropped from sweep).
#
# VNN-COMP 8-tool consensus is read post-hoc from
#   ~/v/other/VNNCOMP/vnncomp2025_results/<tool>/<benchmark>/results.csv
# at aggregation time. No re-run needed for those.
#
# Usage (mirrors run_all_sweeps.sh):
#   bash examples/FlowConformal/experiments/run_all_remaining.sh [--dry-run] [--smoke] [--phase <A..F>]
#
# Flags:
#   --dry-run     Print commands; don't execute.
#   --smoke       1-instance smoke per cell (validates pipeline only).
#   --phase <X>   Run only one phase (A, B, C, D, E, F); default = all.
#
# Pre-launch action (per the existing CSVs in outputs/):
#   Phases A, D, F-cifar100 RESUME from idx 100 — they preserve the
#   existing rows 0..99 and append rows 100..end. Do NOT delete those
#   CSVs (run_cell.sh's START_IDX env var skips the indices that are
#   already there, and the runner opens the CSV in append mode).
#
#   Phases B, C, F-vit, F-tinyimagenet, E are FRESH runs (no existing
#   CSVs). If you want to be paranoid about old logs, you can delete:
#     rm -f examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_{cifar100_2024,tinyimagenet_2024,vit_2023}_hashemi_clipping_pca.{csv,log}
#     rm -f examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_{cifar100_2024,tinyimagenet_2024,vit_2023}_saver.{csv,log}
#     rm -f examples/FlowConformal/experiments/exp2_prob_scale/outputs/exp2_{vit_2023,tinyimagenet_2024}_rs.{csv,log}
#   But these don't currently exist, so the rm is a safety no-op.
#
# Each cell uses the same run_cell.sh wrapper as run_all_sweeps.sh, so a
# single hung instance can't sink the rest of the sweep.

set -u

# ---------- config ----------
PY=/home/sasakis/miniconda3/envs/n2v/bin/python
REPO=/home/sasakis/v/tools/n2v
DRY_RUN=0
SMOKE=0
PHASE=all

# Instance counts. Bumped from the runner-side defaults of 100 to
# the full benchmark size for each Exp 2 / Exp 1 cell that we extend.
N_EXP2_FULL=201           # cifar100/tinyimagenet/vit have 201 instances
N_ACASXU_FULL=186         # acasxu_2023 has 186 instances

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"

HARDCAP_SMOKE=600         # 10 min — 1-instance smoke per cell

# Helpers — copied from run_all_sweeps.sh so this script is standalone.
log_start() {
  # Append a START banner (NOT overwrite) so we don't lose existing
  # log content. Important for resume cases where the existing .log
  # already has output from prior runs we want to preserve.
  local logfile="$1"; local label="$2"
  echo "[$(date -Iseconds)] === START: $label ===" >> "$logfile"
}
log_end() {
  local logfile="$1"; local label="$2"; local rc="$3"
  echo "" >> "$logfile"
  echo "[$(date -Iseconds)] === END:   $label (rc=$rc) ===" >> "$logfile"
}

run_sweep() {
  # Per-cell driver. For full sweeps (no --smoke), uses run_cell.sh
  # which invokes the runner once per instance with a per-instance
  # shell timeout — so a single hung instance can't kill the rest of
  # the cell. For smokes, calls the runner directly (only 1 instance,
  # no hang risk worth caring about).
  local exp_dir="$1"  # e.g. exp2_prob_scale
  local module="$2"   # e.g. exp2_run_ours
  local args="$3"     # e.g. "--benchmark cifar100_2024 --n-instances 201"
  local logname="$4"  # e.g. exp2_cifar100_2024_ours
  local logfile="examples/FlowConformal/experiments/$exp_dir/outputs/$logname.log"

  # Optional: env-style "START_IDX=<N>" prefix. When the caller wants
  # to resume a partial sweep without rerunning early-index rows, pass
  # START_IDX=<N> as an env override to run_cell.sh. Plumbed through
  # the local START_IDX_PREFIX so it shows in the log banner.
  local start_idx_label=""
  if [[ -n "${START_IDX_PREFIX:-}" ]]; then
    start_idx_label=" (START_IDX=$START_IDX_PREFIX)"
  fi
  echo "[$(date +%H:%M:%S)] >>> $logname$start_idx_label"
  if [[ $SMOKE -eq 1 ]]; then
    local cmd="timeout --kill-after=60 ${HARDCAP_SMOKE}s $PY -u -m examples.FlowConformal.experiments.$exp_dir.$module $args --smoke"
    echo "  [smoke] $cmd > $logfile"
    if [[ $DRY_RUN -eq 1 ]]; then
      return 0
    fi
    log_start "$logfile" "$logname"
    $cmd >> "$logfile" 2>&1
  else
    local cmd="bash examples/FlowConformal/experiments/run_cell.sh $exp_dir.$module $args"
    if [[ -n "${START_IDX_PREFIX:-}" ]]; then
      cmd="START_IDX=$START_IDX_PREFIX $cmd"
    fi
    echo "  [full] $cmd > $logfile"
    if [[ $DRY_RUN -eq 1 ]]; then
      return 0
    fi
    log_start "$logfile" "$logname"
    if [[ -n "${START_IDX_PREFIX:-}" ]]; then
      START_IDX="$START_IDX_PREFIX" bash examples/FlowConformal/experiments/run_cell.sh \
        "$exp_dir.$module" $args >> "$logfile" 2>&1
    else
      bash examples/FlowConformal/experiments/run_cell.sh \
        "$exp_dir.$module" $args >> "$logfile" 2>&1
    fi
  fi
  local rc=$?
  log_end "$logfile" "$logname" "$rc"
  if [[ $rc -eq 124 ]]; then
    echo "  HARD-CAP HIT — cell killed by 'timeout'; partial CSV preserved"
  elif [[ $rc -eq 137 ]]; then
    echo "  KILLED (SIGKILL) — likely OOM or stuck process; partial CSV preserved"
  elif [[ $rc -ne 0 ]]; then
    echo "  EXIT NON-ZERO ($rc) — see $logfile"
  else
    tail -3 "$logfile" | sed 's/^/    /'
  fi
  return $rc
}

# Benchmark groupings used below.
EXP2_3=(cifar100_2024 tinyimagenet_2024 vit_2023)
EXP1_PROBSTAR=(acasxu_2023 collins_rul_cnn_2022 dist_shift_2023
               linearizenn_2024 metaroom_2023 tllverify_2023)

# ---------- PHASE A: ours on Exp 2 (3 benchmarks) — RESUME from idx 100 ---
phase_A_ours_exp2_full() {
  echo
  echo "===================== PHASE A: ours on Exp 2 (cifar100, tinyimagenet, vit) — resume from idx 100, run through n=$N_EXP2_FULL ====================="
  # The original Phase 2 runs of ours on these three benchmarks were
  # capped at the runner-side default of n=100 instances, so the
  # existing CSVs already have rows for indices 0..99. We resume from
  # idx 100 (i.e., "instance 101") and append rows 100..200, preserving
  # the existing data. The runner opens the CSV in append mode when
  # invoked with --instance-idx, and run_cell.sh's START_IDX env var
  # skips the indices that already have rows.
  for bench in "${EXP2_3[@]}"; do
    START_IDX_PREFIX=100 run_sweep exp2_prob_scale exp2_run_ours \
              "--benchmark $bench --n-instances $N_EXP2_FULL" \
              "exp2_${bench}_ours"
  done
}

# ---------- PHASE B: Hashemi-PCA m=2500 on Exp 2 (3 benchmarks) at 201 ----
phase_B_hashemi_pca_exp2_full() {
  echo
  echo "===================== PHASE B: Hashemi-PCA m=2500 on Exp 2 (cifar100, tinyimagenet, vit) at n=$N_EXP2_FULL ====================="
  # Hashemi-PCA at m=2500 was wall-matched to ours during Phase 2'' on
  # cifar100 and tinyimagenet (~50s/inst, comparable to ours' 58-65s
  # under the 100s VNN-COMP budget). vit_2023 added here so all three
  # Exp 2 benchmarks have the m=2500 Hashemi-PCA row at full 201.
  # K=32 PCA components mirrors the published clipping-block paper's
  # high-output-dim recommendation.
  for bench in "${EXP2_3[@]}"; do
    run_sweep exp2_prob_scale exp2_run_hashemi_clipping_pca \
              "--benchmark $bench --pca-components 32 --m 2500 --n-instances $N_EXP2_FULL" \
              "exp2_${bench}_hashemi_clipping_pca"
  done
}

# ---------- PHASE C: SaVer on Exp 2 (3 benchmarks) at 201 ----------
phase_C_saver_exp2_full() {
  echo
  echo "===================== PHASE C: SaVer on Exp 2 (cifar100, tinyimagenet, vit) at n=$N_EXP2_FULL ====================="
  # SaVer (Convertino HSCC 2025) sample-based DKW; applies to any
  # network. Default delta=0.05 (post-fix) so DKW certification is
  # mathematically possible at m=8000.
  for bench in "${EXP2_3[@]}"; do
    run_sweep exp2_prob_scale exp2_run_saver \
              "--benchmark $bench --n-instances $N_EXP2_FULL" \
              "exp2_${bench}_saver"
  done
}

# ---------- PHASE D: SaVer on Exp 1 acasxu — RESUME from idx 100 ---------
phase_D_saver_acasxu_full() {
  echo
  echo "===================== PHASE D: SaVer on Exp 1 acasxu_2023 — resume from idx 100, run through n=$N_ACASXU_FULL ====================="
  # Existing exp1_acasxu_2023_saver.csv has 100 rows. Resume from idx
  # 100 to add rows 100..185 without re-running the existing 0..99.
  # The other 5 core Exp 1 benchmarks (collins_rul_cnn, dist_shift,
  # linearizenn, metaroom, tllverify) were already run on their full
  # instance counts under Phase 5b — see run_all_sweeps.sh.
  START_IDX_PREFIX=100 run_sweep exp1_vnncomp_subset exp1_run_saver \
            "--benchmark acasxu_2023 --n-instances $N_ACASXU_FULL" \
            "exp1_acasxu_2023_saver"
}

# ---------- PHASE E: ProbStar on Exp 1 (6 benches) + Exp 2 (3) ----
phase_E_probstar_all() {
  echo
  echo "===================== PHASE E: ProbStar on Exp 1 (6 benchmarks) + Exp 2 (3 benchmarks) ====================="
  # ProbStar / StarV (Tran et al.). Returns NOT_APPLICABLE per-instance
  # when StarV's loader can't handle the network ops (e.g., transformer
  # attention, residual Add, Gemm). For Exp 2 this is uniformly
  # NOT_APPLICABLE on tinyimagenet/cifar100/vit per Phase 9's history,
  # but we re-run at the full 201 instances anyway so the row in the
  # paper table reports per-instance ProbStar verdicts (NOT_APPLICABLE
  # is itself useful information for the reviewer).
  #
  # Exp 1 ProbStar runner has no --n-instances flag (uses all instances
  # in list_instances by default).
  for bench in "${EXP1_PROBSTAR[@]}"; do
    run_sweep exp1_vnncomp_subset exp1_run_probstar \
              "--benchmark $bench" \
              "exp1_${bench}_probstar"
  done
  for bench in "${EXP2_3[@]}"; do
    run_sweep exp2_prob_scale exp2_run_probstar \
              "--benchmark $bench --n-instances $N_EXP2_FULL" \
              "exp2_${bench}_probstar"
  done
}

# ---------- PHASE F: RS on Exp 2 (where applicable) ----------
phase_F_rs_exp2() {
  echo
  echo "===================== PHASE F: RS on Exp 2 (cifar100 RESUME + try vit/tinyimagenet fresh) ====================="
  # Cohen et al. 2019 randomized smoothing; classifier-only.
  # cifar100_2024 was previously run at n=100; resume from idx 100 to
  # extend to 201 without losing existing rows.
  START_IDX_PREFIX=100 run_sweep exp2_prob_scale exp2_run_rs \
            "--benchmark cifar100_2024 --n-instances $N_EXP2_FULL" \
            "exp2_cifar100_2024_rs"
  # vit_2023 and tinyimagenet_2024 are fresh runs (no existing CSVs).
  # If the runner's EXP2_RS_APPLICABLE choices reject them (no
  # noise-trained smooth weights available), the cell will exit
  # non-zero and the failure will be recorded in the log — that is
  # acceptable evidence that RS doesn't apply for those benchmarks.
  for bench in vit_2023 tinyimagenet_2024; do
    run_sweep exp2_prob_scale exp2_run_rs \
              "--benchmark $bench --n-instances $N_EXP2_FULL" \
              "exp2_${bench}_rs"
  done
}

# ---------- dispatch ----------
case "$PHASE" in
  A|a) phase_A_ours_exp2_full ;;
  B|b) phase_B_hashemi_pca_exp2_full ;;
  C|c) phase_C_saver_exp2_full ;;
  D|d) phase_D_saver_acasxu_full ;;
  E|e) phase_E_probstar_all ;;
  F|f) phase_F_rs_exp2 ;;
  all)
    phase_A_ours_exp2_full
    phase_B_hashemi_pca_exp2_full
    phase_C_saver_exp2_full
    phase_D_saver_acasxu_full
    phase_E_probstar_all
    phase_F_rs_exp2
    ;;
  *) echo "unknown phase: $PHASE (valid: A, B, C, D, E, F, or 'all')" >&2; exit 2 ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_all_remaining complete ==="
