#!/usr/bin/env bash
# Companion to run_all_sweeps.sh / run_all_remaining.sh. Runs the
# follow-up paper experiments added on 2026-05-05:
#
#   Phase G — AMLS verification-method ablation.
#       8 methods (scenario, scenario_v2, amls, is_tilted, derived,
#       amls_bounded, amls_bounded_union, raw_mc_uniform) on the 10
#       ACAS Xu probe instances. Same locked Phase 5d hparams.
#       Output: examples/FlowConformal/experiments/exp_ablation/outputs/
#               ablation_verify_method_<method>.csv
#
#   Phase H — Synthetic volume comparison.
#       4 benchmarks (3d_banana, synth_5d, synth_10d, synth_20d) ×
#       3 sample-budget configs (small/default/large) × 2 methods
#       (ours bounded-AMLS, hashemi_clipping) × 5 seeds. The 1-Lipschitz
#       identity-activation nets give a closed-form reach-set volume
#       (zonotope determinant) that both methods are scored against.
#       Output: examples/FlowConformal/experiments/exp3_synthetic/outputs/
#               exp3_<bench>_flow_unsat_ours_<config>.csv
#               exp3_<bench>_unsat_hashemi_clipping_<config>.csv
#
# Wall-clock budget (Phase G ~80 min, Phase H ~90 min) — plan ~3 hrs.
#
# Usage:
#   bash examples/FlowConformal/experiments/run_all_extra.sh \
#        [--dry-run] [--smoke] [--phase G|H]
#
# Flags mirror run_all_remaining.sh.

set -u

PY=/home/sasakis/miniconda3/envs/n2v/bin/python
REPO=/home/sasakis/v/tools/n2v
DRY_RUN=0
SMOKE=0
PHASE=all

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --smoke) SMOKE=1; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"

HARDCAP_SMOKE=600    # 10 min per smoke cell

log_start() {
  local logfile="$1"; local label="$2"
  echo "[$(date -Iseconds)] === START: $label ===" >> "$logfile"
}
log_end() {
  local logfile="$1"; local label="$2"; local rc="$3"
  echo "" >> "$logfile"
  echo "[$(date -Iseconds)] === END:   $label (rc=$rc) ===" >> "$logfile"
}

# Generic single-command wrapper. No per-instance shell timeout — the
# extra experiments are batch runs (one process per cell, all instances
# inline). Use HARDCAP_SMOKE for the smoke path only.
run_cmd() {
  local label="$1"
  local logfile="$2"
  shift 2
  echo "[$(date +%H:%M:%S)] >>> $label"
  echo "  cmd: $*"
  echo "  log: $logfile"
  if [[ $DRY_RUN -eq 1 ]]; then
    return 0
  fi
  mkdir -p "$(dirname "$logfile")"
  log_start "$logfile" "$label"
  if [[ $SMOKE -eq 1 ]]; then
    timeout --kill-after=60 ${HARDCAP_SMOKE}s "$@" >> "$logfile" 2>&1
  else
    "$@" >> "$logfile" 2>&1
  fi
  local rc=$?
  log_end "$logfile" "$label" "$rc"
  if [[ $rc -ne 0 ]]; then
    echo "  EXIT NON-ZERO ($rc) — see $logfile"
  else
    tail -3 "$logfile" | sed 's/^/    /'
  fi
  return $rc
}

# ---------- PHASE G: AMLS verification-method ablation ----------
phase_G_amls_ablation() {
  echo
  echo "===================== PHASE G: AMLS verification-method ablation (8 methods × 10 ACAS Xu probe instances) ====================="
  local out_dir="examples/FlowConformal/experiments/exp_ablation/outputs"
  local logfile="$out_dir/ablation_verify_method.log"
  local extra_args=""
  if [[ $SMOKE -eq 1 ]]; then
    extra_args="--smoke"
  fi
  run_cmd "phaseG_amls_ablation" "$logfile" \
    $PY -u -m examples.FlowConformal.experiments.exp_ablation.ablation_run_verify_method \
    $extra_args
}

# ---------- PHASE H: Synthetic volume sweep ----------
EXP3_BENCHES=(3d_banana synth_5d synth_10d synth_20d)

# Three sample-budget configs per method. Tagged in the output CSV
# filename so per-config rows don't clobber each other.
#
# Ours (n_train, flow_epochs, scenario_n, volume_m, volume_n_samples):
#   small   — quick / low-fidelity baseline
#   default — locked Phase 5d production config
#   large   — push-the-budget upper end (volume MC at 4× default)
#
# Hashemi (m):
#   small=1000, default=8000, large=16000.
declare -A OURS_CONFIG_ARGS=(
  [small]="--n-train 2000 --flow-epochs 500 --scenario-n-samples 1000 --volume-m 1000 --volume-n-samples 100000"
  [default]="--n-train 5000 --flow-epochs 2000 --scenario-n-samples 2000 --volume-m 8000 --volume-n-samples 200000"
  [large]="--n-train 10000 --flow-epochs 5000 --scenario-n-samples 4000 --volume-m 16000 --volume-n-samples 400000"
)
declare -A HASHEMI_CONFIG_ARGS=(
  [small]="--m 1000"
  [default]="--m 8000"
  [large]="--m 16000"
)

phase_H_volume_sweep() {
  echo
  echo "===================== PHASE H: Synthetic volume sweep (4 benches × 3 configs × 2 methods × 5 seeds) ====================="
  local out_dir="examples/FlowConformal/experiments/exp3_synthetic/outputs"
  local seeds_arg=""
  if [[ $SMOKE -eq 1 ]]; then
    seeds_arg="--smoke"
  else
    seeds_arg="--seeds 5"
  fi

  for bench in "${EXP3_BENCHES[@]}"; do
    for cfg in small default large; do
      # ---- ours ----
      local ours_csv="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.csv"
      local ours_log="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.log"
      run_cmd "phaseH_${bench}_ours_${cfg}" "$ours_log" \
        $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \
        --benchmark "$bench" --score flow --spec unsat \
        $seeds_arg \
        --output-csv "$ours_csv" \
        ${OURS_CONFIG_ARGS[$cfg]}
      # ---- hashemi ----
      local hash_csv="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.csv"
      local hash_log="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.log"
      run_cmd "phaseH_${bench}_hashemi_${cfg}" "$hash_log" \
        $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_hashemi_clipping \
        --benchmark "$bench" --spec unsat \
        $seeds_arg \
        --output-csv "$hash_csv" \
        ${HASHEMI_CONFIG_ARGS[$cfg]}
    done
  done
}

# ---------- dispatch ----------
case "$PHASE" in
  G|g) phase_G_amls_ablation ;;
  H|h) phase_H_volume_sweep ;;
  all)
    phase_G_amls_ablation
    phase_H_volume_sweep
    ;;
  *) echo "unknown phase: $PHASE (valid: G, H, or 'all')" >&2; exit 2 ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_all_extra complete ==="
