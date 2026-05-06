#!/usr/bin/env bash
# Companion to run_all_extra.sh. Adds the experiments designed on
# 2026-05-06:
#
#   Phase G' — Verification-method ablation with shared (flow, q).
#       20 randomly-sampled ACAS Xu instances; each calibrates the flow
#       ONCE and loops all 8 verifiers against the shared (flow, q)
#       tuple, so the only varying axis is the verifier. Outputs to
#       ablation_shared_flow_<method>.csv (PREFIX-DISTINCT from the
#       previous Phase G outputs ablation_verify_method_<method>.csv,
#       which are preserved untouched).
#
#   Phase H' — Volume sweep on NEW benchmarks only.
#       2d_banana, 3d_banana (already wired in v2 with cached exact),
#       synth_2d, synth_3d. Three configs (small / default / large)
#       per (method × benchmark). DOES NOT touch synth_5d / synth_10d /
#       synth_20d (already covered by run_all_extra.sh).
#       Filenames stay orthogonal to v1: exp3_<bench>_*_<config>.csv.
#       The script ABORTS rather than overwrite an existing CSV unless
#       --force is passed.
#
#   Phase I — Sound starset-approx baseline.
#       n2v's reach_pytorch_model(model, Star, method='approx') across
#       ALL 7 benchmarks (2d_banana, 3d_banana, synth_2d, synth_3d,
#       synth_5d, synth_10d, synth_20d) at 5 seeds each. Output
#       prefix exp3_<bench>_unsat_starset_approx.csv (NEW, no
#       overwrite risk).
#
# Usage:
#   bash examples/FlowConformal/experiments/run_all_extras_v2.sh \
#        [--dry-run] [--force] [--phase G|H|I]
#
# --force  : allow OVERWRITING existing CSVs (default: abort with a
#            warning when an output path already exists).

set -u

PY=/home/sasakis/miniconda3/envs/n2v/bin/python
REPO=/home/sasakis/v/tools/n2v
DRY_RUN=0
FORCE=0
PHASE=all

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --phase) PHASE="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"

log_start() {
  local logfile="$1"; local label="$2"
  echo "[$(date -Iseconds)] === START: $label ===" >> "$logfile"
}
log_end() {
  local logfile="$1"; local label="$2"; local rc="$3"
  echo "" >> "$logfile"
  echo "[$(date -Iseconds)] === END:   $label (rc=$rc) ===" >> "$logfile"
}

# Abort if any of $@ already exist as files unless --force was passed.
# Returns 0 if safe to proceed, 1 if a collision was detected.
guard_no_overwrite() {
  local hit=0
  for f in "$@"; do
    if [[ -e "$f" ]]; then
      echo "  ABORT: would overwrite existing file: $f" >&2
      hit=1
    fi
  done
  if [[ $hit -eq 1 && $FORCE -eq 0 ]]; then
    echo "  Pass --force to overwrite. Stopping this cell." >&2
    return 1
  fi
  return 0
}

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
  "$@" >> "$logfile" 2>&1
  local rc=$?
  log_end "$logfile" "$label" "$rc"
  if [[ $rc -ne 0 ]]; then
    echo "  EXIT NON-ZERO ($rc) — see $logfile"
  else
    tail -3 "$logfile" | sed 's/^/    /'
  fi
  return $rc
}

# ---------- PHASE G': shared-flow ablation ----------
phase_Gp_shared_flow_ablation() {
  echo
  echo "===================== PHASE G': shared-flow ablation (20 random ACAS Xu instances × 8 verifiers) ====================="
  local out_dir="examples/FlowConformal/experiments/exp_ablation/outputs"
  local logfile="$out_dir/ablation_shared_flow.log"
  # Per-method CSVs that this cell will write. Guard each.
  local methods=(scenario scenario_v2 amls is_tilted derived
                 amls_bounded amls_bounded_union raw_mc_uniform)
  local targets=()
  for m in "${methods[@]}"; do
    targets+=("$out_dir/ablation_shared_flow_${m}.csv")
  done
  if ! guard_no_overwrite "${targets[@]}"; then
    return 1
  fi
  run_cmd "phaseGp_shared_flow_ablation" "$logfile" \
    $PY -u -m examples.FlowConformal.experiments.exp_ablation.ablation_shared_flow \
    --n-instances 20
}

# ---------- PHASE H': new-benchmark volume sweep ----------
EXP3_NEW_BENCHES=(2d_banana 3d_banana synth_2d synth_3d)

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

phase_Hp_volume_sweep_new_benches() {
  echo
  echo "===================== PHASE H': volume sweep on NEW benchmarks (4 new benches × 3 configs × 2 methods × 5 seeds) ====================="
  local out_dir="examples/FlowConformal/experiments/exp3_synthetic/outputs"

  for bench in "${EXP3_NEW_BENCHES[@]}"; do
    for cfg in small default large; do
      local ours_csv="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.csv"
      local ours_log="$out_dir/exp3_${bench}_flow_unsat_ours_${cfg}.log"
      if guard_no_overwrite "$ours_csv"; then
        run_cmd "phaseHp_${bench}_ours_${cfg}" "$ours_log" \
          $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours \
          --benchmark "$bench" --score flow --spec unsat \
          --seeds 5 \
          --output-csv "$ours_csv" \
          ${OURS_CONFIG_ARGS[$cfg]}
      fi
      local hash_csv="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.csv"
      local hash_log="$out_dir/exp3_${bench}_unsat_hashemi_clipping_${cfg}.log"
      if guard_no_overwrite "$hash_csv"; then
        run_cmd "phaseHp_${bench}_hashemi_${cfg}" "$hash_log" \
          $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_hashemi_clipping \
          --benchmark "$bench" --spec unsat \
          --seeds 5 \
          --output-csv "$hash_csv" \
          ${HASHEMI_CONFIG_ARGS[$cfg]}
      fi
    done
  done
}

# ---------- PHASE I: sound starset-approx baseline ----------
EXP3_ALL_BENCHES=(2d_banana 3d_banana synth_2d synth_3d
                  synth_5d synth_10d synth_20d)

phase_I_starset_approx_all() {
  echo
  echo "===================== PHASE I: starset-approx baseline (7 benchmarks × 5 seeds) ====================="
  local out_dir="examples/FlowConformal/experiments/exp3_synthetic/outputs"
  for bench in "${EXP3_ALL_BENCHES[@]}"; do
    local csv="$out_dir/exp3_${bench}_unsat_starset_approx.csv"
    local log="$out_dir/exp3_${bench}_unsat_starset_approx.log"
    if guard_no_overwrite "$csv"; then
      run_cmd "phaseI_${bench}_starset_approx" "$log" \
        $PY -u -m examples.FlowConformal.experiments.exp3_synthetic.exp3_run_starset_approx \
        --benchmark "$bench" --spec unsat --seeds 5 \
        --output-csv "$csv"
    fi
  done
}

# ---------- dispatch ----------
case "$PHASE" in
  G|g) phase_Gp_shared_flow_ablation ;;
  H|h) phase_Hp_volume_sweep_new_benches ;;
  I|i) phase_I_starset_approx_all ;;
  all)
    phase_Gp_shared_flow_ablation
    phase_Hp_volume_sweep_new_benches
    phase_I_starset_approx_all
    ;;
  *) echo "unknown phase: $PHASE (valid: G, H, I, or 'all')" >&2; exit 2 ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_all_extras_v2 complete ==="
