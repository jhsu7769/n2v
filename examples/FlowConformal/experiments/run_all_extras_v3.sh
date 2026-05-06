#!/usr/bin/env bash
# Companion to run_all_extras_v2.sh. Adds the verification-method
# ablation v2 (designed 2026-05-06):
#
#   Phase G'' — shared-flow ablation, full sweep on 2 benchmarks.
#       Method shortlist (5 methods, dropping scenario_v2 / derived /
#       amls_bounded_union after Phase G' showed those redundant):
#         scenario, amls, amls_bounded, is_tilted, raw_mc_uniform.
#       Benchmarks:
#         * acasxu_2023 — ALL 186 instances. Headline UNSAT-recall on
#           the same distribution the production paper table reports.
#         * tllverify_2023 — ALL 32 instances. Reach-boundary close to
#           unsafe halfspace; the benchmark where unbounded AMLS
#           historically over-flagged → motivated bounded AMLS.
#
#   The shared-flow refactor makes calibration cost amortise: 1 flow +
#   q per (instance, box), all 5 verifiers reuse it. Output filenames
#   are prefix-distinct from the older v2 outputs:
#     ablation_shared_flow_<benchmark>_<method>.csv.
#
# Usage:
#   bash examples/FlowConformal/experiments/run_all_extras_v3.sh \
#        [--dry-run] [--force] [--benchmark acasxu_2023|tllverify_2023]
#
# --force  : allow OVERWRITING existing CSVs (default: abort).
# --benchmark : run only one benchmark (default: both).
#
# Wall-clock estimate: ACAS Xu ~3.5h (186 inst × ~70s/inst), tllverify
# ~30 min (32 inst × ~70s/inst). Total ~4 hrs.

set -u

PY=/home/sasakis/miniconda3/envs/n2v/bin/python
REPO=/home/sasakis/v/tools/n2v
DRY_RUN=0
FORCE=0
BENCHMARK=both

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --force) FORCE=1; shift ;;
    --benchmark) BENCHMARK="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

cd "$REPO"

METHODS=(scenario amls amls_bounded is_tilted raw_mc_uniform)

log_start() { echo "[$(date -Iseconds)] === START: $2 ===" >> "$1"; }
log_end() {
  echo "" >> "$1"
  echo "[$(date -Iseconds)] === END:   $2 (rc=$3) ===" >> "$1"
}

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
  local label="$1"; local logfile="$2"
  shift 2
  echo "[$(date +%H:%M:%S)] >>> $label"
  echo "  cmd: $*"
  echo "  log: $logfile"
  if [[ $DRY_RUN -eq 1 ]]; then return 0; fi
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

run_benchmark() {
  local bench="$1"
  local n_inst="$2"
  local out_dir="examples/FlowConformal/experiments/exp_ablation/outputs"
  local logfile="$out_dir/ablation_shared_flow_${bench}.log"
  local prefix="ablation_shared_flow_${bench}"

  echo
  echo "===================== Phase G'' shared-flow ablation: $bench (${n_inst} inst × ${#METHODS[@]} methods) ====================="

  local targets=()
  for m in "${METHODS[@]}"; do
    targets+=("$out_dir/${prefix}_${m}.csv")
  done
  if ! guard_no_overwrite "${targets[@]}"; then
    return 1
  fi

  run_cmd "phaseGpp_${bench}" "$logfile" \
    $PY -u -m examples.FlowConformal.experiments.exp_ablation.ablation_shared_flow \
    --benchmark "$bench" \
    --n-instances "$n_inst" \
    --methods "${METHODS[@]}"
}

case "$BENCHMARK" in
  acasxu_2023)   run_benchmark acasxu_2023 186 ;;
  tllverify_2023) run_benchmark tllverify_2023 32 ;;
  both)
    run_benchmark acasxu_2023 186
    run_benchmark tllverify_2023 32
    ;;
  *) echo "unknown benchmark: $BENCHMARK" >&2; exit 2 ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_all_extras_v3 complete ==="
