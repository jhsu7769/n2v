#!/usr/bin/env bash
# Pre-launch validation battery — two phases.
#
# === Phase A: config search (lock probe) ===
# Iterates ``mega → full → ... → nano`` until 5/5 instances pass for
# each ours cell that hasn't been validated by the existing lock-probe
# CSV. After the probe locks a config, that's the largest hparam
# tuple that fits the per-row budget.
#
# Cells covered:
#   * Exp 2 × {vit_2023, cifar100_2024} ours
#     - vit_2023 uses ``amls_bounded`` with the per-benchmark
#       ``max_levels=5`` override (now plumbed through the probe).
#     - cifar100_2024 uses ``amls_bounded_union`` for its 99-disjunct
#       classification spec.
#   * Exp 1 × {malbeware, metaroom_2023} ours — verified by probe v4
#     in this session, but re-running here as part of the overnight
#     audit (cheap; both lock at mega in ~7-8 min each).
#
# Exp 3 / Exp 4 are intentionally fixed-config-by-design (synthetic
# scaling study; per-depth tuning would defeat the comparison). Those
# are smoke-validated only in Phase B.
#
# === Phase B: smoke validation ===
# 1-instance smoke for every (experiment, tool, benchmark) cell we
# plan to sweep, using whatever's currently in ``PER_BENCHMARK_CONFIG``.
# Cells covered by Phase A are skipped (the probe ran 5 instances
# there, which is stronger than a smoke).
#
# === Output ===
# Phase A writes a probe CSV under
#   ``examples/FlowConformal/probes/outputs/probe_amls_bounded_lock_overnight.csv``
# Phase B writes per-smoke logs and a summary to
#   ``examples/FlowConformal/experiments/outputs/smoke_logs/<slug>.log``
#   ``examples/FlowConformal/experiments/outputs/smoke_summary.csv``
#
# In the morning, review:
#   1. The probe CSV — verify each Phase-A cell has ``cell_status='ok'``
#      at ``mega``. If a smaller config was locked, update
#      ``PER_BENCHMARK_CONFIG`` accordingly before the real sweep.
#   2. The smoke summary — every row should be ``PASS``. Any
#      ``FAIL/TIMEOUT`` rows need investigation before launch.
#
# Usage:
#   bash examples/FlowConformal/experiments/run_all_smokes.sh
#
# Overrides:
#   PER_SMOKE_TIMEOUT_S  hard wall per smoke (default 1800)
#   PROBE_INSTANCES_PER_CELL  default 5
#   PY                   python interpreter (default n2v env)
#
# Exit code: 0 if all probes locked + all smokes passed, else
# (probe failures + smoke failures + smoke timeouts).

set -u

PY=${PY:-/home/sasakis/miniconda3/envs/n2v/bin/python}
PER_SMOKE_TIMEOUT_S=${PER_SMOKE_TIMEOUT_S:-1800}
PROBE_INSTANCES_PER_CELL=${PROBE_INSTANCES_PER_CELL:-5}
LOG_DIR=examples/FlowConformal/experiments/outputs/smoke_logs
SUMMARY=examples/FlowConformal/experiments/outputs/smoke_summary.csv
PROBE_CSV=examples/FlowConformal/probes/outputs/probe_amls_bounded_lock_overnight.csv
PROBE_LOG=examples/FlowConformal/probes/outputs/probe_amls_bounded_lock_overnight.log

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PROBE_CSV")"
echo "slug,status,wall_s,verdict,notes" > "$SUMMARY"

T_OVERALL_START=$(date +%s)
N_PASS=0
N_FAIL=0
N_TIMEOUT=0

# ====================================================================
# Phase A: config search via lock probe
# ====================================================================
PROBE_BENCHMARKS="vit_2023,cifar100_2024,malbeware,metaroom_2023"

echo
echo "===================================================================="
echo "  PHASE A: config search (lock probe)"
echo "  benchmarks: $PROBE_BENCHMARKS"
echo "  instances per cell: $PROBE_INSTANCES_PER_CELL"
echo "  log: $PROBE_LOG"
echo "===================================================================="
T_PROBE_START=$(date +%s)

"$PY" -u -m examples.FlowConformal.probes.probe_amls_bounded_lock \
    --benchmarks "$PROBE_BENCHMARKS" \
    --output-csv "$PROBE_CSV" \
    --instances-per-cell "$PROBE_INSTANCES_PER_CELL" \
    > "$PROBE_LOG" 2>&1
PROBE_RC=$?

T_PROBE_END=$(date +%s)
PROBE_WALL=$((T_PROBE_END - T_PROBE_START))

# Parse locked configs out of the probe CSV.
echo
echo "Phase A finished (rc=$PROBE_RC, wall=${PROBE_WALL}s)"
echo
echo "Per-benchmark locked configs (from $PROBE_CSV):"

LOCKED_CONFIGS=$(
"$PY" - <<'PYEOF'
import csv, os
path = os.environ.get('PROBE_CSV', 'examples/FlowConformal/probes/outputs/probe_amls_bounded_lock_overnight.csv')
locked = {}
if os.path.exists(path):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('cell_status', '').strip() == 'ok':
                locked[r['benchmark']] = r['config']
for b in ('vit_2023', 'cifar100_2024', 'malbeware', 'metaroom_2023'):
    print(f'  {b:30s} -> {locked.get(b, "NOT LOCKED — investigate before launch")}')
PYEOF
)
PROBE_CSV="$PROBE_CSV" eval "$(echo "  $LOCKED_CONFIGS")" 2>/dev/null
echo "$LOCKED_CONFIGS"

# Track probe failures (any benchmark not at 'ok' status).
PROBE_FAIL_COUNT=$(
PROBE_CSV="$PROBE_CSV" "$PY" - <<'PYEOF'
import csv, os
path = os.environ['PROBE_CSV']
locked = set()
if os.path.exists(path):
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get('cell_status', '').strip() == 'ok':
                locked.add(r['benchmark'])
expected = {'vit_2023', 'cifar100_2024', 'malbeware', 'metaroom_2023'}
missing = expected - locked
print(len(missing))
PYEOF
)


# ====================================================================
# Phase B: smoke validation for every other cell
# ====================================================================
echo
echo "===================================================================="
echo "  PHASE B: smoke validation"
echo "  per-smoke timeout: ${PER_SMOKE_TIMEOUT_S}s"
echo "===================================================================="

run_smoke() {
  local slug="$1"
  local module="$2"
  shift 2
  local extra_args=("$@")

  local logfile="$LOG_DIR/$slug.log"
  local t_start=$(date +%s)

  echo
  echo "==== [$(date +%H:%M:%S)] $slug ===="
  echo "  module: $module"
  echo "  args:   ${extra_args[*]} --smoke"
  echo "  log:    $logfile"

  timeout --kill-after=30 "${PER_SMOKE_TIMEOUT_S}s" \
    "$PY" -u -m "$module" "${extra_args[@]}" --smoke \
    > "$logfile" 2>&1
  local rc=$?
  local t_end=$(date +%s)
  local wall=$((t_end - t_start))

  local verdict=$(grep -oE 'verdict=[A-Z_]+' "$logfile" | tail -1 | sed 's/verdict=//')
  [[ -z "$verdict" ]] && verdict="-"

  local status notes
  if [[ $rc -eq 0 ]] && grep -q '\[smoke\] PASS' "$logfile"; then
    status="PASS"
    notes=""
    N_PASS=$((N_PASS + 1))
  elif [[ $rc -eq 124 ]] || [[ $rc -eq 137 ]]; then
    status="TIMEOUT"
    notes="shell timeout ${PER_SMOKE_TIMEOUT_S}s"
    N_TIMEOUT=$((N_TIMEOUT + 1))
  else
    status="FAIL"
    notes="exit_code=$rc; see $logfile"
    N_FAIL=$((N_FAIL + 1))
  fi

  printf '%s,%s,%d,%s,"%s"\n' "$slug" "$status" "$wall" "$verdict" "$notes" >> "$SUMMARY"
  echo "  → $status  wall=${wall}s  verdict=$verdict"
}

# --------------------------------------------------------------------
# Exp 1: hashemi_clipping smokes for the two new benchmarks. The 5
# original benchmarks were validated end-to-end by the Phase 5e + lock
# probe; ours on the new benchmarks was validated by Phase A above.
# --------------------------------------------------------------------
echo
echo "----- EXP 1 (hashemi on new benchmarks) -----"
for bench in malbeware metaroom_2023; do
  run_smoke "exp1_${bench}_hashemi_clipping" \
    "examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_run_hashemi_clipping" \
    --benchmark "$bench"
done

# --------------------------------------------------------------------
# Exp 2: full smoke matrix EXCEPT vit_2023 / cifar100_2024 ours
# (covered by Phase A) and cifar10_resnet110 ours (already locked at
# mega in the original lock probe).
# --------------------------------------------------------------------
echo
echo "----- EXP 2 (smokes; ours for vit/cifar100 covered in Phase A) -----"

# ours — only tinyimagenet_2024 and cifar10_resnet110 here (others in Phase A).
for bench in tinyimagenet_2024 cifar10_resnet110; do
  run_smoke "exp2_${bench}_ours" \
    "examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_ours" \
    --benchmark "$bench"
done

# hashemi_clipping
for bench in vit_2023 tinyimagenet_2024 cifar100_2024 cifar10_resnet110; do
  run_smoke "exp2_${bench}_hashemi_clipping" \
    "examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_hashemi_clipping" \
    --benchmark "$bench"
done

# αβ-CROWN (subprocess path) — includes the new locally-built
# cifar10_resnet110 ONNX/vnnlib artifacts.
for bench in vit_2023 tinyimagenet_2024 cifar100_2024 cifar10_resnet110; do
  run_smoke "exp2_${bench}_alpha_beta_crown" \
    "examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_alpha_beta_crown" \
    --benchmark "$bench"
done

# RS (image classification only)
for bench in cifar10_resnet110 cifar100_2024; do
  run_smoke "exp2_${bench}_rs" \
    "examples.FlowConformal.experiments.exp2_prob_scale.exp2_run_rs" \
    --benchmark "$bench"
done

# --------------------------------------------------------------------
# Exp 3: ours × 4 score families × 4 benchmarks on the unsat spec,
# plus hashemi × 4 benchmarks, plus one representative sat-spec smoke
# to confirm honest UNKNOWN abstention with falsifier OFF.
# --------------------------------------------------------------------
echo
echo "----- EXP 3 (score families × benchmarks) -----"
for bench in 3d_banana synth_5d synth_10d synth_20d; do
  for score in flow hyperrect ellipsoid gmm; do
    run_smoke "exp3_${bench}_${score}_unsat_ours" \
      "examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours" \
      --benchmark "$bench" --score "$score" --spec unsat
  done
  run_smoke "exp3_${bench}_unsat_hashemi_clipping" \
    "examples.FlowConformal.experiments.exp3_synthetic.exp3_run_hashemi_clipping" \
    --benchmark "$bench" --spec unsat
done

run_smoke "exp3_synth_5d_flow_sat_ours_abstention" \
  "examples.FlowConformal.experiments.exp3_synthetic.exp3_run_ours" \
  --benchmark synth_5d --score flow --spec sat

# --------------------------------------------------------------------
# Exp 4: ours/hashemi at every depth (canonical scaling curve);
# αβ-CROWN/NeuralSAT at d ∈ {2, 8, 16} to confirm subprocess plumbing.
# --------------------------------------------------------------------
echo
echo "----- EXP 4 (depth scaling) -----"
for d in 2 4 8 16 24 32 40; do
  run_smoke "exp4_d${d}_ours" \
    "examples.FlowConformal.experiments.exp4_scaling.exp4_run_ours" \
    --depth "$d"
done
for d in 2 4 8 16 24 32 40; do
  run_smoke "exp4_d${d}_hashemi_clipping" \
    "examples.FlowConformal.experiments.exp4_scaling.exp4_run_hashemi_clipping" \
    --depth "$d"
done
for d in 2 8 16; do
  run_smoke "exp4_d${d}_alpha_beta_crown" \
    "examples.FlowConformal.experiments.exp4_scaling.exp4_run_alpha_beta_crown" \
    --depth "$d"
  run_smoke "exp4_d${d}_neuralsat" \
    "examples.FlowConformal.experiments.exp4_scaling.exp4_run_neuralsat" \
    --depth "$d"
done

# ====================================================================
# Final summary
# ====================================================================
T_OVERALL_END=$(date +%s)
TOTAL_WALL=$((T_OVERALL_END - T_OVERALL_START))

echo
echo "===================================================================="
echo "                       OVERNIGHT SUMMARY"
echo "===================================================================="
echo
echo "  PHASE A — lock probe (config search)"
echo "    wall: ${PROBE_WALL}s"
echo "    log:  $PROBE_LOG"
echo "    csv:  $PROBE_CSV"
echo "$LOCKED_CONFIGS"
echo "    benchmarks failing to lock: $PROBE_FAIL_COUNT / 4"
echo
echo "  PHASE B — smoke battery"
N_TOTAL=$((N_PASS + N_FAIL + N_TIMEOUT))
echo "    Total smokes: $N_TOTAL"
echo "    PASS:         $N_PASS"
echo "    FAIL:         $N_FAIL"
echo "    TIMEOUT:      $N_TIMEOUT"
echo "    summary:      $SUMMARY"
echo "    logs:         $LOG_DIR/"
echo
printf '  Overall wall: %dh %02dm %02ds\n' \
  $((TOTAL_WALL / 3600)) $(((TOTAL_WALL % 3600) / 60)) $((TOTAL_WALL % 60))

if [[ "$PROBE_FAIL_COUNT" -gt 0 || $N_FAIL -gt 0 || $N_TIMEOUT -gt 0 ]]; then
  echo
  echo "  ACTION ITEMS before launching the real sweep:"
  if [[ "$PROBE_FAIL_COUNT" -gt 0 ]]; then
    echo "    * Phase A: $PROBE_FAIL_COUNT benchmark(s) failed to lock."
    echo "      Review $PROBE_CSV to see which configs were attempted."
  fi
  if [[ $N_FAIL -gt 0 || $N_TIMEOUT -gt 0 ]]; then
    echo "    * Phase B failed/timed-out smokes:"
    awk -F',' 'NR>1 && ($2=="FAIL" || $2=="TIMEOUT") {printf "        %s  %s\n", $1, $2}' "$SUMMARY"
  fi
fi

exit $((PROBE_FAIL_COUNT + N_FAIL + N_TIMEOUT))
