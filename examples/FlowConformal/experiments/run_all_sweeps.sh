#!/usr/bin/env bash
# Master sweep launcher for the FlowConformal paper experiments.
#
# Runs every (experiment, tool, benchmark) sweep in priority order from
# the README's execution-order table. Each sweep gets its own log file
# under ``examples/FlowConformal/experiments/<exp>/outputs/``.
#
# Usage:
#   bash examples/FlowConformal/experiments/run_all_sweeps.sh [--dry-run] [--smoke] [--phase <N>]
#
# Flags:
#   --dry-run     Print the commands that would be executed; don't run.
#   --smoke       Append --smoke to every runner (1 instance/seed each).
#                 Useful for end-to-end pipeline validation before the
#                 full sweep (which can take ~73 hours sequentially).
#   --phase <N>   Run only one phase (1-11); default = all in order.
#                 Reordered for deadline-driven priority (highest paper
#                 value first; LP-bound malbeware Hashemi cells last so
#                 they're the natural cut if time runs out):
#                 Phase 1 = Exp 1 ours+hashemi for 5 benchmarks
#                          (acasxu, collins_rul, dist_shift,
#                           linearizenn, tllverify; excludes
#                           malbeware → phase 10, metaroom → phase 6)
#                 Phase 2 = Exp 2 Tier-A (ours / Hashemi / αβ-CROWN)
#                 Phase 3 = Exp 4 (controlled scaling)
#                 Phase 4 = Exp 3 (synthetic geometry)
#                 Phase 5 = Exp 1 SaVer (5 benchmarks; same exclusions
#                           as Phase 1) + Exp 2 RS
#                 Phase 6 = Exp 1 metaroom (ours+hashemi+SaVer)
#                 Phase 7 = ablations
#                 Phase 8 = soundness audit (cifar10_resnet110)
#                 Phase 9 = Exp 2 Tier-B (ProbStar + SaVer)
#                 Phase 10 = Exp 1 malbeware (ours+hashemi+SaVer)
#                            — natural-cut tail if deadline approaches.
#                 Phase 11 = gate-only FUR stress-test (ours+hashemi
#                            re-run on every SAT-ground-truth instance
#                            with falsifier OFF; baseline tools already
#                            gate-only and read from Phase 1/2/5/6/9
#                            outputs at aggregation time). Last because
#                            it depends on the SAT-instance set already
#                            being verified by the headline runs.
#
# Aggregation runs only at the end of all 10 phases. Aggregators
# (exp1_aggregate.py etc.) are pure functions of whatever CSVs exist
# in their outputs/ directories at invocation time, so you can run
# them manually at any point between phases without disrupting the
# running sweep.
#
# The script runs FOREGROUND-SEQUENTIAL by default. Each (benchmark,
# tool) pair is one full sweep before the next starts. To parallelise,
# launch with ``--phase <N>`` per phase and run them on different
# machines / GPUs.
set -u

# ---------- config ----------
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

# Smoke (1-instance) cell-level timeout. For full sweeps the
# per-instance timeout lives inside run_cell.sh (it wraps each
# individual instance with its own ``timeout`` so a hung instance
# can't sink the rest of the cell — see VNN-COMP's
# run_single_instance.sh for the same pattern).
HARDCAP_SMOKE=600         # 10 min — 1-instance smoke per cell

# Helpers: prepend / append timestamp banners to a per-cell log file
# so we can see when each cell started, when it ended, and (by
# subtraction) how long it ran — independent of any aggregator or
# CSV inspection.
log_start() {
  local logfile="$1"; local label="$2"
  echo "[$(date -Iseconds)] === START: $label ===" > "$logfile"
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
  local exp_dir="$1"  # e.g. exp1_vnncomp_subset
  local module="$2"   # e.g. exp1_run_ours
  local args="$3"     # e.g. "--benchmark acasxu_2023"
  local logname="$4"  # e.g. exp1_acasxu_2023_ours
  local logfile="examples/FlowConformal/experiments/$exp_dir/outputs/$logname.log"

  echo "[$(date +%H:%M:%S)] >>> $logname"
  if [[ $SMOKE -eq 1 ]]; then
    # Smoke: 1 instance, no run_cell.sh wrapper. Cell-level shell
    # timeout still applies (HARDCAP_SMOKE).
    local cmd="timeout --kill-after=60 ${HARDCAP_SMOKE}s $PY -u -m examples.FlowConformal.experiments.$exp_dir.$module $args --smoke"
    echo "  [smoke] $cmd > $logfile"
    if [[ $DRY_RUN -eq 1 ]]; then
      return 0
    fi
    log_start "$logfile" "$logname"
    $cmd >> "$logfile" 2>&1
  else
    # Full sweep: per-instance shell timeout via run_cell.sh.
    local cmd="bash examples/FlowConformal/experiments/run_cell.sh $exp_dir.$module $args"
    echo "  [full] $cmd > $logfile"
    if [[ $DRY_RUN -eq 1 ]]; then
      return 0
    fi
    log_start "$logfile" "$logname"
    $cmd >> "$logfile" 2>&1
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

# Exp 1 benchmark grouping — ``CORE_5`` are the small/fast cells that
# go in Phase 1 + Phase 5. ``metaroom_2023`` and ``malbeware`` are
# multi-class CNN classifiers whose Hashemi-clipping LP cost dominates
# (250s/inst on malbeware, 196s/inst on metaroom in our smokes), so
# they get their own phases (6 and 10) — Phase 10 is the natural cut
# if we run out of time before the deadline.
EXP1_CORE5=(acasxu_2023 collins_rul_cnn_2022 dist_shift_2023
            linearizenn_2024 tllverify_2023)

# ---------- phase 1: Exp 1 ours+hashemi for the 5 core benchmarks ----
phase_1_exp1_core_ours_hashemi() {
  echo
  echo "===================== PHASE 1: Exp 1 ours+hashemi (5 core benchmarks) ====================="
  # Headline soundness numbers come from these cells. SaVer is split
  # into Phase 5; metaroom into Phase 6; malbeware into Phase 10.
  for bench in "${EXP1_CORE5[@]}"; do
    for tool in ours hashemi_clipping; do
      run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
                "--benchmark $bench" \
                "exp1_${bench}_${tool}"
    done
  done
}

# ---------- phase 1b: Exp 1 ours+hashemi for collins_rul re-run ----
phase_1b_exp1_collins_rul_ours_hashemi() {
  echo
  echo "===================== PHASE 1: Exp 1 ours+hashemi collins_rul re-run ====================="
  # Headline soundness numbers come from these cells. SaVer is split
  # into Phase 5; metaroom into Phase 6; malbeware into Phase 10.
  for bench in collins_rul_cnn_2022; do
    for tool in ours hashemi_clipping; do
      run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
                "--benchmark $bench" \
                "exp1_${bench}_${tool}"
    done
  done
}

# ---------- phase 2: Exp 2 Tier-A ----------
phase_2_exp2_tier_a() {
  echo
  echo "===================== PHASE 2: Exp 2 Tier-A ====================="
  # ours + hashemi_clipping on the 3 VNN-COMP Exp 2 benchmarks.
  # cifar10_resnet110 dropped (89% SAT verdicts come from the shared
  # APGD short-circuit so the ours-vs-hashemi comparison is
  # uninformative; no VNN-COMP consensus to anchor FUR; deprioritized
  # in favor of the 3 VNN-COMP benchmarks where consensus exists).
  # αβ-CROWN deliberately not re-run — VNN-COMP 2025 published
  # results provide its verdicts on these 3 benchmarks under their own
  # config; same-hardware re-run would be ~5-8 hr for marginal value
  # since wall-clock comparison is provided by Exp 4 (controlled
  # 1-Lipschitz scaling family on identical hardware).
  for bench in vit_2023 tinyimagenet_2024 cifar100_2024; do
    for tool in ours hashemi_clipping; do
      run_sweep exp2_prob_scale "exp2_run_$tool" \
                "--benchmark $bench" \
                "exp2_${bench}_${tool}"
    done
  done
}

# ---------- phase 3: Exp 4 (controlled scaling) ----------
phase_3_exp4() {
  echo
  echo "===================== PHASE 3: Exp 4 ====================="
  # Tool-major, depth-minor: ours through every depth first, then
  # Hashemi, then sound verifiers. Lets the log reader watch a single
  # tool's full scaling curve before the next starts.
  for tool in ours hashemi_clipping alpha_beta_crown neuralsat; do
    for depth in 2 4 8 16 24 32 40; do
      run_sweep exp4_scaling "exp4_run_$tool" \
                "--depth $depth" \
                "exp4_d${depth}_${tool}"
    done
  done
}

# ---------- phase 4: Exp 3 (synthetic geometry) ----------
phase_4_exp3() {
  echo
  echo "===================== PHASE 4: Exp 3 ====================="
  for bench in 3d_banana synth_5d synth_10d synth_20d; do
    for spec in unsat sat; do
      # All four score families are wired (flow, hyperrect, ellipsoid,
      # gmm). Hyperrect/ellipsoid/gmm are closed-form; flow uses bounded
      # AMLS. The full sweep produces the 4-score-family geometry plot.
      for score in flow hyperrect ellipsoid gmm; do
        run_sweep exp3_synthetic exp3_run_ours \
                  "--benchmark $bench --score $score --spec $spec" \
                  "exp3_${bench}_${score}_${spec}_ours"
      done
      run_sweep exp3_synthetic exp3_run_hashemi_clipping \
                "--benchmark $bench --spec $spec" \
                "exp3_${bench}_${spec}_hashemi_clipping"
    done
  done
}

# ---------- phase 5: Exp 1 SaVer (5 core) + Exp 2 RS ----------
phase_5_exp1_saver_and_exp2_rs() {
  echo
  echo "===================== PHASE 5: Exp 1 SaVer (5 core) + Exp 2 RS ====================="
  # Exp 1 SaVer for the 5 core benchmarks (metaroom → Phase 6,
  # malbeware → Phase 10). SaVer is sample-based DKW; cheap per
  # instance regardless of network size.
  for bench in "${EXP1_CORE5[@]}"; do
    run_sweep exp1_vnncomp_subset exp1_run_saver \
              "--benchmark $bench" \
              "exp1_${bench}_saver"
  done
  # RS (Cohen randomized smoothing) on cifar100_2024 only. cifar10_resnet110
  # dropped from the sweep entirely (see Phase 2 comment).
  run_sweep exp2_prob_scale exp2_run_rs \
            "--benchmark cifar100_2024" \
            "exp2_cifar100_2024_rs"
}

# ---------- phase 6: Exp 1 metaroom (ours, hashemi-clipping-PCA, saver) ----------
phase_6_exp1_metaroom() {
  echo
  echo "===================== PHASE 6: Exp 1 metaroom (ours+hashemi-PCA+saver) ====================="
  # metaroom_2023 separated from Phase 1 because Hashemi-clipping LP
  # cost on its 20-class output is borderline at the 100s VNN-COMP
  # budget. We use Hashemi-PCA (K=10) here per the published
  # clipping-block paper's recommended config for medium-output-dim
  # benchmarks.
  run_sweep exp1_vnncomp_subset exp1_run_ours \
            "--benchmark metaroom_2023" "exp1_metaroom_2023_ours"
  run_sweep exp1_vnncomp_subset exp1_run_hashemi_clipping_pca \
            "--benchmark metaroom_2023 --pca-components 10" \
            "exp1_metaroom_2023_hashemi_clipping_pca"
  run_sweep exp1_vnncomp_subset exp1_run_saver \
            "--benchmark metaroom_2023" "exp1_metaroom_2023_saver"
}

# ---------- phase 2-prime: Hashemi-PCA re-run on cifar100/tinyimagenet ----------
phase_2prime_hashemi_pca_rerun() {
  echo
  echo "===================== PHASE 2': Hashemi-PCA re-run (cifar100, tinyimagenet) ====================="
  # The original Phase 2 Hashemi runs without PCA TIMEOUTed on every
  # would-be UNSAT instance for cifar100 (out=100) and tinyimagenet
  # (out=200) because raw clipping_block solves m=8000 LPs per
  # instance with 2*output_dim constraints each. The published
  # clipping-block paper (Hashemi et al. 2025) uses deflation-PCA
  # before the convex-hull projection on high-output-dim networks; we
  # mirror that here with K=32. Original (no-PCA) CSVs are preserved
  # in-place as ``exp2_<bench>_hashemi_clipping.csv`` for paper
  # citation; new outputs land in ``exp2_<bench>_hashemi_clipping_pca.csv``.
  # vit_2023 is not re-run — its existing Phase 2 row already has
  # 100/100 UNSAT under raw clipping (output dim 9, no PCA needed).
  for bench in cifar100_2024 tinyimagenet_2024; do
    # m=750 (down from default 8000) reduces the LP count from 8000 to
    # 750. PCA K=32 still reduces per-LP constraints from 200 to 64.
    # Together this brings per-instance wall to ~50-60s — comparable
    # to ours' wall on the same benchmarks (58-65s) and fits the 100s
    # VNN-COMP budget. See conversation 2026-05-03 for the wall sweep.
    run_sweep exp2_prob_scale exp2_run_hashemi_clipping_pca \
              "--benchmark $bench --pca-components 32 --m 750" \
              "exp2_${bench}_hashemi_clipping_pca"
  done
}

# ---------- phase 5b: Exp 1 SaVer re-run with fixed delta ----------
phase_5b_saver_rerun() {
  echo
  echo "===================== PHASE 5b: Exp 1 SaVer re-run (fixed delta=0.05) ====================="
  # The original Phase 5 SaVer runs used delta=0.001 with DKW
  # epsilon=0.01, which is mathematically incapable of certifying
  # UNSAT (DKW bound 0.01 > delta 0.001 always). Bumped to delta=0.05
  # in baselines/run_saver.py. Re-run all 5 core SaVer cells. Existing
  # CSVs need to be deleted before re-launch so we don't append.
  for bench in acasxu_2023 collins_rul_cnn_2022 dist_shift_2023 \
               linearizenn_2024 tllverify_2023; do
    run_sweep exp1_vnncomp_subset exp1_run_saver \
              "--benchmark $bench" \
              "exp1_${bench}_saver"
  done
}

# ---------- phase 6prime: new multi-output benchmark cells ----------
phase_6prime_lsnc_relu() {
  echo
  echo "===================== PHASE 6': Exp 1 lsnc_relu (ours+hashemi) ====================="
  # New multi-output control benchmark added for the
  # geometry-advantage story (output dim 8, similar to ACAS Xu's 5).
  # Use raw clipping_block at the budget-fitted m=4000 (set in
  # PER_BENCHMARK_CONFIG). SaVer runs in Phase 9b so we don't
  # double-run it here.
  for tool in ours hashemi_clipping; do
    run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
              "--benchmark lsnc_relu" \
              "exp1_lsnc_relu_${tool}"
  done
}

phase_6prime_relusplitter() {
  echo
  echo "===================== PHASE 6': Exp 1 relusplitter (ours+hashemi) ====================="
  # New MNIST classifier benchmark (output dim 10). Same per-benchmark
  # m=4000 reduction; SaVer runs in Phase 9b.
  for tool in ours hashemi_clipping; do
    run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
              "--benchmark relusplitter" \
              "exp1_relusplitter_${tool}"
  done
}

# ---------- phase 6b: ours-only re-run on metaroom + lsnc_relu after gate revert ----------
phase_6b_metaroom_ours_rerun() {
  echo
  echo "===================== PHASE 6b: metaroom OURS re-run (strict gate restored) ====================="
  # Phase 6's metaroom ours data was generated with the relaxed gate
  # (pi_upper-only), which produced 2 false UNSATs against αβ-CROWN's
  # SAT verdicts. Strict gate is restored
  # (not detected_unsafe AND pi_upper <= eps_2_target), re-run ours.
  # Hashemi-PCA + SaVer don't use the gate, their data stays valid.
  run_sweep exp1_vnncomp_subset exp1_run_ours \
            "--benchmark metaroom_2023" "exp1_metaroom_2023_ours"
}

phase_6b_lsnc_relu_ours_rerun() {
  echo
  echo "===================== PHASE 6b: lsnc_relu OURS re-run (strict gate restored) ====================="
  # Phase 6'\''s lsnc_relu ours data had 11 false UNSATs out of 12
  # GT-SAT under the relaxed gate (the SAT instances\' rare-but-existent
  # unsafe regions were detected by AMLS but the mass bound passed).
  # Strict gate restored, re-run ours. Hashemi-clipping data (under-budget
  # m=4000) is gate-independent and stays valid.
  run_sweep exp1_vnncomp_subset exp1_run_ours \
            "--benchmark lsnc_relu" "exp1_lsnc_relu_ours"
}

# ---------- phase 2'': Hashemi-PCA re-run on cifar100/tinyimagenet at m=2500 ----------
phase_2pp_hashemi_pca_rerun_m2500() {
  echo
  echo "===================== PHASE 2'': Hashemi-PCA re-run at m=2500 (cifar100, tinyimagenet) ====================="
  # Phase 2' ran with m=750 (~9-10s wall, ~6× faster than ours).
  # Reviewer concern: Hashemi was given dramatically less budget than
  # ours. Smoke probe shows m=2500 puts Hashemi-PCA wall at:
  #   cifar100:    ~50s (vs ours 58.6s)
  #   tinyimagenet: 50.5s (vs ours 56.1s)
  # Comparable wall, well under the 100s VNN-COMP budget. m=4000 was
  # over budget (132s on cifar100); m=3000 starts pushing past ours
  # on tinyimagenet (76s).
  for bench in cifar100_2024 tinyimagenet_2024; do
    run_sweep exp2_prob_scale exp2_run_hashemi_clipping_pca \
              "--benchmark $bench --pca-components 32 --m 2500" \
              "exp2_${bench}_hashemi_clipping_pca"
  done
}

# ---------- phase 7: ablations ----------
phase_7_ablations() {
  echo
  echo "===================== PHASE 7: ablations ====================="
  # Per-axis ablation rows on the locked Phase 5d pipeline. See
  # examples/FlowConformal/experiments/exp_ablation/README.md for
  # axis definitions, value sets, and per-row wall-clock estimates.
  # Total: ~7-8 hours sequentially.
  local logdir="examples/FlowConformal/experiments/exp_ablation/outputs"
  mkdir -p "$logdir"
  # Slimmed to verify_method only. Other axes (conformal_params,
  # amls_hparam, flow_training, score) are appendix-grade sensitivity
  # studies; their omission is documented in the supplementary code.
  # Score-function ablation is covered by Exp 3 (volume tightness).
  for runner in verify_method; do
    local module="examples.FlowConformal.experiments.exp_ablation.ablation_run_${runner}"
    local logfile="$logdir/ablation_${runner}.log"
    echo "[$(date +%H:%M:%S)] >>> ablation: $runner"
    if [[ $SMOKE -eq 1 ]]; then
      local cmd="timeout --kill-after=60 ${HARDCAP_SMOKE}s $PY -u -m $module --smoke"
    else
      local cmd="$PY -u -m $module"
    fi
    echo "  $cmd > $logfile"
    if [[ $DRY_RUN -eq 0 ]]; then
      log_start "$logfile" "ablation_${runner}"
      $cmd >> "$logfile" 2>&1 || true
      log_end "$logfile" "ablation_${runner}" "$?"
    fi
  done
  # Aggregate at the end so the markdown table is up to date.
  if [[ $DRY_RUN -eq 0 ]]; then
    local agg_log="$logdir/ablation_aggregate.log"
    log_start "$agg_log" "ablation_aggregate"
    $PY -u -m examples.FlowConformal.experiments.exp_ablation.ablation_aggregate \
        >> "$agg_log" 2>&1 || true
    log_end "$agg_log" "ablation_aggregate" "$?"
  fi
}

# ---------- phase 8: REMOVED (was: soundness audit on cifar10_resnet110) ----------
# Audit was scoped only to cifar10_resnet110 (the one Exp 2 benchmark
# with no VNN-COMP consensus). Since cifar10_resnet110 is dropped from
# the sweep entirely (see Phase 2 comment), there's nothing left to
# audit. Soundness on remaining benchmarks is established via FUR vs
# VNN-COMP 2025 8-tool consensus (in the aggregator output).

# ---------- phase 9: Exp 2 Tier-B (ProbStar + SaVer) ----------
phase_9_exp2_tier_b() {
  echo
  echo "===================== PHASE 9: Exp 2 Tier-B (ProbStar + SaVer) ====================="
  # Tier-B baselines: ProbStar (StarV / Tran et al.) and SaVer
  # (Convertino HSCC 2025). ProbStar uniformly emits NOT_APPLICABLE
  # on Exp 2's networks (StarV's loader doesn't support transformer
  # attention, residual Add, or Gemm nodes). SaVer is sample-based
  # and applies to any network. cifar10_resnet110 dropped along with
  # the rest of the sweep (see Phase 2 comment).
  for bench in vit_2023 tinyimagenet_2024 cifar100_2024; do
    for tool in probstar saver; do
      run_sweep exp2_prob_scale "exp2_run_$tool" \
                "--benchmark $bench" \
                "exp2_${bench}_${tool}"
    done
  done
}

# ---------- phase 9b: Tier-B baselines for new Exp 1 multi-output benches ----
phase_9b_lsnc_relu_tier_b() {
  echo
  echo "===================== PHASE 9b: lsnc_relu Tier-B (ProbStar + SaVer) ====================="
  for tool in probstar saver; do
    run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
              "--benchmark lsnc_relu" \
              "exp1_lsnc_relu_${tool}"
  done
}

phase_9b_relusplitter_tier_b() {
  echo
  echo "===================== PHASE 9b: relusplitter Tier-B (ProbStar + SaVer) ====================="
  for tool in probstar saver; do
    run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
              "--benchmark relusplitter" \
              "exp1_relusplitter_${tool}"
  done
}

# ---------- phase 10: Exp 1 malbeware (all three tools) — natural cut tail ----
phase_10_exp1_malbeware() {
  echo
  echo "===================== PHASE 10: Exp 1 malbeware (ours+hashemi+saver) ====================="
  # malbeware separated from Phase 1 because Hashemi-clipping LP cost
  # on its 25-class output is ~250 s/instance (~9.1 hr for the full
  # 150-instance cell). Placed last so it's the natural cut if we run
  # out of time before the deadline. Headline Exp 1 tables can either
  # include this row (if Phase 10 finishes) or omit it with a single-
  # sentence appendix note (if cut).
  for tool in ours hashemi_clipping saver; do
    run_sweep exp1_vnncomp_subset "exp1_run_$tool" \
              "--benchmark malbeware" \
              "exp1_malbeware_${tool}"
  done
}

# ---------- phase 11: gate-only FUR stress-test ----------
phase_11_gate_fur_study() {
  echo
  echo "===================== PHASE 11: gate-only FUR stress-test ====================="
  # Re-runs ours + Hashemi-clipping on every SAT-ground-truth instance
  # with use_falsifier=False, so the verdict gate is exposed directly to
  # every SAT instance (instead of being preempted by APGD on most of
  # them). Isolates the intrinsic gate FUR from the deployed-system FUR
  # measured in Phase 1+2.
  #
  # Other probabilistic baselines (ProbStar / SaVer / RS) never invoked
  # APGD in their pipelines, so their existing Phase 1/2/5/6/9 SAT-only
  # rows are already gate-only and need no re-run — gate_fur_study/
  # aggregate.py reads them in place.
  #
  # Skips vit_2023 (zero SAT-ground-truth instances).
  local gate_benches=(acasxu_2023 collins_rul_cnn_2022 dist_shift_2023
                      linearizenn_2024 malbeware metaroom_2023 tllverify_2023
                      tinyimagenet_2024 cifar100_2024)
  for bench in "${gate_benches[@]}"; do
    for tool in ours hashemi; do
      run_sweep gate_fur_study "run_${tool}_no_falsifier" \
                "--benchmark $bench" \
                "gate_fur_${bench}_${tool}_no_falsifier"
    done
  done
}

# ---------- aggregation ----------
aggregate_all() {
  echo
  echo "===================== AGGREGATION ====================="
  for cmd in \
    "examples.FlowConformal.experiments.exp1_vnncomp_subset.exp1_aggregate" \
    "examples.FlowConformal.experiments.exp2_prob_scale.exp2_aggregate" \
    "examples.FlowConformal.experiments.exp3_synthetic.exp3_aggregate" \
    "examples.FlowConformal.experiments.exp4_scaling.exp4_aggregate" \
    "examples.FlowConformal.experiments.gate_fur_study.aggregate"; do
    echo "  $PY -m $cmd"
    if [[ $DRY_RUN -eq 0 ]]; then
      $PY -m "$cmd" 2>&1 | tail -2
    fi
  done
}

# ---------- dispatch ----------
case "$PHASE" in
  1)  phase_1_exp1_core_ours_hashemi ;;
  2)  phase_2_exp2_tier_a ;;
  2prime|2p) phase_2prime_hashemi_pca_rerun ;;
  2pp|2primeprime) phase_2pp_hashemi_pca_rerun_m2500 ;;
  6b_metaroom) phase_6b_metaroom_ours_rerun ;;
  6b_lsnc) phase_6b_lsnc_relu_ours_rerun ;;
  3)  phase_3_exp4 ;;
  4)  phase_4_exp3 ;;
  5)  phase_5_exp1_saver_and_exp2_rs ;;
  5b) phase_5b_saver_rerun ;;
  6)  phase_6_exp1_metaroom ;;
  6prime_lsnc|6p_lsnc) phase_6prime_lsnc_relu ;;
  6prime_relu|6p_relu) phase_6prime_relusplitter ;;
  7)  phase_7_ablations ;;
  8)  echo "phase 8 (soundness audit) was removed — cifar10_resnet110 dropped from sweep" >&2 ;;
  9)  phase_9_exp2_tier_b ;;
  9b_lsnc) phase_9b_lsnc_relu_tier_b ;;
  9b_relu) phase_9b_relusplitter_tier_b ;;
  10) phase_10_exp1_malbeware ;;
  11) phase_11_gate_fur_study ;;
  all)
    # ─── COMPLETED IN PRIOR RUN — leave commented ───
    # phase_1_exp1_core_ours_hashemi          # done (Phase 1)
    # phase_2_exp2_tier_a                     # done (Phase 2; cifar100/tiny TIMEOUTed → see Phase 2'')
    # phase_1b_exp1_collins_rul_ours_hashemi  # done (Phase 1b after loader fix)
    # phase_3_exp4                            # done (Exp 4 scaling)
    # phase_5_exp1_saver_and_exp2_rs          # SaVer done but invalid (delta bug); RS for cifar100 OK; see Phase 5b
    # phase_2prime_hashemi_pca_rerun          # done at m=750; will re-run at m=2500 in Phase 2''
    # phase_4_exp3                            # done with all Exp 3 fixes (falsifier strip, MC volume, AND-of-OR distribution)

    # ─── PHASE 5b: Exp 1 SaVer re-run with delta=0.05 ───
    # Pre-launch action: delete the 5 buggy delta=0.001 SaVer CSVs:
    #   exp1_{acasxu_2023,collins_rul_cnn_2022,dist_shift_2023,
    #         linearizenn_2024,tllverify_2023}_saver.{csv,log}
    phase_5b_saver_rerun

    # ─── PHASE 6b: metaroom + lsnc_relu OURS-only re-runs ───
    # The previous Phase 6 + 6' runs used the (then-relaxed) gate which
    # produced 2 false UNSATs on metaroom and 11 false UNSATs on
    # lsnc_relu. Strict gate is restored
    # (not detected_unsafe AND pi_upper <= eps_2_target). Hashemi data
    # for both benches doesn't go through the gate so it stays valid;
    # only ours needs re-running.
    # Pre-launch action: delete the existing ours CSVs:
    #   exp1_{metaroom_2023,lsnc_relu}_ours.{csv,log}
    phase_6b_metaroom_ours_rerun
    phase_6b_lsnc_relu_ours_rerun

    # ─── PHASE 6': relusplitter (full, both methods) ───
    # Previous run was killed mid-way at instance 11 (ours under the
    # relaxed gate). Full re-run with strict gate; Hashemi never
    # started so this is fresh for both.
    phase_6prime_relusplitter

    # ─── PHASE 9: Exp 2 Tier-B (ProbStar + SaVer) ───
    phase_9_exp2_tier_b

    # ─── PHASE 9b: Tier-B for new Exp 1 benches ───
    phase_9b_lsnc_relu_tier_b
    phase_9b_relusplitter_tier_b

    # ─── PHASE 7: ablations ───
    phase_7_ablations

    # ─── PHASE 2'': Hashemi-PCA cifar100/tinyimagenet at m=2500 ───
    # Replaces Phase 2'-at-m=750 with a config that matches ours' wall
    # (~50s vs ours ~58s), removing the budget-fairness concern.
    # Pre-launch action: delete the m=750 CSVs:
    #   exp2_{cifar100_2024,tinyimagenet_2024}_hashemi_clipping_pca.{csv,log}
    phase_2pp_hashemi_pca_rerun_m2500

    # ─── PHASE 10: malbeware (fallback only) ───
    # phase_10_exp1_malbeware

    # ─── PHASE 11: gate-only FUR stress-test ───
    phase_11_gate_fur_study

    aggregate_all
    ;;
  *) echo "unknown phase: $PHASE (valid: 1..7, 9, 9b_lsnc, 9b_relu, 10, 11, 2prime, 2pp, 5b, 6b_metaroom, 6b_lsnc, 6prime_lsnc, 6prime_relu, or 'all')" >&2; exit 2 ;;
esac

echo
echo "[$(date +%H:%M:%S)] === run_all_sweeps complete ==="
