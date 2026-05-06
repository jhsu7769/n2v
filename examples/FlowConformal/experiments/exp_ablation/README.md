# Ablation experiment

Per-row ablation that varies one design choice at a time on the locked
Phase 5d pipeline (flow + AMLS + post-falsifier-fix calibration).
Results are written to `outputs/ablation_<row>_<value>.csv` and
aggregated into a single markdown table by `ablation_aggregate.py`.

## Rows

| axis | values | bench | script |
|---|---|---|---|
| Score function | hyperrect, ellipsoid (Mahalanobis), GMM(10) [TODO], flow | 3D banana | `ablation_run_score.py` |
| Verification method | scenario, scenario_v2, amls, is_tilted, derived, amls_bounded, amls_bounded_union | ACAS Xu probe | `ablation_run_verify_method.py` |
| AMLS hyperparameters | rho in {0.05, 0.1, 0.2}; mcmc_steps in {5, 10, 20, 40} | ACAS Xu probe | `ablation_run_amls_hparam.py` |
| Conformal parameters | alpha in {0.001, 0.01, 0.05, 0.1}; m in {500, 2000, 8000}; ell-offset in {0, 1, 5}; beta_2 in {0.001, 0.01, 0.1} | ACAS Xu probe | `ablation_run_conformal_params.py` |
| Flow training | n_train in {1K, 2K, 5K, 10K, 20K, 50K} x flow_epochs in {500, 1000, 2000, 5000} | ACAS Xu probe | `ablation_run_flow_training.py` |

## Benchmark: 10-instance ACAS Xu probe

The probe instance list is defined in
`examples/FlowConformal/ablations/phase5c_probe_sweep.py::_INSTANCES`
(20 candidates: 4 persistent false-UNSATs from Phase 5b + 5
P4->P5b regression cases + 11 typical UNSAT controls). The Phase 7
runners use the first 10 by default (per the 2026-04-27 paper-experiments
plan; pass `--probe-size 20` to use the full list). Per-instance
wall-clock is ~30-90 s with the locked pipeline; per ablation value
the full 10-instance probe is ~6-15 min.

## Run order

1. **Smoke each script first** (each takes ~1-3 min, runs 1 ablation
   value on the first 2 probe instances):

   ```sh
   cd /home/sasakis/v/tools/n2v
   PY=/home/sasakis/miniconda3/envs/n2v/bin/python
   for s in verify_method calib_size amls_hparam flow_training score; do
     timeout 300 $PY -u -m \
       examples.FlowConformal.experiments.exp_ablation.ablation_run_$s \
       --smoke 2>&1 | tail -10
   done
   ```

2. **Full runs.** Each script is independent; run sequentially or in
   parallel as resources allow:

   ```sh
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_run_verify_method
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_run_calib_size
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_run_amls_hparam
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_run_flow_training
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_run_score
   ```

3. **Aggregate** into a single markdown table:

   ```sh
   $PY -u -m examples.FlowConformal.experiments.exp_ablation.\
ablation_aggregate --out docs/audits/2026-XX-XX-exp-ablation.md
   ```

## Wall-clock estimates (full)

| script | per-value | full row |
|---|---|---|
| `ablation_run_verify_method.py` | 12-30 min | ~100 min (5 methods) |
| `ablation_run_calib_size.py` | 14-22 min | ~55 min (3 values) |
| `ablation_run_amls_hparam.py` | 12-30 min | ~135 min (3 + 4 values) |
| `ablation_run_flow_training.py` | 18-22 min | ~125 min (6 values) |
| `ablation_run_score.py` | <1 min/seed for hyperrect/ellipsoid; ~3-5 min/seed for flow | ~25 min (5 seeds, 3 scores) |

Total: roughly 7-8 hours of compute for the full ablation.

## Known gaps and TODOs

- **AMLS hyperparameter knobs are not exposed at the
  `run_verification_pipeline` level**. The ablation script uses a
  monkey-patch on `n2v.probabilistic.flow.amls.amls_certify_spec`; this
  is sound because the pipeline is the only caller. Fix by adding
  `amls_quantile` / `amls_n_mcmc_steps` / `amls_mcmc_step_size` kwargs
  through `_flow_unsat_pipeline` and `run_verification_pipeline`.
- **Flow training knobs (coupling / ema / standardize) are not
  exposed** at the pipeline level either. Same monkey-patch pattern;
  fix by adding kwargs through `_train_flow` and the public pipeline
  signature.
- **`standardize_outputs=True` in the ablation triggers double
  whitening** because `run_verification_pipeline` already pre-whitens
  outputs before training. A "true" no-pre-whitening row would also
  need to disable the pipeline's whitening glue (see "Whitening glue
  for run_verification_pipeline" comment in
  `examples/FlowConformal/benchmarks/_common.py`). Documented in the
  script docstring; left as future work.
- **GMM(10) score is not implemented**. There is no `GMMScore` class
  in `n2v/probabilistic/flow/scores.py`. A negative-log-density score
  under a fitted `sklearn.mixture.GaussianMixture` would be ~30 LOC to
  add; sketch is in the script docstring. Until then, the row is
  written as `nan` and skipped.
## Output layout

```
exp_ablation/
  outputs/
    ablation_verify_method_<method>.csv          (one per method)
    ablation_amls_hparam_rho<rho>.csv            (one per rho)
    ablation_amls_hparam_mcmc<steps>.csv         (one per mcmc step count)
    ablation_conformal_params_alpha<v>.csv       (one per alpha)
    ablation_conformal_params_m<m>.csv           (one per m)
    ablation_conformal_params_elloff<v>.csv      (one per ell-offset)
    ablation_conformal_params_beta2<v>.csv       (one per beta_2)
    ablation_flow_training_n<N>_e<E>.csv         (one per (n_train, flow_epochs) cell)
    ablation_score.csv                           (3D banana; single CSV with `score` column)
    ablation_score_smoke.csv                     (smoke output)
```

Each ACAS-Xu CSV has the schema:
`onnx_file, vnnlib_file, verdict, q, worst_max_margin, amls_levels_used, wall_s, error`.

The aggregator (`ablation_aggregate.py`) reads ONLY from this `outputs/`
directory — there are no fallback paths into legacy locations. Any
missing cell renders as "(missing)" in the report instead of silently
substituting older methodology data.
