# Experiment 3: Synthetic Validation

Validates the flow-conformal framework on synthetic benchmarks where the
exact (or near-exact) reach-set volume is known. We compare **OUR
method only** here (verification_method='amls_bounded', Phase 5d locked
config); third-party probabilistic baselines are wired in separately.

See `docs/plans/2026-04-27-paper-experiments-design.md` (Experiment 3).

## Benchmarks

1. **3D banana classifier** (`ThreeBlobClassifier3D`) — 3D inputs in
   [-1, 1]^3, 3D output logits. Exact reach-set volume estimated from
   the cached Star-union (~213.72; (1-α)·V is the tightness floor).
2. **Higher-dim 1-Lipschitz nets** (5D, 10D, 20D) — identity-activation
   nets so the composed map is purely linear and the reach set is a
   zonotope with closed-form volume `|det(W_total)| · prod(ub - lb)`.
   Input box: `[-0.5, 0.5]^dim`. Halfspace query: `y[0] >= 1e6`
   (always UNSAT — we are testing volume tightness, not verdict
   correctness).

## Files

- `networks.py` — `OneLipschitzNet`, plus `make_synthetic_5d/10d/20d`.
- `exact_volumes.py` — `exact_volume_linear_net` (closed form for the
  identity-activation case) and `mc_ground_truth_volume` (fallback for
  nonlinear activations).
- `exp3_run_3d_banana.py` — runs AMLS pipeline on the 3D banana
  benchmark across `K` seeds.
- `exp3_run_synthetic.py` — runs AMLS pipeline on the 5D/10D/20D nets
  across `K` seeds.
- `outputs/` — CSVs are written here at runtime.

## Locked Phase 5d config

| param | value |
|---|---|
| verification_method | amls_bounded |
| amls_max_levels | 30 |
| alpha | 0.001 |
| n_train | 5000 (synth_5/10/20d); 2000 (3d_banana) |
| flow_epochs | 2000 |
| flow_config | base (h128/L4) |
| scenario_n_samples | 2000 |
| scenario_beta | 0.001 |

## How to run

Smoke (≤ 60s per script; reduces n_train, flow_epochs, scenario_n,
1 seed only):

```bash
cd /home/sasakis/v/tools/n2v
/home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_run_3d_banana --smoke

/home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_run_synthetic --smoke
```

Full (5 seeds; outputs go to `outputs/exp3_*.csv`):

```bash
/home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_run_3d_banana

/home/sasakis/miniconda3/envs/n2v/bin/python -u -m \
    examples.FlowConformal.experiments.exp3_synthetic.exp3_run_synthetic
```

## Output schema

`outputs/exp3_3d_banana_ours.csv`:

| col | meaning |
|---|---|
| seed | calibration seed |
| verdict | UNSAT / UNKNOWN / SAT |
| q | calibrated conformal threshold |
| volume_estimate | MC volume of `{y : score(y) <= q}` |
| volume_ratio_vs_exact | volume_estimate / `(1-α)·Star_union_vol` |
| coverage_empirical | empirical (1-α)-coverage on a 2k test set |
| train_s, verify_s, total_s | wall-clock breakdown |

`outputs/exp3_synthetic_ours.csv` adds a `dim` column; the rest is the
same. The exact-volume reference is `(1-α)·|det(W_total)|·prod(ub-lb)`
(closed form, identity-activation linear net).

## Expected wall-clock per benchmark

- 3D banana: training ~120s + verify ~30s + Star-union MC ~30s ≈ 3 min/seed.
- 5D synthetic: ~2 min/seed; 10D ~3 min/seed; 20D ~5 min/seed.
- Full sweep: 5 seeds × (3 min for 3D + 2+3+5 min for 5/10/20D) ≈
  ~65 min total sequentially.

## Reference

Design: `docs/plans/2026-04-27-paper-experiments-design.md` §
"Experiment 3: Synthetic Validation".
