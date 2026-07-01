# Running FairN2V on the medical-costs (insurance) model

Fairness verification of a **regression** model that predicts medical insurance **charges (\$)**.
Sensitive attribute: **sex**. The verifier proves, per person, whether changing sex (and optionally
wiggling age/BMI a little) can move the predicted cost outside a tolerance band of **±\$δ**.

> This is the regression extension of FairN2V. Classification datasets (adult, german, …) ask "does the
> predicted class flip?"; here we ask "does the predicted dollar amount leave the ±δ band?"

---

## 0. One-time setup

From the repo root, with the project venv active and `n2v` installed:

```bash
cd examples/FairN2V

# 1) Build the dataset (downloads insurance.csv via kagglehub, writes data/medcost_data.npz)
python dataset_prep/make_medcost_npz.py

# 2) Train + export the regression models (writes models/MC-1.onnx, MC-2.onnx, MC-3.onnx)
python dataset_prep/train_medcost_reg.py
```

`make_medcost_npz.py` produces a 9-feature representation (raw values; the loader normalizes):

```
idx: 0    1    2         3     4       5..8
     age, sex, bmi, ...  ->  actually [age, bmi, children, sex, smoker, region_NE/NW/SE/SW]
              order is fixed by FEATURE_ORDER; sex is index 3
```

`train_medcost_reg.py` trains 3 sizes (small/medium/large), standardizes the target for stable
training, then **folds the inverse-scaling into a final linear layer so the ONNX outputs dollars**.
Expect MAE ≈ \$2.4k–3.8k.

---

## 1. Run the verifier

```bash
python run_fairn2v.py --dataset medcost --num-obs 100 --delta 500
```

Flags:

| Flag | Meaning | Default |
|---|---|---|
| `--dataset medcost` | Select the medical-costs regression profile | required |
| `--delta` | Tolerance band half-width, **in dollars** | adapter default (\$500) |
| `--num-obs` | Number of people (test samples) to verify | 100 |
| `--models` | Subset of models, e.g. `--models MC-1` | all three (MC-1/2/3) |

Useful variants:

```bash
# fast smoke test
python run_fairn2v.py --dataset medcost --num-obs 5 --models MC-1 --delta 500

# sweep the tolerance to see how strict "fair" is
python run_fairn2v.py --dataset medcost --num-obs 100 --delta 1000
python run_fairn2v.py --dataset medcost --num-obs 100 --delta 2000
```

---

## 2. What it actually checks

Per person, for each ε in the grid `{0.0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1}`:

1. **Flip sex** (the counterfactual).
2. **Widen age & BMI by ±ε** — note ε is in **normalized [0,1] feature space**, so ε=0.05 ≈ 5% of each
   feature's full range (age span 18–64 → ±~2.3 yrs; BMI span ~16–53 → ±~1.9).
3. Compute the **exact** set of possible predicted costs (reachability).
4. **Fair** iff that whole set stays within `[y_ref − δ, y_ref + δ]`, where `y_ref` is the model's
   prediction on the *original* person.

Two notions fall out of the ε grid:

- **ε = 0 → counterfactual fairness:** pure sex flip. *"Does changing only sex move the cost by >δ?"*
  This is the cleanest measure of sex-dependence.
- **ε > 0 → individual fairness:** sex flip **plus** an age/BMI perturbation. This conflates
  sex-sensitivity with general input sensitivity (a Lipschitz-style robustness measure), so read it as
  "how stable is the prediction in a neighborhood," not pure sex-fairness.

Verdicts are **exact** (`method='exact'`), so "fair" is a proof, not a spot check. `unknown` only
appears on timeout.

---

## 3. Outputs

Each run writes `results/<timestamp>/`:

| File | Contents |
|---|---|
| `counterfactual_*.csv` | per-model fair / unfair % at ε=0 |
| `individual_*.csv` | per-model × ε fair / unfair / unknown % |
| `timing_*.csv` | per-model × ε timing |
| `individual_fairness_combined.png/.pdf` | stacked-area fairness-vs-ε figure |
| `*_table.tex` | LaTeX tables |

---

## 4. Reading the numbers

- Higher **FairPercent** = more people for whom the property is certified.
- Compare **across δ**: a model that's "unfair" at δ=\$500 but "fair" at δ=\$2000 means flipping sex moves
  cost by something between \$500 and \$2000 for those people.
- Compare **across models**: smaller nets (MC-1) tend to be fairer (less spurious sex-dependence);
  larger nets fit harder and pick up more.
- Compare **across ε**: the decay rate is the model's input sensitivity — a flat curve is robust, a
  steep one is brittle.

See `claude-planning/` for the design/implementation write-ups and a sample interpreted result set.

---

## 5. Gotchas

- δ is in **dollars** (the ONNX outputs dollars via the folded de-scale layer). If you ever retrain
  without that layer, δ would be in standardized units instead.
- ε is in **normalized feature units**, not raw years/BMI points.
- `sex` must stay at **index 3** (set in `make_medcost_npz.py:FEATURE_ORDER`); the adapter's
  `sensitive_features=[3]` depends on it.
- To verify on a held-out test split, pass `config['sample_indices']` (a list of row indices) instead
  of `--num-obs`; normalization stays consistent because the npz holds the whole dataset.
