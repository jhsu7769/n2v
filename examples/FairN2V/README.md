# FairN2V - Fairness Verification of Tabular Classifiers

Exact reachability-based verification of two notions of fairness on
binary classifiers trained on tabular datasets — UCI Adult-Income,
German Credit, and Bank Marketing — selected at run time via `--dataset`:

- **Counterfactual fairness** — flipping a sensitive attribute (e.g.
  sex) must not change the prediction. Verified at ε = 0.
- **Individual fairness** — for every input, no perturbation within an
  ε-ball (combined with a flip of the sensitive attribute) changes the
  prediction. Verified across multiple ε values.

Each verdict is summarized as a Verified Fairness (VF) score: the
proportion of test samples for which fairness is formally certified.

This example is a Python port — for the [`n2v`](../../) toolbox — of the
MATLAB **FairNNV** example that ships with NNV. The verification logic, models,
and datasets are carried over directly; see [References](#references).

All four profiles have been checked against NNV on MATLAB's exact seed-500
samples (`rng(500); randsample(...)`), across both fairness notions and the
paper's full ε grid. **Adult**, **`adult_debiased`**, and **`bank`** reproduce
the NNV results *exactly* — bit-for-bit on every model and ε (48 of 48 cells).
**`german`** matches everywhere except 6 cells, each off by at most 1: those
are the rows that sit at a perturbable feature's extreme, where this port
builds a valid input box but MATLAB fed `ImageStar` a degenerate `lb > ub`
one. The divergence is therefore a deliberate soundness fix, not a mismatch —
matching NNV exactly there would mean reproducing the bug.

## References

- **FairNNV**: Tumlin, A.M., Manzanas Lopez, D., Robinette,
  P., Zhao, Y., Derr, T., Johnson, T.T. *FairNNV: The neural network
  verification tool for certifying fairness.* Proceedings of the 5th
  ACM International Conference on AI in Finance (ICAIF '24), 2024.
- **Counterfactual fairness definition**: Kusner, M.J., Loftus, J.R.,
  Russell, C., Silva, R. *Counterfactual fairness.* NeurIPS 2017.
- **Adult-Income dataset**: Dheeru & Efi. *UCI Machine Learning
  Repository — Adult.* 2017.
- **German-Credit dataset**: Dheeru, D., & Efi, K. T. *UCI Machine 
  Learning Repository: Statlog (German Credit Data) data set.* 2017. 
- **Bank-Marketing dataset**: Moro, S., Cortez, P., & Rita, P. *UCI 
  Machine Learning Repository: Bank Marketing data set.* 2014. 

## Models

Twelve ONNX classifiers in `models/`, grouped by the dataset profile whose
`model_list` (in [`adapter.py`](adapter.py)) selects them:

| Model        | Profile          | Architecture        | Notes |
|--------------|------------------|---------------------|-------|
| AC-1         | `adult`          | 13 → 16 → 8 → 2     | "Small": two narrow hidden layers |
| AC-3         | `adult`          | 13 → 50 → 2         | "Medium": one wider hidden layer  |
| AC-4         | `adult`          | 13 → 100 → 100 → 2  | "Large": two wide hidden layers   |
| ACD-1/3/4    | `adult_debiased` | same as AC-1/3/4    | Debiased (fairness-trained) Adult nets; same data + declaration as `adult` |
| GC-1         | `german`         | 20 → 50 → 2         | |
| GC-2         | `german`         | 20 → 100 → 2        | |
| GC-3         | `german`         | 20 → 9 → 2          | |
| BM-5         | `bank`           | 16 → 22 → 10 → 2    | |
| BM-6         | `bank`           | 16 → 9 → 9 → 2      | |
| BM-7         | `bank`           | 16 → 64 → 64 → 2    | |

Each model ends in a softmax; the runner strips it before reachability and
verifies on the logits (softmax is order-preserving, so the predicted class is
unchanged and the output specification stays linear).

## Data

Each `data/*.npz` is a lossless NumPy conversion of the corresponding `.mat`
from the NNV source examples — it loads with `np.load` alone (no scipy at run
time) and contents are unchanged. `X` is samples × features and `y` is one-hot
labels (column 0 is the class used by the verification pipeline):

| File              | Source `.mat`                                       | `X`          | `y`         |
|-------------------|-----------------------------------------------------|--------------|-------------|
| `adult_data.npz`  | `…/examples/NNV3.0/FairNNV/data/adult_data.mat`     | `(9769, 13)` | `(9769, 2)` |
| `german_data.npz` | `…/examples/Submission/ICAIF24/data/german_data.mat`| `(150, 20)`  | `(150, 2)`  |
| `bank_data.npz`   | `…/examples/Submission/ICAIF24/data/bank_data.mat`  | `(6098, 16)` | `(6098, 2)` |

The `adult_debiased` profile reuses `adult_data.npz` (same data; only the
verified models differ).

## Layout

```
examples/FairN2V/
├── README.md
├── run_fairn2v.py      Top-level runner; sets config and chains the steps
├── verify.py           Loads ONNX, runs reachability + verification, writes CSVs
├── plot_results.py     Reads the latest CSVs, generates figures + LaTeX tables
├── adapter.py          DatasetAdapter + per-dataset loaders (LOADERS / RUN_PROFILES)
├── models/             AC-*, ACD-*, GC-*, BM-*.onnx
├── data/               adult_data.npz, german_data.npz, bank_data.npz
└── results/            Timestamped output (<yymmdd-HHMMSS>/)
```

`verify.py` and `plot_results.py` can also run standalone — they
fall back to default paths in this folder when `config` is not already
in scope.

## Running

Requires the `n2v` package importable (from the repo root: `pip install -e .`)
and Python 3.9+; dependencies are in [`requirements.txt`](../../requirements.txt).
The runner and the two step scripts resolve `models/`, `data/`, and `results/`
relative to this folder, so no paths need configuring.

### Default sweep

```bash
cd examples/FairN2V
python run_fairn2v.py                  # Adult (default)
python run_fairn2v.py --dataset german
python run_fairn2v.py --dataset bank
python run_fairn2v.py --dataset adult_debiased
```

`adult_debiased` reuses the Adult data and fairness declaration but verifies the
paper's debiased (fairness-trained) networks `ACD-1, ACD-3, ACD-4` — the
debiased half of its biased-vs-debiased comparison.

`--dataset` selects a profile from `RUN_PROFILES` in
[`adapter.py`](adapter.py) (the data file and the models to verify); `adult`
is the default, so existing invocations are unchanged. The Adult run verifies
AC-1, AC-3, and AC-4 on 100 observations, counterfactual fairness (ε = 0) plus
individual fairness across the paper's ε grid, then writes the CSVs, figure,
and LaTeX tables to `results/<timestamp>/`. `--num-obs N` is auto-capped to the
dataset size (German has only 150 samples), and `--models GC-1 GC-2` overrides
the profile's model list.

### Smoke / custom run

There is no separate smoke flag. Either edit the `CONFIGURATION` block at the
top of [`run_fairn2v.py`](run_fairn2v.py) (e.g. `model_list=['AC-1']`,
`num_obs=10`, `epsilon_individual=[0.01]`, `timeout=120`) and run it the same
way, or call the step scripts' `main(config)` directly for a one-off:

```python
from pathlib import Path
import verify, plot_results

config = {
    'models_dir': Path('models'), 'data_dir': Path('data'),
    'output_dir': Path('results/smoke'), 'data_file': 'adult_data.npz',
    'model_list': ['AC-1'], 'num_obs': 10, 'random_seed': 500, 'timeout': 120,
    'epsilon_counterfactual': [0.0], 'epsilon_individual': [0.01],
    'save_png': True, 'save_pdf': True,
}

verify.main(config)
plot_results.main(config)
```

## Configuration parameters

Edit the `CONFIGURATION` block at the top of [`run_fairn2v.py`](run_fairn2v.py),
or pass a pre-populated `config` dict to the step scripts' `main(config)`
(the runner uses `setdefault`, so any caller-supplied values are preserved):

| Key                       | Default                          | Effect |
|---------------------------|----------------------------------|--------|
| `dataset`                 | `'adult'`                        | Dataset profile (`adult`, `adult_debiased`, `german`, `bank`); see `RUN_PROFILES` in `adapter.py` |
| `model_list`              | profile default (`AC-1, AC-3, AC-4` for adult) | Which models to verify (filenames without `.onnx`) |
| `num_obs`                 | `100` (capped to dataset size)   | Number of test observations |

| `random_seed`             | `500`                            | RNG seed (NumPy `default_rng`) |
| `timeout`                 | `600`                            | Per-epsilon timeout (s) |
| `epsilon_counterfactual`  | `[0.0]`                          | ε grid for counterfactual |
| `epsilon_individual`      | `[0.01,0.02,0.03,0.05,0.07,0.1]` | ε grid for individual |
| `save_png` / `save_pdf`   | `True`                           | Figure formats to write |

## Outputs

A timestamped subfolder `results/<yymmdd-HHMMSS>/` is created per run
and contains:

- `counterfactual_<ts>.csv` — per-model fair / unfair %
- `individual_<ts>.csv`     — per-model × ε fair / unfair / unknown %
- `timing_<ts>.csv`         — per-model × ε total + per-sample time
- `counterfactual_table.tex` — booktabs-style LaTeX table
- `individual_fairness_combined.png` / `.pdf` — area plot across models
- `timing_table.tex`         — LaTeX timing table

## Expected runtime

Measured on a MacBook Pro, CPU only (n2v runs on CPU here). Verification wall
time for a full sweep (3 models, ε ∈ {0, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1},
100 obs; plotting adds ~1 s):

| `--dataset`      | sweep time | dominated by (model @ ε=0.1)   |
|------------------|-----------:|--------------------------------|
| `adult`          |    ~60 s   | AC-4 (~0.31 s/sample)          |
| `adult_debiased` |    ~65 s   | ACD-4 (~0.33 s/sample)         |
| `bank`           |    ~72 s   | BM-7 (~0.48 s/sample)          |
| `german`         |    ~87 s   | GC-2 (~0.56 s/sample)          |

A **smoke** run (one small net, e.g. `--models AC-1 --num-obs 10`, ε ∈ {0.01})
is **~2–3 s**. Per-sample cost grows steeply with ε — larger input boxes mean
more ReLU case-splitting in exact reachability — and with width/depth, so the
largest net at ε = 0.1 dominates each sweep; ε = 0 (a single point) is
near-instant. German is slowest overall despite only 150 samples because GC-2
is wide (20 → 100 → 2) and splits heavily at high ε.
