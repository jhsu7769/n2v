"""
Build data/folktables_data.npz for the FairN2V folktables profiles.

Downloads the folktables ACSIncome task for one state-year (California, 2018
1-Year person survey) and writes it in the same `X`/`y` layout the other
FairN2V datasets use:

    X : (n_samples, n_features) float64   -- RAW feature values (the adapter
                                              min-max normalizes them itself)
    y : (n_samples, 2)         float64    -- one-hot label; column 0 is the
                                              class index the pipeline verifies

ACSIncome predicts whether personal income exceeds $50k. The raw feature order
is ['AGEP','COW','SCHL','MAR','OCCP','POBP','RELP','WKHP','SEX','RAC1P']. We
keep the 9 non-race features as-is and replace the lone ordinal race column
RAC1P with a 4-way ONE-HOT block over OMB-style groups [White, Black, Asian,
Other], giving 13 features:

    [AGEP, COW, SCHL, MAR, OCCP, POBP, RELP, WKHP, SEX,
     RACE_White, RACE_Black, RACE_Asian, RACE_Other]
       (col 8 = SEX, binary)        (cols 9..12 = race one-hot)

This ONE dataset serves BOTH fairness profiles (same noun, different verb):
  * `folktables`       verifies SEX (binary,  sensitive col 8)
  * `folktables_race`  verifies RACE (one-hot, sensitive cols 9..12)
Race is one-hot (not the ordinal RAC1P code) because race is unordered; the 6
small ACS race codes fold into "Other" so every one-hot column is well populated.

Unlike adult/german/bank (whose .mat test sets shipped with NNV), there is no
external baseline here: this script *creates* the dataset and train_folktables.py
trains the nets on it. We subsample to a fixed size with a fixed seed so the
artifact is reproducible and repo-friendly; verification draws its 100 samples
from this same set (matching how the other profiles verify the data they hold).

USAGE (from examples/FairN2V/):
    python dataset_prep/make_folktables_npz.py   # CA, 2018, 1-Year, 20000 rows
"""

from pathlib import Path

import numpy as np
from folktables import ACSDataSource, ACSIncome

STATE = 'CA'
SURVEY_YEAR = '2018'
HORIZON = '1-Year'
SUBSAMPLE = 20000          # rows kept (repo-friendly; CA 1-Yr is ~195k)
SUBSAMPLE_SEED = 0

# RAC1P raw code -> OMB-style group index for the one-hot race block.
# White=1, Black=2, Asian=6 each get their own column; the 6 small codes
# (AmerIndian/AlaskaNative/AIAN-spec/NHPI/OtherRace/TwoOrMore) fold into Other.
RACE_GROUPS = ['White', 'Black', 'Asian', 'Other']
RACE_CODE_TO_GROUP = {1: 0, 2: 1, 6: 2}  # everything else -> 3 (Other)
RAC1P_COL = ACSIncome.features.index('RAC1P')

# This script lives in examples/FairN2V/dataset_prep/; anchor on the FairN2V
# root so data/ and the download cache resolve regardless of cwd.
fairn2v_root = Path(__file__).resolve().parent.parent
out_path = fairn2v_root / 'data' / 'folktables_data.npz'
cache_dir = fairn2v_root / '_folktables_cache'  # raw download, kept out of data/


def onehot_race(X):
    """Replace the ordinal RAC1P column with a 4-way OMB one-hot block.

    Returns (X_new, feature_names): the 9 non-race features in their original
    order, then the 4 race one-hot columns appended at the end (cols 9..12).
    """
    groups = np.array([RACE_CODE_TO_GROUP.get(int(c), 3) for c in X[:, RAC1P_COL]])
    onehot = np.zeros((X.shape[0], len(RACE_GROUPS)), dtype=np.float64)
    onehot[np.arange(X.shape[0]), groups] = 1.0

    keep = [j for j in range(X.shape[1]) if j != RAC1P_COL]  # drop RAC1P column
    X_new = np.hstack([X[:, keep], onehot])
    names = [ACSIncome.features[j] for j in keep] + [f'RACE_{g}' for g in RACE_GROUPS]
    return X_new, names


def main():
    cache_dir.mkdir(exist_ok=True)
    source = ACSDataSource(survey_year=SURVEY_YEAR, horizon=HORIZON,
                           survey='person', root_dir=str(cache_dir))
    data = source.get_data(states=[STATE], download=True)

    # df_to_numpy applies ACSIncome's filters + returns features / bool label.
    X_all, y_all, _group = ACSIncome.df_to_numpy(data)
    X_all = X_all.astype(np.float64)
    y_bool = y_all.astype(bool)  # True == income > $50k

    # Fixed-seed subsample for a reproducible, repo-friendly artifact.
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    n = min(SUBSAMPLE, X_all.shape[0])
    idx = rng.choice(X_all.shape[0], size=n, replace=False)
    idx.sort()
    X, y_bool = X_all[idx], y_bool[idx]

    # One-hot the race column -> 13-feature representation shared by both profiles.
    X, feats = onehot_race(X)

    # One-hot label, column 0 = class index (1 if >50k else 0) the pipeline reads.
    pos = y_bool.astype(np.float64)            # 1.0 for >50k
    y = np.column_stack([pos, 1.0 - pos])      # (n, 2): col0 = class, col1 = 1-col0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)

    print(f"Wrote {out_path}")
    print(f"  X {X.shape} {X.dtype}   y {y.shape} {y.dtype}")
    print(f"  features: {feats}")
    print(f"  SEX (sensitive col 8) raw uniques: {np.unique(X[:, 8])}")
    block = X[:, 9:13]
    print(f"  race one-hot cols 9..12 = {RACE_GROUPS}")
    print(f"  per-group counts: {dict(zip(RACE_GROUPS, block.sum(0).astype(int)))}")
    print(f"  all rows exactly one race? {bool((block.sum(1) == 1).all())}")
    print(f"  class balance (col0): >50k={int(pos.sum())} "
          f"<=50k={int((1 - pos).sum())} ({100 * pos.mean():.1f}% positive)")
    rng_per_feat = X.max(axis=0) - X.min(axis=0)
    const = [feats[i] for i in range(len(feats)) if rng_per_feat[i] == 0]
    print(f"  constant features: {const if const else 'none'}")


if __name__ == '__main__':
    main()
