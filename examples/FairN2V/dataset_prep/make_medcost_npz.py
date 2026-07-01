"""Turn data file into npz."""
import kagglehub
import pandas as pd
import numpy as np
import os
from pathlib import Path


FEATURE_ORDER = ["age", "bmi", "children", "sex", "smoker",
                 "region_northeast", "region_northwest",
                 "region_southeast", "region_southwest"]

def process_data(df):
    """Encode dataframe."""
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    df = pd.read_csv(os.path.join(path, "insurance.csv"))
    
    encoded = df.copy()
    encoded["sex"] = (encoded["sex"] == "male").astype(int)  # 1 = male, 0 = female
    encoded["smoker"] = (encoded["smoker"] == "yes").astype(int)  # 1 = smoker, 0 = non-smoker
    encoded = pd.get_dummies(encoded, columns=["region"], prefix="region")

    bool_cols = encoded.select_dtypes(include="bool").columns
    encoded[bool_cols] = encoded[bool_cols].astype(int)
    return encoded


def df_to_numpy(df):
    """Convert dataframe to X and y NumPy arrays."""
    X = df[FEATURE_ORDER].to_numpy()
    y = df[["charges"]].to_numpy()
    return X, y


def main():
    """
    Build data/medcost_data.npz.
    """
    path = kagglehub.dataset_download("mirichoi0218/insurance")
    df = pd.read_csv(os.path.join(path, "insurance.csv"))
    encoded_df = process_data(df)
    X, y = df_to_numpy(encoded_df)
    
    fairn2v_root = Path(__file__).resolve().parent.parent
    out_path = fairn2v_root / "data" / "medcost_data.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Wrote {out_path}")
    print(f"X {X.shape} {X.dtype}\ty {y.shape} {y.dtype}")
    print("Done!")

if __name__ == "__main__":
    main()

