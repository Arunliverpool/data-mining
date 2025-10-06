#!/usr/bin/env python3
# Part 2 — Preprocessing comparison via CV (no model sweep yet)
# Usage:
#   python part2_preprocessing_cv.py --train data/train.csv --target Target \
#       --out reports/cv_preprocessing_results.csv --seed 42 --folds 5

import argparse
import json
import os
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer, make_column_selector as Select
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

# imbalanced-learn pipeline lets us drop rows in a pipeline step (fit_resample)
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)

# ---------- Helpers ----------
def make_ohe():
    """Return a dense OneHotEncoder compatible with sklearn >=1.2 and older."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        # for older sklearn where 'sparse' is the arg
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def to_readable(obj):
    """Nicely stringify parameter objects in cv_results_."""
    if obj is None or obj == "passthrough":
        return "none"
    name = type(obj).__name__ if not isinstance(obj, str) else obj
    if hasattr(obj, "get_params"):
        return name
    return str(obj)

# ---------- Outlier droppers (train-fold only) ----------
class IsolationForestDropper(BaseEstimator):
    """Drops outliers flagged by IsolationForest during fit; ignored at predict time."""
    def __init__(self, contamination=0.01, random_state=42):
        self.contamination = contamination
        self.random_state = random_state

    def fit_resample(self, X, y):
        # Use only numeric columns to fit the detector; fill NaNs temporarily.
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                # No numeric columns; nothing to drop
                return X, y
            X_fit = X_num.copy()
            X_fit = X_fit.fillna(X_fit.median(numeric_only=True))
        else:
            X_fit = np.nan_to_num(X, nan=0.0)

        iso = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state
        )
        mask = iso.fit_predict(X_fit) == 1  # 1 = inlier, -1 = outlier

        if hasattr(X, "iloc"):
            X_res = X.iloc[mask]
        else:
            X_res = X[mask]
        y_res = y[mask]
        return X_res, y_res

class LOFDropper(BaseEstimator):
    """Drops outliers flagged by LocalOutlierFactor during fit; ignored at predict time."""
    def __init__(self, contamination=0.01, n_neighbors=20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                return X, y
            X_fit = X_num.copy()
            X_fit = X_fit.fillna(X_fit.median(numeric_only=True))
        else:
            X_fit = np.nan_to_num(X, nan=0.0)

        lof = LocalOutlierFactor(
            contamination=self.contamination,
            n_neighbors=self.n_neighbors
        )
        labels = lof.fit_predict(X_fit)         # 1=inlier, -1=outlier
        mask = labels == 1

        if hasattr(X, "iloc"):
            X_res = X.iloc[mask]
        else:
            X_res = X[mask]
        y_res = y[mask]
        return X_res, y_res

# ---------- Main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (with labels).")
    ap.add_argument("--target", default="Target (Col44)", help="Name of target column")
    ap.add_argument("--out", default="reports/cv_preprocessing_results.csv", help="CSV output path.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--contamination", type=float, default=0.01, help="Outlier fraction for iForest/LOF.")
    args = ap.parse_args()

    # 1) Load data
    df = pd.read_csv(args.train)
    if args.target not in df.columns:
        # fallback: use the last column as target
        args.target = df.columns[-1]
        print(f"[info] --target not found; using last column as target: {args.target}")

    y = df[args.target].astype(int).values
    X = df.drop(columns=[args.target])

    # 2) Column selectors
    num_selector = Select(dtype_include=np.number)
    cat_selector = Select(dtype_exclude=np.number)

    # 3) Sub-pipelines
    num_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),  # toggled
        ("scaler", "passthrough"),                    # toggled
    ])

    cat_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  # toggled
        ("onehot", make_ohe()),                                # fixed dense OHE
    ])

    pre = ColumnTransformer(transformers=[
        ("num", num_pipe, num_selector),
        ("cat", cat_pipe, cat_selector),
    ], remainder="drop")

    # 4) Full pipeline (outlier dropper -> preprocessor -> probe model)
    pipe = ImbPipeline(steps=[
        ("outliers", "passthrough"),  # toggled
        ("pre", pre),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=args.seed, n_jobs=-1))
    ])

    # 5) Parameter grid (preprocessing + outliers only)
    param_grid = {
        # numeric imputation and scaling
        "pre__num__imputer__strategy": ["mean", "median"],
        "pre__num__scaler": ["passthrough", StandardScaler(), MinMaxScaler(), RobustScaler()],
        # categorical imputation (mode vs constant "MISSING")
        "pre__cat__imputer__strategy": ["most_frequent", "constant"],
        "pre__cat__imputer__fill_value": ["MISSING"],
        # outlier handling
        "outliers": [
            "passthrough",
            IsolationForestDropper(contamination=args.contamination, random_state=args.seed),
            LOFDropper(contamination=args.contamination, n_neighbors=20),
        ],
    }

    # 6) CV + scoring
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="f1",          # positive class = 1 by default for {0,1} labels
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=False,
        verbose=1
    )

    # 7) Fit
    print("[info] starting grid search over preprocessing choices...")
    grid.fit(X, y)
    print("[info] done.")

    # 8) Results table
    results = pd.DataFrame(grid.cv_results_)
    # Keep only the useful columns
    keep = ["mean_test_score", "std_test_score", "params"]
    results = results[keep].copy()

    # Expand params into columns and prettify values
    params_df = results["params"].apply(pd.Series)
    for col in params_df.columns:
        params_df[col] = params_df[col].map(to_readable)

    results = pd.concat([params_df, results.drop(columns=["params"])], axis=1)
    results = results.rename(columns={
        "mean_test_score": "mean_f1",
        "std_test_score": "std_f1",
        "pre__num__imputer__strategy": "num_impute",
        "pre__num__scaler": "scale",
        "pre__cat__imputer__strategy": "cat_impute",
        "pre__cat__imputer__fill_value": "cat_fill",
        "outliers": "outliers"
    })

    results = results.sort_values(by=["mean_f1", "std_f1"], ascending=[False, True]).reset_index(drop=True)

    # Ensure output dir
    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.out, index=False)

    # 9) Report best config
    best_row = results.iloc[0].to_dict()
    best_summary = {
        "num_impute": best_row["num_impute"],
        "scale": best_row["scale"],
        "cat_impute": best_row["cat_impute"],
        "cat_fill": best_row["cat_fill"],
        "outliers": best_row["outliers"],
        "mean_f1": round(float(best_row["mean_f1"]), 4),
        "std_f1": round(float(best_row["std_f1"]), 4),
        "seed": args.seed,
        "folds": args.folds
    }
    best_json_path = os.path.join(out_dir, "part2_best_preprocessing.json")
    with open(best_json_path, "w") as f:
        json.dump(best_summary, f, indent=2)

    print("\n=== Part 2 — Best preprocessing (by CV F1) ===")
    print(json.dumps(best_summary, indent=2))
    print(f"\n[saved] full table -> {args.out}")
    print(f"[saved] best summary -> {best_json_path}")
    print("\nNext step: carry these settings (P★) into Part 3 model sweep & tuning.")
    

if __name__ == "__main__":
    main()
