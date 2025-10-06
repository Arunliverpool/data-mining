#!/usr/bin/env python3
"""
Part 3 — Final Train & Prediction

What this does
--------------
- Loads `best_model_config.json` from Part 2 (contains P*, M*, H*, CV metrics).
- Rebuilds the exact preprocessing pipeline (P*) and final model (M*, H*).
- Fits on the full train set.
- Applies the same P* to the unlabeled test set and generates predictions.
- Writes the required result file: <SID>.infs4203
  * Rows 1..2713: predicted labels (0/1, one per line)
  * Row 2714: "<accuracy>,<f1>" (from Part 2 CV), both rounded to 3 d.p.

Usage
-----
python part3_final_train_predict.py --train train.csv --test test_data.csv --sid sXXXXXXX --seed 42

Outputs
-------
- <SID>.infs4203
- final_pipeline.joblib (fitted preprocessor+model, for your records)
"""

import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB


# -------- utils --------
def infer_columns(df: pd.DataFrame, target_col: str | None) -> Tuple[List[str], List[str], str]:
    if target_col is None:
        target_col = df.columns[-1]
    features = [c for c in df.columns if c != target_col]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols, target_col


# -------- Outlier score adder (same patched version used before) --------
class OutlierScoreAdder(BaseEstimator, TransformerMixin):
    """
    Adds one outlier score column from numeric columns.
    method: 'none' | 'iforest' | 'lof'
    Internally imputes (mean) + StandardScaler before scoring.
    """
    def __init__(self, num_cols: List[str], method: str = "none", random_state: int = 42, n_neighbors: int = 20):
        self.num_cols = num_cols
        self.method = method
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X_df = self._ensure_df(X)
        X_num = X_df[self.num_cols].astype(float)

        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp = self.imputer_.fit_transform(X_num)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        if self.method == "iforest":
            from sklearn.ensemble import IsolationForest
            self.model_ = IsolationForest(
                random_state=self.random_state,
                n_estimators=200,
                contamination="auto",
            ).fit(X_scaled)
        elif self.method == "lof":
            n_fit = min(self.n_neighbors, max(5, X_scaled.shape[0] - 1))
            self.model_ = LocalOutlierFactor(
                n_neighbors=n_fit,
                novelty=True
            ).fit(X_scaled)
        else:
            self.model_ = None
        return self

    def transform(self, X):
        X_df = self._ensure_df(X)
        X_num = X_df[self.num_cols].astype(float)
        X_imp = self.imputer_.transform(X_num)
        X_scaled = self.scaler_.transform(X_imp)

        if self.method in {"iforest", "lof"}:
            scores = -self.model_.score_samples(X_scaled)
        else:
            scores = np.zeros(X_scaled.shape[0], dtype=float)
        return scores.reshape(-1, 1)

    @staticmethod
    def _ensure_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("OutlierScoreAdder expects a pandas DataFrame as input.")


# -------- Config dataclass --------
@dataclass
class PreprocessConfig:
    scale: str                 # 'minmax' | 'standard'
    outlier: str               # 'none' | 'iforest' | 'lof'
    num_impute: str = "mean"
    cat_impute: str = "most_frequent"


def build_preprocessor(cfg: PreprocessConfig, num_cols: List[str], cat_cols: List[str], seed: int):
    if cfg.scale == "minmax":
        scaler = MinMaxScaler()
    elif cfg.scale == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scale: {cfg.scale}")

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.num_impute)),
        ("scaler", scaler),
    ])
    cat_imputer = (
        SimpleImputer(strategy="most_frequent")
        if cfg.cat_impute != "constant"
        else SimpleImputer(strategy="constant", fill_value="MISSING")
    )
    cat_pipe = Pipeline([
        ("imputer", cat_imputer),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]
    if cfg.outlier in {"iforest", "lof"}:
        transformers.append(
            ("outlier", OutlierScoreAdder(num_cols=num_cols, method=cfg.outlier, random_state=seed), num_cols)
        )

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_model(model_name: str) -> BaseEstimator:
    if model_name == "RandomForest":
        return RandomForestClassifier(random_state=0, n_jobs=-1)
    if model_name == "DecisionTree":
        return DecisionTreeClassifier(random_state=0)
    if model_name == "kNN":
        return KNeighborsClassifier()
    if model_name == "GaussianNB":
        return GaussianNB()
    raise ValueError(f"Unsupported model_name: {model_name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=str)
    ap.add_argument("--test", required=True, type=str)
    ap.add_argument("--target", default=None, type=str)
    ap.add_argument("--sid", required=True, type=str, help="Student ID for output file name, e.g., s1234567")
    ap.add_argument("--seed", default=42, type=int)
    args = ap.parse_args()

    # Load best config from Part 2
    with open("best_model_config.json", "r") as f:
        best = json.load(f)

    model_name: str = best["model_name"]
    best_params: dict = best["best_params"]
    cv_mean_f1: float = float(best["cv_mean_f1"])
    cv_mean_acc: float = float(best["cv_mean_accuracy"])
    pre_cfg_json = best["preprocess"]
    cfg = PreprocessConfig(**pre_cfg_json)

    print(f"[INFO] Using P*: {asdict(cfg)}")
    print(f"[INFO] Using M*: {model_name} with params {best_params}")

    # Load data
    df_train = pd.read_csv(args.train)
    num_cols, cat_cols, target_col = infer_columns(df_train, args.target)
    X_train = df_train[num_cols + cat_cols].copy()
    y_train = df_train[target_col].astype(int)

    df_test = pd.read_csv(args.test)
    # Ensure all expected feature columns exist in test (missing columns will be imputed / OHE handles unknowns)
    X_test = df_test[num_cols + cat_cols].copy()

    # Build preprocessor & model
    pre = build_preprocessor(cfg, num_cols, cat_cols, args.seed)
    clf = build_model(model_name)

    # Assemble pipeline and set best params (they use 'clf__...' keys)
    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf),
    ])
    pipe.set_params(**best_params)

    # Fit on full training data
    pipe.fit(X_train, y_train)

    # Predict on test
    test_pred = pipe.predict(X_test)
    test_pred = pd.Series(test_pred).astype(int).tolist()

    # Save final pipeline (optional but handy)
    dump(pipe, "final_pipeline.joblib")

    # Build output file
    out_path = f"{args.sid}.infs4203"

    # Sanity checks
    n_test = len(test_pred)
    print(f"[INFO] Test predictions: {n_test}")
    if n_test != 2713:
        print(f"[WARN] Expected 2713 predictions, got {n_test}. The file will still be written.")

    # Format last line with the CV metrics from Part 2 (rounded to 3 d.p.)
    acc_str = f"{cv_mean_acc:.3f}"
    f1_str  = f"{cv_mean_f1:.3f}"

    with open(out_path, "w", encoding="utf-8") as f:
        for p in test_pred:
            f.write(f"{int(p)}\n")
        f.write(f"{acc_str},{f1_str}\n")

    print("\n[OK] Wrote:", out_path)
    print(f"Last row (CV metrics): {acc_str},{f1_str}")


if __name__ == "__main__":
    main()
