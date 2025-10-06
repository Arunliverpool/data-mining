#!/usr/bin/env python3
"""
Part 1 — Preprocessing Selection (P*) via Cross-Validation

What it does
------------
- Loads train.csv (assumes last column is the target).
- Infers numeric vs categorical columns (OHE for categoricals).
- Builds candidate preprocessors:
    * Scaling: MinMax vs Standard
    * Outlier feature: none vs IsolationForest score vs LOF score
  (Imputation = overall mean for numeric, mode for categoricals — class-wise impute is excluded here to avoid leakage.)
- Evaluates each preprocessor with a baseline RandomForest using StratifiedKFold CV (F1).
- Saves:
    * CSV of CV results
    * JSON of best config
    * Fitted preprocessor (joblib) to reuse in later parts

Usage
-----
pip install pandas numpy scikit-learn joblib
python part1_preprocess_selection.py --train /path/to/train.csv --seed 42

Outputs
-------
- preprocess_cv_results.csv
- best_preprocess_config.json
- best_preprocess.joblib
"""

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


# --------------------------- Utils ---------------------------

def set_global_seed(seed: int):
    np.random.seed(seed)


def infer_columns(df: pd.DataFrame, target_col: str = None) -> Tuple[List[str], List[str], str]:
    """Infer numeric and categorical columns; assume last column is target if not given."""
    if target_col is None:
        target_col = df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col]
    num_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in feature_cols if c not in num_cols]
    return num_cols, cat_cols, target_col


# ---------------- Outlier score feature transformers ----------------

class OutlierScoreAdder(BaseEstimator, TransformerMixin):
    """
    Adds a single outlier score column computed from numeric columns.
    method: 'none' | 'iforest' | 'lof'
    Internally imputes (mean) and scales (StandardScaler) before scoring to
    avoid NaNs and make distance-based methods stable.
    """
    def __init__(self, num_cols: List[str], method: str = "none", random_state: int = 42, n_neighbors: int = 20):
        self.num_cols = num_cols
        self.method = method
        self.random_state = random_state
        self.n_neighbors = n_neighbors

    def fit(self, X, y=None):
        X_df = self._ensure_df(X)
        X_num = X_df[self.num_cols].astype(float)

        # Impute + scale so LOF/IF see finite, comparable features
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler

        self.imputer_ = SimpleImputer(strategy="mean")
        X_imp = self.imputer_.fit_transform(X_num)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        if self.method == "iforest":
            self.model_ = IsolationForest(
                random_state=self.random_state,
                n_estimators=200,
                contamination="auto",
            )
            self.model_.fit(X_scaled)
        elif self.method == "lof":
            # keep n_neighbors <= n_samples-1
            n_fit = min(self.n_neighbors, max(5, X_scaled.shape[0] - 1))
            self.model_ = LocalOutlierFactor(
                n_neighbors=n_fit,
                novelty=True
            )
            self.model_.fit(X_scaled)
        else:
            self.model_ = None
        return self

    def transform(self, X):
        X_df = self._ensure_df(X)
        X_num = X_df[self.num_cols].astype(float)

        # Apply same impute + scale
        X_imp = self.imputer_.transform(X_num)
        X_scaled = self.scaler_.transform(X_imp)

        if self.method in {"iforest", "lof"}:
            scores = -self.model_.score_samples(X_scaled)  # invert: higher => more outlier
        else:
            scores = np.zeros(X_scaled.shape[0], dtype=float)
        return scores.reshape(-1, 1)

    @staticmethod
    def _ensure_df(X):
        if isinstance(X, pd.DataFrame):
            return X
        raise TypeError("OutlierScoreAdder expects a pandas DataFrame as input.")


# --------------------------- Config ---------------------------

@dataclass
class PreprocessConfig:
    scale: str                 # 'minmax' | 'standard'
    outlier: str               # 'none' | 'iforest' | 'lof'
    num_impute: str = "mean"   # fixed here to 'mean' per spec (overall average)
    cat_impute: str = "most_frequent"

    def key(self) -> str:
        return f"scale={self.scale}|outlier={self.outlier}"


def make_preprocessor(cfg: PreprocessConfig, num_cols: List[str], cat_cols: List[str], seed: int) -> ColumnTransformer:
    """Build a ColumnTransformer per config."""
    # Imputers
    num_imputer = SimpleImputer(strategy=cfg.num_impute)
    cat_imputer = SimpleImputer(strategy=cfg.cat_impute)

    # Scalers
    if cfg.scale == "minmax":
        scaler = MinMaxScaler()
    elif cfg.scale == "standard":
        scaler = StandardScaler(with_mean=True, with_std=True)
    else:
        raise ValueError(f"Unknown scaler: {cfg.scale}")

    # Pipelines for subsets
    num_pipe = Pipeline([
        ("imputer", num_imputer),
        ("scaler", scaler),
    ])

    cat_pipe = Pipeline([
        ("imputer", cat_imputer),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = [
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ]

    # Optional outlier score feature
    if cfg.outlier in {"iforest", "lof"}:
        transformers.append(
            ("outlier", OutlierScoreAdder(num_cols=num_cols, method=cfg.outlier, random_state=seed), num_cols)
        )
    # else: omit (no extra column)

    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def grid_of_configs() -> List[PreprocessConfig]:
    scales = ["minmax", "standard"]
    outliers = ["none", "iforest", "lof"]
    return [PreprocessConfig(scale=s, outlier=o) for s in scales for o in outliers]


# --------------------------- CV Harness ---------------------------

def evaluate_configs(
    X: pd.DataFrame,
    y: pd.Series,
    num_cols: List[str],
    cat_cols: List[str],
    seed: int,
    n_splits: int = 5
) -> pd.DataFrame:
    results = []
    scorer = make_scorer(f1_score, pos_label=1)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    for cfg in grid_of_configs():
        pre = make_preprocessor(cfg, num_cols, cat_cols, seed)
        # Baseline RF; stable, handles OHE+numerics together
        clf = RandomForestClassifier(
            n_estimators=300,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced_subsample",
            max_depth=None,
        )
        pipe = Pipeline([
            ("preprocess", pre),
            ("clf", clf),
        ])

        scores = cross_val_score(pipe, X, y, cv=cv, scoring=scorer, n_jobs=-1)
        results.append({
            "config": cfg.key(),
            "scale": cfg.scale,
            "outlier": cfg.outlier,
            "num_impute": cfg.num_impute,
            "cat_impute": cfg.cat_impute,
            "f1_mean": float(np.mean(scores)),
            "f1_std": float(np.std(scores, ddof=1)),
            "scores": list(map(float, scores)),
        })

    return pd.DataFrame(results).sort_values(["f1_mean", "f1_std"], ascending=[False, True]).reset_index(drop=True)


# --------------------------- Main ---------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--target", type=str, default=None, help="Target column name (defaults to last column)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_global_seed(args.seed)

    # Load data
    df = pd.read_csv(args.train)
    num_cols, cat_cols, target_col = infer_columns(df, args.target)
    X = df[num_cols + cat_cols].copy()
    y = df[target_col].astype(int)  # assumes binary 0/1; adjust if needed

    print(f"[INFO] Target: {target_col}")
    print(f"[INFO] Numeric cols ({len(num_cols)}): {num_cols[:5]}{'...' if len(num_cols)>5 else ''}")
    print(f"[INFO] Categorical cols ({len(cat_cols)}): {cat_cols[:5]}{'...' if len(cat_cols)>5 else ''}")

    # Evaluate
    cv_df = evaluate_configs(X, y, num_cols, cat_cols, seed=args.seed)
    cv_df.to_csv("preprocess_cv_results.csv", index=False)
    print("\n=== Top Preprocessors (by CV F1) ===")
    print(cv_df.head(10)[["config", "f1_mean", "f1_std"]])

    # Best config
    best_row = cv_df.iloc[0].to_dict()
    best_cfg = PreprocessConfig(scale=best_row["scale"], outlier=best_row["outlier"])
    with open("best_preprocess_config.json", "w") as f:
        json.dump(asdict(best_cfg) | {"f1_mean": best_row["f1_mean"], "f1_std": best_row["f1_std"]}, f, indent=2)

    # Fit best preprocessor on full training data and persist
    best_pre = make_preprocessor(best_cfg, num_cols, cat_cols, args.seed)
    # We fit it inside a pipeline so OHE sees all categories in train
    pre_pipe = Pipeline([("preprocess", best_pre)])
    pre_pipe.fit(X, y)
    dump(pre_pipe, "best_preprocess.joblib")

    print("\n[OK] Saved:")
    print(" - preprocess_cv_results.csv")
    print(" - best_preprocess_config.json")
    print(" - best_preprocess.joblib")


if __name__ == "__main__":
    main()
