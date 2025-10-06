#!/usr/bin/env python3
"""
Part 2 — Model & Hyperparameter Search (M*, H*) via Cross-Validation

Usage:
  pip install pandas numpy scikit-learn joblib
  python part2_model_search.py --train train.csv --seed 42 --cv 5

Outputs:
  - model_cv_results.csv          # all CV trials across all models
  - best_model_config.json        # selected M*, H* with mean F1/Acc, std
"""

import argparse
import json
from dataclasses import asdict, dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB


# ------------- utils -------------
def infer_columns(df: pd.DataFrame, target_col: str | None) -> Tuple[List[str], List[str], str]:
    if target_col is None:
        target_col = df.columns[-1]
    features = [c for c in df.columns if c != target_col]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols, target_col


# ------------- Outlier score adder (same as Part 1 patched) -------------
class OutlierScoreAdder(BaseEstimator, TransformerMixin):
    """
    Adds a single outlier score column computed from numeric columns.
    method: 'none' | 'iforest' | 'lof'
    Internally imputes (mean) and scales (StandardScaler) before scoring.
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


# ------------- Preprocess config (read from Part 1 JSON) -------------
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
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.cat_impute)),
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


# ------------- Model grids -------------
def model_grids(quick: bool = False) -> Dict[str, Tuple[BaseEstimator, Dict[str, list]]]:
    grids = {}

    # Decision Tree
    dt = DecisionTreeClassifier(random_state=0)
    dt_grid = {
        "clf__criterion": ["gini", "entropy"],
        "clf__max_depth": [None, 5, 10, 20] if not quick else [None, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__ccp_alpha": [0.0, 1e-4, 1e-3] if not quick else [0.0],
    }
    grids["DecisionTree"] = (dt, dt_grid)

    # Random Forest
    rf = RandomForestClassifier(random_state=0, n_jobs=-1)
    rf_grid = {
        "clf__n_estimators": [200, 400] if not quick else [300],
        "clf__max_depth": [None, 10, 20] if not quick else [None, 20],
        "clf__max_features": ["sqrt", "log2"],
        "clf__class_weight": [None, "balanced_subsample"],
    }
    grids["RandomForest"] = (rf, rf_grid)

    # k-NN (needs scaling — already in P*)
    knn = KNeighborsClassifier()
    knn_grid = {
        "clf__n_neighbors": [3, 5, 7, 11, 15] if not quick else [5, 11],
        "clf__metric": ["euclidean", "manhattan", "chebyshev"],
        "clf__weights": ["uniform", "distance"],
    }
    grids["kNN"] = (knn, knn_grid)

    # Gaussian Naive Bayes
    gnb = GaussianNB()
    gnb_grid = {
        "clf__var_smoothing": [1e-9, 1e-8, 1e-7] if not quick else [1e-9, 1e-8],
    }
    grids["GaussianNB"] = (gnb, gnb_grid)

    return grids


# ------------- Main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, type=str)
    ap.add_argument("--target", default=None, type=str)
    ap.add_argument("--seed", default=42, type=int)
    ap.add_argument("--cv", default=5, type=int)
    ap.add_argument("--quick", action="store_true", help="Smaller grids for faster run")
    args = ap.parse_args()

    # Load data
    df = pd.read_csv(args.train)
    num_cols, cat_cols, target_col = infer_columns(df, args.target)
    X = df[num_cols + cat_cols].copy()
    y = df[target_col].astype(int)

    # Load P* config from Part 1
    with open("best_preprocess_config.json", "r") as f:
        cfg_json = json.load(f)
    cfg = PreprocessConfig(
        scale=cfg_json["scale"],
        outlier=cfg_json["outlier"],
        num_impute=cfg_json.get("num_impute", "mean"),
        cat_impute=cfg_json.get("cat_impute", "most_frequent"),
    )

    print(f"[INFO] Using P*: scale={cfg.scale}, outlier={cfg.outlier}")

    # Build preprocessor (UNFITTED) so it’s refit inside each CV fold
    pre = build_preprocessor(cfg, num_cols, cat_cols, args.seed)

    scorer_f1 = make_scorer(f1_score, pos_label=1)
    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

    all_rows = []
    best_overall = None  # (mean_f1, model_name, best_params, std_f1, mean_acc, std_acc, best_estimator)

    for model_name, (estimator, grid) in model_grids(args.quick).items():
        pipe = Pipeline([
            ("pre", pre),
            ("clf", estimator),
        ])

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring={"f1": scorer_f1, "accuracy": "accuracy"},
            refit="f1",
            cv=cv,
            n_jobs=-1,
            error_score="raise"
        )
        gs.fit(X, y)

        # Collect all trials
        res = pd.DataFrame(gs.cv_results_)
        res["model"] = model_name
        # Keep compact columns
        keep_cols = [
            "model", "params", "mean_test_f1", "std_test_f1", "mean_test_accuracy", "std_test_accuracy",
            "rank_test_f1"
        ]
        all_rows.append(res[keep_cols].copy())

        # Track best-overall on F1
        m = float(gs.best_score_)
        s = float(res.loc[res["rank_test_f1"] == 1, "std_test_f1"].iloc[0])
        acc_mean = float(res.loc[res["rank_test_f1"] == 1, "mean_test_accuracy"].iloc[0])
        acc_std = float(res.loc[res["rank_test_f1"] == 1, "std_test_accuracy"].iloc[0])
        candidate = (m, model_name, gs.best_params_, s, acc_mean, acc_std, gs.best_estimator_)

        if (best_overall is None) or (candidate[0] > best_overall[0]):
            best_overall = candidate

        print(f"[{model_name}] best F1={m:.4f} (±{s:.4f}) | Acc={acc_mean:.4f} (±{acc_std:.4f})")
        print(f"  best params: {gs.best_params_}")

    # Save all CV trials
    all_df = pd.concat(all_rows, ignore_index=True)
    all_df = all_df.sort_values(["mean_test_f1", "std_test_f1"], ascending=[False, True]).reset_index(drop=True)
    all_df.to_csv("model_cv_results.csv", index=False)

    # Save best model config
    mean_f1, model_name, best_params, std_f1, acc_mean, acc_std, best_estimator = best_overall
    best = {
        "model_name": model_name,
        "best_params": best_params,
        "cv_mean_f1": round(mean_f1, 6),
        "cv_std_f1": round(std_f1, 6),
        "cv_mean_accuracy": round(acc_mean, 6),
        "cv_std_accuracy": round(acc_std, 6),
        "preprocess": asdict(cfg),
        "cv_folds": args.cv,
        "seed": args.seed,
    }
    with open("best_model_config.json", "w") as f:
        json.dump(best, f, indent=2)

    # (Optional) persist the best estimator fitted on full data (useful for quick checks; we’ll still retrain in Part 3)
    dump(best_estimator, "best_pipeline_cvfit.joblib")

    print("\n=== Best Overall (by F1) ===")
    print(json.dumps(best, indent=2))
    print("\n[OK] Saved:")
    print(" - model_cv_results.csv")
    print(" - best_model_config.json")
    print(" - best_pipeline_cvfit.joblib")


if __name__ == "__main__":
    main()
