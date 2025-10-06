#!/usr/bin/env python3
"""
Part 3 — Model + Hyperparameter search with P★ preprocessing fixed.

Usage example:
  python part3_model_search.py \
    --train data/train.csv \
    --target "Target (Col44)" \
    --out reports/cv_model_results.csv \
    --seed 42 --folds 5 --contamination 0.01 \
    --models dt,rf,knn,gnb,mnb
"""

import argparse, json, os, warnings
import numpy as np, pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import make_scorer, f1_score, accuracy_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB

from sklearn.base import BaseEstimator
from sklearn.neighbors import LocalOutlierFactor
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)


# ---------- helpers ----------
def make_ohe():
    """Dense OneHotEncoder; compatible with sklearn >=1.2 and older."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def to_readable(obj):
    """Pretty param values for CSV/printouts."""
    if obj is None or obj == "passthrough":
        return "none"
    return obj if isinstance(obj, str) else type(obj).__name__


# ---------- LOF dropper (train-fold only) ----------
class LOFDropper(BaseEstimator):
    """Drops outliers flagged by LocalOutlierFactor during fit; ignored at predict time."""
    def __init__(self, contamination=0.01, n_neighbors=20):
        self.contamination = contamination
        self.n_neighbors = n_neighbors

    def fit_resample(self, X, y):
        # Use numeric columns only to fit LOF; fill NaNs temporarily
        if isinstance(X, pd.DataFrame):
            X_num = X.select_dtypes(include=[np.number])
            if X_num.shape[1] == 0:
                return X, y
            X_fit = X_num.copy()
            X_fit = X_fit.fillna(X_fit.median(numeric_only=True))
        else:
            X_fit = np.nan_to_num(X, nan=0.0)

        lof = LocalOutlierFactor(contamination=self.contamination,
                                 n_neighbors=self.n_neighbors)
        mask = (lof.fit_predict(X_fit) == 1)  # 1=inlier, -1=outlier
        X_res = X.iloc[mask] if hasattr(X, "iloc") else X[mask]
        y_res = y[mask]
        return X_res, y_res


# ---------- build P★ preprocessor ----------
def build_preprocessor_pstar(numeric_cols, categorical_cols):
    """
    P★ from Part 2:
      - numeric impute: median
      - scale: MinMaxScaler
      - categorical impute: constant("MISSING")
      - encoding: OneHotEncoder(handle_unknown='ignore', dense)
    """
    num_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ])
    cat_pipe = ImbPipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", make_ohe()),
    ])
    return ColumnTransformer(transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")


def main():
    # ---- args ----
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True, help="Path to train.csv (with labels)")
    ap.add_argument("--target", default="Target (Col44)", help="Target column name")
    ap.add_argument("--out", default="reports/cv_model_results.csv", help="CSV output path")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--contamination", type=float, default=0.01, help="LOF contamination (0..1)")
    ap.add_argument("--models", default="dt,rf,knn,gnb,mnb",
                    help="Comma-separated subset of models to run: dt,rf,knn,gnb,mnb")
    args = ap.parse_args()

    # ---- load data ----
    df = pd.read_csv("train.csv")
    if args.target not in df.columns:
        args.target = df.columns[-1]
        print(f"[info] --target not found; using last column as target: {args.target}")

    # explicit column lists (your schema)
    numeric_cols = [f"Num_Col{i}" for i in range(1, 26)]
    categorical_cols = [f"Nom_Col{i}" for i in range(26, 44)]

    # be robust to dtype issues
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # split
    y = df[args.target].astype(int).values
    X = df.drop(columns=[args.target])

    # ---- fixed P★ preprocessor + LOF outlier step ----
    pre = build_preprocessor_pstar(numeric_cols, categorical_cols)
    outlier_step = LOFDropper(contamination=args.contamination, n_neighbors=20)

    # ---- models + grids ----
    wanted = {m.strip() for m in args.models.split(",")}
    models = []
    if "dt" in wanted:
        models.append(("DT", DecisionTreeClassifier(random_state=args.seed), {
            "clf__criterion": ["gini", "entropy"],
            "clf__max_depth": [4, 6, 8, 12, None],
            "clf__min_samples_leaf": [1, 3, 5],
            "clf__ccp_alpha": [0.0, 1e-4, 1e-3],
        }))
    if "rf" in wanted:
        models.append(("RF", RandomForestClassifier(random_state=args.seed, n_jobs=-1), {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [None, 8, 12],
            "clf__max_features": ["sqrt", "log2"],
            "clf__min_samples_leaf": [1, 2, 4],
        }))
    if "knn" in wanted:
        models.append(("KNN", KNeighborsClassifier(), {
            "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
            "clf__metric": ["euclidean", "manhattan", "chebyshev"],
            "clf__weights": ["uniform", "distance"],
        }))
    if "gnb" in wanted:
        models.append(("GNB", GaussianNB(), {
            "clf__var_smoothing": [1e-9, 1e-8, 1e-7],
        }))
    if "mnb" in wanted:
        models.append(("MNB", MultinomialNB(), {
            "clf__alpha": [0.1, 0.5, 1.0],
        }))

    # ---- CV + scoring ----
    cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    scoring = {"f1": make_scorer(f1_score), "accuracy": make_scorer(accuracy_score)}

    # ---- run searches and collect results ----
    all_rows = []
    best_global = None  # (mean_f1, -std_f1, model_name, best_estimator, best_params)

    for name, clf, grid in models:
        print(f"\n[info] searching model: {name}")
        pipe = ImbPipeline(steps=[
            ("outliers", outlier_step),  # fixed from P★
            ("pre", pre),                # fixed P★
            ("clf", clf),
        ])

        gs = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=scoring,
            refit="f1",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
            verbose=1
        )
        gs.fit(X, y)

        # pull rows from cv_results_
        res = pd.DataFrame(gs.cv_results_)
        for _, r in res.iterrows():
            params = {k.replace("param_", ""): to_readable(v) for k, v in r.items() if k.startswith("param_")}
            all_rows.append({
                "model": name,
                **params,
                "mean_f1": float(r["mean_test_f1"]),
                "std_f1": float(r["std_test_f1"]),
                "mean_accuracy": float(r["mean_test_accuracy"]),
            })

        # track global best by mean_f1 then lower std_f1
        mean_f1 = float(res.loc[res["rank_test_f1"] == 1, "mean_test_f1"].iloc[0])
        std_f1 = float(res.loc[res["rank_test_f1"] == 1, "std_test_f1"].iloc[0])
        best_params = gs.best_params_
        print(f"[best {name}] F1={mean_f1:.4f} ± {std_f1:.4f}  params={best_params}")

        key = (mean_f1, -std_f1)
        if best_global is None or key > (best_global[0], best_global[1]):
            best_global = (mean_f1, -std_f1, name, gs.best_estimator_, best_params)

    # ---- save leaderboard ----
    results = pd.DataFrame(all_rows)
    results = results.sort_values(by=["mean_f1", "std_f1"], ascending=[False, True]).reset_index(drop=True)

    out_dir = os.path.dirname(args.out) or "."
    os.makedirs(out_dir, exist_ok=True)
    results.to_csv(args.out, index=False)

    # ---- save winner summary ----
    best_mean_f1, neg_std, best_name, best_est, best_params = best_global
    best_json = {
        "model": best_name,
        "mean_f1": round(best_mean_f1, 4),
        "std_f1": round(-neg_std, 4),
        "params": {k.replace("clf__", ""): to_readable(v) for k, v in best_params.items()},
        "preprocessing": {
            "num_impute": "median",
            "scale": "MinMaxScaler",
            "cat_impute": "constant('MISSING')",
            "encoding": "OneHotEncoder(ignore_unknown, dense)",
            "outliers": f"LOFDropper(contamination={args.contamination})",
        },
        "seed": args.seed,
        "folds": args.folds,
    }
    best_path = os.path.join(out_dir, "part3_best_model.json")
    with open(best_path, "w") as f:
        json.dump(best_json, f, indent=2)

    print("\n=== Part 3 — Best model (with P★) ===")
    print(json.dumps(best_json, indent=2))
    print(f"\n[saved] leaderboard -> {args.out}")
    print(f"[saved] winner -> {best_path}")
    print("\nNext: refit this pipeline on ALL training rows and generate test predictions (Part 4).")


if __name__ == "__main__":
    main()
