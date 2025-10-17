#!/usr/bin/env python3
# main.py
import argparse, json, sys, os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor, KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import make_scorer, f1_score, accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin

# =========================
# Config (reproducible!)
# =========================
SEED_DEFAULT = 42
CV_DEFAULT = 5

# =========================
# OHE helper (version-safe)
# =========================
def OHE():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# =========================
# Outlier score appender (optional)
# =========================
class OutlierScoreAppender(BaseEstimator, TransformerMixin):
    """
    Adds an outlier score as one extra numeric feature.
    Fits on numerics only with median -> StandardScaler -> IF/LOF.
    """
    def __init__(self, num_cols=None, method="none", contamination="auto", n_neighbors=20, random_state=42):
        self.num_cols = num_cols
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("OutlierScoreAppender expects a pandas DataFrame")
        self.num_cols_ = list(self.num_cols) if self.num_cols is not None \
            else X.select_dtypes(include=[np.number]).columns.tolist()

        self._imp = SimpleImputer(strategy="median").fit(X[self.num_cols_])
        Zimp = self._imp.transform(X[self.num_cols_])
        self._sc  = StandardScaler().fit(Zimp)
        Z = self._sc.transform(Zimp)

        if self.method == "iforest":
            self.model_ = IsolationForest(
                contamination=self.contamination, n_estimators=200, random_state=self.random_state
            ).fit(Z)
        elif self.method == "lof":
            k = max(5, min(self.n_neighbors, Z.shape[0] - 1))
            self.model_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Z)
        else:
            self.model_ = None
        return self

    def transform(self, X):
        Z = self._sc.transform(self._imp.transform(X[self.num_cols_]))
        if self.model_ is None:
            score = np.zeros((len(X), 1))
        else:
            score = self.model_.score_samples(Z).reshape(-1, 1)
        X2 = X.copy()
        X2["outlier_score"] = score
        return X2

# =========================
# Build preprocessor
# =========================
def build_preprocessor(num_cols, cat_cols, scaler_name, outlier_method, seed):
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()

    steps = []
    if outlier_method != "none":
        steps.append(("outlier", OutlierScoreAppender(num_cols=num_cols, method=outlier_method, random_state=seed)))

    num_cols_use = list(num_cols) + (["outlier_score"] if outlier_method != "none" else [])

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median", add_indicator=True)),
                ("scaler", scaler)
            ]), num_cols_use),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OHE())
            ]), cat_cols),
        ],
        verbose_feature_names_out=False
    )

    return Pipeline(steps + [("ct", pre)]) if steps else Pipeline([("ct", pre)])

# =========================
# Write spec-compliant report
# =========================
def write_infs4203_file(path, test_preds, acc_mean, f1_mean):
    """
    Writes exactly 2,714 rows:
      - rows 1..2713: integer predictions for test rows
      - row 2714: accuracy, f1 (rounded to 3 decimals)
    Every row ends with a comma.
    """
    with open(path, "w", encoding="utf-8") as f:
        for p in test_preds:
            f.write(f"{int(p)},\n")
        f.write(f"{acc_mean:.3f},{f1_mean:.3f},\n")

# =========================
# Main pipeline
# =========================
def main():
    parser = argparse.ArgumentParser(description="INFS4203 Data-oriented project runner")

    # Defaults so you can simply run: python main.py
    parser.add_argument("--train", default="train.csv", help="Path to train.csv")
    parser.add_argument("--test",  default="test_data.csv", help="Path to test_data.csv")
    parser.add_argument("--sid",   default="s4755276", help="Student ID, e.g., s1234567 (used for output filename)")

    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--cv",   type=int, default=CV_DEFAULT)

    # If your course weeks did NOT cover IF/LOF, set default to 'none'
    parser.add_argument("--outlier_methods", default="none,iforest,lof",
                        help="Comma-separated from {none,iforest,lof}. Use 'none' if unsure.")
    parser.add_argument("--scalers", default="standard,minmax",
                        help="Comma-separated from {standard,minmax}")

    args = parser.parse_args()

    SEED = args.seed
    CV   = args.cv
    sid  = args.sid.strip()

    # keep your existing SID check
    assert sid.startswith("s") and sid[1:].isdigit() and len(sid) == 8, "sid must look like s1234567"

    # === Load data
    df = pd.read_csv(args.train)
    target_col = "Target (Col44)"
    y = pd.to_numeric(df[target_col], errors="raise").astype(int).values
    X = df.drop(columns=[target_col])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)

    # ===== STEP 1: preprocessing variant scan (optional shortlist with RF)
    variant_results = []
    scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    outlier_methods = [m.strip() for m in args.outlier_methods.split(",") if m.strip()]
    for scaler_name in scalers:
        for outlier_method in outlier_methods:
            pre = build_preprocessor(num_cols, cat_cols, scaler_name, outlier_method, SEED)
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=20, max_features="sqrt",
                class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
            )
            pipe = Pipeline([("pre", pre), ("clf", rf)])
            cv = cross_validate(pipe, X, y, cv=skf, scoring={"f1": "f1"}, n_jobs=-1, return_train_score=False)
            f1_mean, f1_std = cv["test_f1"].mean(), cv["test_f1"].std(ddof=1)
            print(f"{scaler_name:8s} | {outlier_method:7s} -> F1 {f1_mean:.4f} ± {f1_std:.4f}")
            variant_results.append({
                "scaler": scaler_name, "outlier": outlier_method,
                "f1_mean": float(f1_mean), "f1_std": float(f1_std),
            })

    variant_df = pd.DataFrame(variant_results).sort_values(["f1_mean","f1_std"], ascending=[False, True]).reset_index(drop=True)
    variant_df.to_csv("preprocess_cv_results.csv", index=False)
    best_variant = variant_df.iloc[0].to_dict()
    print("\n=== Ranked preprocessing variants (best first) ===")
    print(variant_df.head(10))
    print(f"\nSelected P*: scaler={best_variant['scaler']} | outlier={best_variant['outlier']} "
          f"with F1 {best_variant['f1_mean']:.4f} ± {best_variant['f1_std']:.4f}")

    pre_best = build_preprocessor(num_cols, cat_cols, best_variant["scaler"], best_variant["outlier"], SEED)

    # ===== STEP 2: model grids (Weeks 2–8 core classifiers)
    grids = {
        "DecisionTree": (
            DecisionTreeClassifier(random_state=SEED),
            {
                "clf__criterion": ["gini"],
                "clf__max_depth": [10, None],
                "clf__min_samples_leaf": [1, 4],
                "clf__min_samples_split": [2, 10],
                "clf__class_weight": [None, "balanced"],
            }
        ),
        "RandomForest": (
            RandomForestClassifier(random_state=SEED, n_jobs=-1, max_features="sqrt"),
            {
                "clf__n_estimators": [200],
                "clf__max_depth": [20, None],
                "clf__min_samples_leaf": [1, 4],
                "clf__min_samples_split": [2, 10],
                "clf__class_weight": ["balanced_subsample"],
            }
        ),
        "kNN": (
            KNeighborsClassifier(metric="euclidean"),
            {
                "clf__n_neighbors": [5, 11, 21],
                "clf__weights": ["uniform", "distance"],
            }
        ),
        "GaussianNB": (
            GaussianNB(),
            { "clf__var_smoothing": [1e-9, 1e-8, 1e-7] }
        ),
    }

    # ===== STEP 3: grid search each model (scoring=F1), pick winner by mean F1 then std
    rank = []
    best_name, best_mean, best_std, best_params, best_est = None, -np.inf, np.inf, None, None
    for name, (est, param_grid) in grids.items():
        pipe = Pipeline([("pre", pre_best), ("clf", est)])
        gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=skf, verbose=2, n_jobs=-1, refit=True)
        gs.fit(X, y)
        mean = float(gs.best_score_)
        std  = float(gs.cv_results_["std_test_score"][gs.best_index_])
        rank.append({"model": name, "f1_mean": mean, "f1_std": std, "params": gs.best_params_})
        print(f"[{name}] F1={mean:.4f} ± {std:.4f}  best={gs.best_params_}")

        if (mean > best_mean) or (np.isclose(mean, best_mean) and std < best_std):
            best_name, best_mean, best_std = name, mean, std
            best_params, best_est = gs.best_params_, gs.best_estimator_

    rank_df = pd.DataFrame(rank).sort_values(["f1_mean","f1_std"], ascending=[False,True]).reset_index(drop=True)
    rank_df.to_csv("model_cv_results.csv", index=False)
    print("\n=== Ranked models (best first) ===")
    print(rank_df.head(10))
    print(f"\nWinner: {best_name} | F1={best_mean:.4f} ± {best_std:.4f}")

    # ===== STEP 4: CV evaluation (Accuracy + F1) on the final chosen pipeline (no custom threshold)
    final_pipe = Pipeline([("pre", pre_best), ("clf", grids[best_name][0])])
    final_pipe.set_params(**best_params)
    scores = cross_validate(final_pipe, X, y, cv=skf, scoring={"acc": "accuracy", "f1": "f1"}, n_jobs=-1)
    acc_cv = float(scores["test_acc"].mean())
    f1_cv  = float(scores["test_f1"].mean())
    print(f"[FINAL CV] Accuracy={acc_cv:.4f} | F1={f1_cv:.4f}")

    # ===== STEP 5: fit on full training & predict test labels (integers)
    final_pipe.fit(X, y)

    test_df = pd.read_csv(args.test)
    # Basic sanity check on expected test size from spec (2713 rows)
    if len(test_df) != 2713:
        print(f"WARNING: Expected 2713 test rows; got {len(test_df)} (continuing anyway).")

    test_pred = final_pipe.predict(test_df).astype(int)

    # ===== STEP 6: write spec-compliant result file
    out_name = f"{sid}.infs4203"
    write_infs4203_file(out_name, test_pred, acc_cv, f1_cv)
    print(f"\nWrote {out_name} with {len(test_pred)} predictions + final line (acc,f1).")

    # ===== STEP 7: also save summary JSON (handy for README)
    summary = {
        "preprocess_choice": {"scale": best_variant["scaler"], "outlier": best_variant["outlier"]},
        "winner_model": best_name,
        "best_params": best_params,
        "cv_mean_f1": best_mean,
        "cv_std_f1": best_std,
        "cv_eval": {"mean_accuracy": acc_cv, "mean_f1": f1_cv},
        "cv": {"n_splits": CV, "seed": SEED}
    }
    with open("best_model_config.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
