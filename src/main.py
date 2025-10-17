#!/usr/bin/env python3
# main.py — no CLI args needed
import os, json
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from preprocess import build_preprocessor

# -------------------------
# Constants / defaults
# -------------------------
TARGET_COL          = "Target (Col44)"
SEED_DEFAULT        = 42
CV_DEFAULT          = 5
EXPECTED_TEST_ROWS  = 2713

# Paths (relative to this file)
HERE        = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH  = os.path.join(HERE, "train.csv")
TEST_PATH   = os.path.join(HERE, "test_data.csv")
CONFIG_PATH = os.path.join(HERE, "best_model_config.json")
SID         = "s4755276"   # <- set your student ID here

# Fallback winner (used if best_model_config.json is missing)
FINAL_FALLBACK = {
    "preprocess_choice": {"scale": "minmax", "outlier": "lof"},  # change to {"scale":"minmax","outlier":"none"} if you prefer
    "winner_model": "RandomForest",
    "best_params": {
        "clf__n_estimators": 200,
        "clf__max_depth": 20,
        "clf__min_samples_leaf": 1,
        "clf__min_samples_split": 2,
        "clf__class_weight": "balanced_subsample",
        "clf__max_features": "sqrt",
    },
}

def write_infs4203_file(path, test_preds, acc_mean, f1_mean):
    """Write 2713 prediction lines + final 'accuracy,f1,' with 3 decimals; every line ends with comma."""
    with open(path, "w", encoding="utf-8") as f:
        for p in test_preds:
            f.write(f"{int(p)},\n")
        f.write(f"{acc_mean:.3f},{f1_mean:.3f},\n")

def _load_locked_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded locked config from {CONFIG_PATH}")
        return cfg
    print("[WARN] best_model_config.json not found; using FINAL_FALLBACK embedded in main.py")
    return FINAL_FALLBACK

def _new_estimator(name, seed, params):
    if name == "DecisionTree":
        return DecisionTreeClassifier(random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(
            random_state=seed,
            n_jobs=-1,
            max_features=params.get("clf__max_features", "sqrt"),
        )
    if name == "kNN":
        return KNeighborsClassifier(metric="euclidean")
    if name == "GaussianNB":
        return GaussianNB()
    raise ValueError(f"Unknown model: {name}")

def main():
    # --- basic checks
    assert SID.startswith("s") and SID[1:].isdigit() and len(SID) == 8, "SID must look like s1234567"
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"Missing {TRAIN_PATH}")
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(f"Missing {TEST_PATH}")

    # --- load data
    df = pd.read_csv(TRAIN_PATH)
    y  = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int).values
    X  = df.drop(columns=[TARGET_COL])
    X_test = pd.read_csv(TEST_PATH)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # --- load locked winner
    cfg     = _load_locked_config()
    scale   = cfg["preprocess_choice"]["scale"]
    outlier = cfg["preprocess_choice"]["outlier"]
    model   = cfg["winner_model"]
    params  = cfg["best_params"]

    # --- build pipeline
    pre  = build_preprocessor(num_cols, cat_cols, scaler_name=scale, outlier_method=outlier, seed=SEED_DEFAULT)
    est  = _new_estimator(model, SEED_DEFAULT, params)
    pipe = Pipeline([("pre", pre), ("clf", est)])
    pipe.set_params(**params)

    # --- CV (Accuracy + binary F1 with positive class = 1)
    skf = StratifiedKFold(n_splits=CV_DEFAULT, shuffle=True, random_state=SEED_DEFAULT)
    scoring = {"acc": "accuracy", "f1": make_scorer(f1_score, average="binary", pos_label=1)}
    scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    acc_cv = float(scores["test_acc"].mean())
    f1_cv  = float(scores["test_f1"].mean())
    print(f"[CV] Accuracy={acc_cv:.4f}  F1(binary)={f1_cv:.4f}")

    # --- fit full train + predict test
    pipe.fit(X, y)
    if len(X_test) != EXPECTED_TEST_ROWS:
        print(f"[WARN] expected {EXPECTED_TEST_ROWS} test rows; got {len(X_test)} (continuing).")
    preds = pipe.predict(X_test).astype(int)

    # --- write report
    out_name = os.path.join(HERE, f"{SID}.infs4203")
    write_infs4203_file(out_name, preds, acc_cv, f1_cv)
    print(f"[DONE] Wrote {out_name}")

if __name__ == "__main__":
    main()
