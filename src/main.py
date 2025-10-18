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
import os
from preprocess import build_preprocessor


TARGET_COL          = "Target (Col44)"
SEED_DEFAULT        = 42
CV_DEFAULT          = 5
EXPECTED_TEST_ROWS  = 2713



HERE      = os.path.dirname(os.path.abspath(__file__))        
ROOT      = os.path.abspath(os.path.join(HERE, ".."))          
DATA_DIR  = os.path.join(ROOT, "datasets")                      

TRAIN_PATH  = os.path.join(DATA_DIR, "train.csv")              
TEST_PATH   = os.path.join(DATA_DIR, "test_data.csv")
CONFIG_PATH = os.path.join(HERE, "best_model_config.json")      
SID         = "s4755276"                                        



FINAL_FALLBACK = {
    "preprocess_choice": {"scale": "minmax", "outlier": "lof"},  
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
    with open(path, "w", encoding="utf-8") as f:
        for p in test_preds:
            f.write(f"{int(p)},\n")
        f.write(f"{acc_mean:.3f},{f1_mean:.3f}\n")

def _load_locked_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded locked config from {CONFIG_PATH}")
        return cfg
    return FINAL_FALLBACK

def model_selection(name, seed, params):
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
 
    df = pd.read_csv(TRAIN_PATH)
    y  = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int).values
    X  = df.drop(columns=[TARGET_COL])
    X_test = pd.read_csv(TEST_PATH)

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    cfg     = _load_locked_config()
    scale   = cfg["preprocess_choice"]["scale"]
    outlier = cfg["preprocess_choice"]["outlier"]
    model   = cfg["winner_model"]
    params  = cfg["best_params"]

    
    pre  = build_preprocessor(num_cols, cat_cols, scaler_name=scale, outlier_method=outlier, seed=SEED_DEFAULT)
    est  = model_selection(model, SEED_DEFAULT, params)
    pipe = Pipeline([("pre", pre), ("clf", est)])
    pipe.set_params(**params)

    skf = StratifiedKFold(n_splits=CV_DEFAULT, shuffle=True, random_state=SEED_DEFAULT)
    scoring = {"acc": "accuracy", "f1": make_scorer(f1_score, average="binary", pos_label=1)}
    scores = cross_validate(pipe, X, y, cv=skf, scoring=scoring, n_jobs=-1)
    acc_cv = float(scores["test_acc"].mean())
    f1_cv  = float(scores["test_f1"].mean())
    print(f"Accuracy={acc_cv:.4f}  F1(binary)={f1_cv:.4f}")


    pipe.fit(X, y)
    preds = pipe.predict(X_test).astype(int)


    out_name = os.path.join(ROOT, f"{SID}.infs4203")   
    write_infs4203_file(out_name, preds, acc_cv, f1_cv)
    
if __name__ == "__main__":
    main()
