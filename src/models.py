import argparse, json, os
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from preprocess import build_preprocessor


HERE      = os.path.dirname(os.path.abspath(__file__))       
ROOT      = os.path.abspath(os.path.join(HERE, ".."))        
DATA_DIR  = os.path.join(ROOT, "datasets")                  

TRAIN_PATH  = os.path.join(DATA_DIR, "train.csv")             
CONFIG_PATH = os.path.join(HERE, "best_model_config.json")     

TARGET_COL = "Target (Col44)"

MODEL_GRIDS = {
    "DecisionTree": (
        DecisionTreeClassifier,
        {
            "clf__criterion": ["gini"],
            "clf__max_depth": [10, None],
            "clf__min_samples_leaf": [1, 4],
            "clf__min_samples_split": [2, 10],
            "clf__class_weight": [None, "balanced"],
        },
    ),
    "RandomForest": (
        RandomForestClassifier,
        {
            "clf__n_estimators": [200],
            "clf__max_depth": [20, None],
            "clf__min_samples_leaf": [1, 4],
            "clf__min_samples_split": [2, 10],
            "clf__class_weight": ["balanced_subsample"],
            "clf__max_features": ["sqrt"],
        },
    ),
    "kNN": (
        KNeighborsClassifier,
        {
            "clf__n_neighbors": [5, 11, 21],
            "clf__weights": ["uniform", "distance"],
            "clf__metric": ["euclidean"],
        },
    ),
    "GaussianNB": (
        GaussianNB,
        {"clf__var_smoothing": [1e-9, 1e-8, 1e-7]},
    ),
}

def main():
    ap = argparse.ArgumentParser(description="Compare preprocess × model grids and save winning config")
    ap.add_argument("--train",    default=TRAIN_PATH)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--cv",       type=int, default=5)
    ap.add_argument("--scalers",  default="standard,minmax")
    ap.add_argument("--outliers", default="none,iforest,lof")   
    ap.add_argument("--out",      default=CONFIG_PATH)
    args = ap.parse_args()

    # --- data
    df = pd.read_csv(args.train)
    y  = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int).values
    X  = df.drop(columns=[TARGET_COL])
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    skf = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.seed)

   
    variant_results = []
    scalers = [s.strip() for s in args.scalers.split(",") if s.strip()]
    outlier_methods = [m.strip() for m in args.outliers.split(",") if m.strip()]

    for s in scalers: # pre
        for o in outlier_methods:
            pre = build_preprocessor(num_cols, cat_cols, scaler_name=s, outlier_method=o, seed=args.seed)
            rf = RandomForestClassifier(
                n_estimators=300, max_depth=20, max_features="sqrt",
                class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1
            )
            pipe = Pipeline([("pre", pre), ("clf", rf)])
            cvres = cross_validate(pipe, X, y, cv=skf, scoring={"f1": "f1"}, n_jobs=-1)
            mean, std = float(cvres["test_f1"].mean()), float(cvres["test_f1"].std(ddof=1))
            print(f"{s:8s} | {o:7s} -> F1 {mean:.4f} ± {std:.4f}")
            variant_results.append({"scaler": s, "outlier": o, "f1_mean": mean, "f1_std": std})

    preprocessing_total = pd.DataFrame(variant_results).sort_values(
        ["f1_mean", "f1_std"], ascending=[False, True]
    ).reset_index(drop=True)

    preprocessing_total.to_csv("preprocess_cv_results.csv", index=False)
    best_variant = preprocessing_total.iloc[0].to_dict()


    print(f"best: ={best_variant['scaler']} | outlier={best_variant['outlier']}")

    pre_best = build_preprocessor(num_cols, cat_cols, best_variant["scaler"], best_variant["outlier"], args.seed)

   
   

    rank = []
    best_name, best_mean, best_std, best_params = None, -np.inf, np.inf, None

    for name, (Cls, grid) in MODEL_GRIDS.items():
        est_params = Cls().get_params()
        est = Cls(random_state=args.seed) if "random_state" in est_params else Cls()
        pipe = Pipeline([("pre", pre_best), ("clf", est)])
        gs = GridSearchCV(pipe, param_grid=grid, scoring="f1", cv=skf, n_jobs=-1, refit=True, verbose=0)
        gs.fit(X, y)

        mean = float(gs.best_score_)
        std  = float(gs.cv_results_["std_test_score"][gs.best_index_])
        print(f"[{name}] F1={mean:.4f} ± {std:.4f}  best={gs.best_params_}")

        rank.append({"model": name, "f1_mean": mean, "f1_std": std, "params": gs.best_params_})
        if (mean > best_mean) or (np.isclose(mean, best_mean) and std < best_std):
            best_name, best_mean, best_std, best_params = name, mean, std, gs.best_params_

    rdf = pd.DataFrame(rank).sort_values(["f1_mean", "f1_std"], ascending=[False, True]).reset_index(drop=True)
    rdf.to_csv("model_cv_results.csv", index=False)
    
    print(rdf.head(1))

    summary = {
        "preprocess_choice": {"scale": best_variant["scaler"], "outlier": best_variant["outlier"]},
        "winner_model": best_name,
        "best_params": best_params,
        "cv_mean_f1": best_mean,
        "cv_std_f1": best_std,
        "cv": {"n_splits": args.cv, "seed": args.seed},
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {args.out}")

if __name__ == "__main__":
    main()
