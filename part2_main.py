# part2_main
import json, numpy as np, pandas as pd
from joblib import dump
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.naive_bayes import GaussianNB
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "Target (Col44)"
SEED, CV = 42, 5

# --- Custom transformer used in P* (needed to rebuild from JSON) ---
class OutlierScoreAppender(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, method="none", contamination="auto", n_neighbors=20, random_state=SEED):
        self.num_cols = num_cols; self.method = method
        self.contamination = contamination; self.n_neighbors = n_neighbors; self.random_state = random_state
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expect pandas DataFrame")
        self.num_cols_ = list(self.num_cols) if self.num_cols is not None \
            else X.select_dtypes(include=[np.number]).columns.tolist()
        self._imp = SimpleImputer(strategy="median").fit(X[self.num_cols_])
        self._sc  = StandardScaler().fit(self._imp.transform(X[self.num_cols_]))
        Z = self._sc.transform(self._imp.transform(X[self.num_cols_]))
        if self.method == "iforest":
            self.model_ = IsolationForest(contamination="auto", n_estimators=200,
                                          random_state=self.random_state).fit(Z)
        elif self.method == "lof":
            k = max(5, min(self.n_neighbors, Z.shape[0]-1))
            self.model_ = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Z)
        else:
            self.model_ = None
        return self
    
    def transform(self, X):
        Z = self._sc.transform(self._imp.transform(X[self.num_cols_]))
        s = np.zeros((len(X),1)) if self.model_ is None else self.model_.score_samples(Z).reshape(-1,1)
        X2 = X.copy(); X2["outlier_score"] = s; return X2

def OHE():
    try:    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn ≥1.4
    except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=False)         # older

def build_pre_from_cfg(cfg: dict, X: pd.DataFrame):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
    scaler = StandardScaler() if cfg["scale"] == "standard" else MinMaxScaler()
    steps = []
    if cfg["outlier"] != "none":
        steps.append(("outlier", OutlierScoreAppender(num_cols=num_cols, method=cfg["outlier"], random_state=SEED)))
        num_use = num_cols + ["outlier_score"]
    else:
        num_use = num_cols
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", scaler)]), num_use),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OHE())]), cat_cols),
    ], verbose_feature_names_out=False)
    steps.append(("pre", pre))
    return Pipeline(steps)

# --- Data ---
df = pd.read_csv("train.csv")
y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int).values
X = df.drop(columns=[TARGET_COL])

# --- Rebuild the winning preprocessor from JSON (no joblib needed) ---
with open("best_preprocess_config.json", "r") as f:
    cfg = json.load(f)  # expects keys: scale ('standard'/'minmax'), outlier ('none'/'iforest'/'lof')
pre = build_pre_from_cfg(cfg, X)

# --- Model grids (compact & effective) ---
grids = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=SEED),
        {"clf__criterion":["gini","entropy"],
         "clf__max_depth":[5,10,20,None],
         "clf__min_samples_leaf":[1,2,4],
         "clf__min_samples_split":[2,5,10],
         "clf__class_weight":[None,"balanced"]}
    ),
    "RandomForest": (
        RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {"clf__n_estimators":[200,400],
         "clf__max_depth":[10,20,None],
         "clf__min_samples_leaf":[1,2,4],
         "clf__min_samples_split":[2,5,10],
         "clf__max_features":["sqrt"],
         "clf__class_weight":[None,"balanced_subsample"]}
    ),
    "kNN": (
        KNeighborsClassifier(),
        {"clf__n_neighbors":[3,5,7,11,15],
         "clf__weights":["uniform","distance"],
         "clf__metric":["euclidean"]}
    ),
    "GaussianNB": (
        GaussianNB(),
        {"clf__var_smoothing":[1e-9,1e-8,1e-7]}
    ),
}

# --- CV search for M*, H* ---
skf = StratifiedKFold(n_splits=CV, shuffle=True, random_state=SEED)
rank = []; best_gs = None
for name, (est, param_grid) in grids.items():
    pipe = Pipeline([("pre", pre), ("clf", est)])
    gs = GridSearchCV(pipe, param_grid=param_grid, scoring="f1", cv=skf, n_jobs=-1, refit=True)
    gs.fit(X, y)
    mean = gs.best_score_; std = gs.cv_results_["std_test_score"][gs.best_index_]
    rank.append({"model":name, "f1_mean":mean, "f1_std":std, "params":gs.best_params_})
    if (best_gs is None) or (mean > best_gs.best_score_ + 1e-12) or (abs(mean-best_gs.best_score_)<1e-12 and std < std):
        best_gs = gs

rank_df = pd.DataFrame(rank).sort_values(["f1_mean","f1_std"], ascending=[False,True]).reset_index(drop=True)
rank_df.to_csv("model_cv_results.csv", index=False)
print(rank_df.head(10))

# --- Threshold tuning (OOF) to maximize F1 ---
best_model_name = rank_df.iloc[0]["model"]
best_params = rank_df.iloc[0]["params"]
final_pipe = Pipeline([("pre", pre), ("clf", grids[best_model_name][0])])
final_pipe.set_params(**best_params)

probs = cross_val_predict(final_pipe, X, y, cv=skf, method="predict_proba", n_jobs=-1)[:,1]
taus = np.linspace(0.01, 0.99, 99)
best_tau, best_f1 = 0.5, -1.0
for t in taus:
    preds = (probs >= t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_tau, best_f1 = t, f1
best_acc = accuracy_score(y, (probs >= best_tau).astype(int))

# --- Fit on all data & save final artifacts ---
final_pipe.fit(X, y)
dump(final_pipe, "best_pipeline_cvfit.joblib")
with open("best_model_config.json","w") as f:
    json.dump({
        "model": best_model_name,
        "best_params": best_params,
        "cv_oof_f1": float(best_f1),
        "cv_oof_acc": float(best_acc),
        "threshold": float(best_tau),
        "seed": SEED, "cv": CV,
        "preprocess": {"scale": cfg["scale"], "outlier": cfg["outlier"]}
    }, f, indent=2)
print(f"Saved best_pipeline_cvfit.joblib | model={best_model_name} | F1={best_f1:.4f} @ τ={best_tau:.2f}")
