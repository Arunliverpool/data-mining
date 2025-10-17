# part1_min_compare.py# part1_min_compare.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.base import BaseEstimator, TransformerMixin

TARGET_COL = "Target (Col44)"
SEED = 42
CV = 5

# --- data ---
df = pd.read_csv("train.csv")
y = pd.to_numeric(df[TARGET_COL], errors="raise").astype(int).values
X = df.drop(columns=[TARGET_COL])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

def OHE():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn ≥ 1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # older versions




# --- outlier score as feature (fixed: __init__ doesn't modify params) ---
class OutlierScoreAppender(BaseEstimator, TransformerMixin):
    def __init__(self, num_cols=None, method="none", contamination="auto", n_neighbors=20, random_state=42):
        self.num_cols = num_cols
        self.method = method
        self.contamination = contamination
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Expect pandas DataFrame")
        self.num_cols_ = list(self.num_cols) if self.num_cols is not None \
            else X.select_dtypes(include=[np.number]).columns.tolist()

        self._imp = SimpleImputer(strategy="median").fit(X[self.num_cols_])
        self._sc  = StandardScaler().fit(self._imp.transform(X[self.num_cols_]))
        Z = self._sc.transform(self._imp.transform(X[self.num_cols_]))

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
        score = np.zeros((len(X), 1)) if self.model_ is None else self.model_.score_samples(Z).reshape(-1, 1)
        X2 = X.copy()
        X2["outlier_score"] = score
        return X2



def build_pipeline(scaler_name, outlier_method):
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    steps = []
    if outlier_method != "none":
        steps.append(("outlier", OutlierScoreAppender(num_cols=num_cols, method=outlier_method, random_state=SEED)))
    num_cols_use = num_cols + (["outlier_score"] if outlier_method != "none" else [])
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("scaler", scaler)]), num_cols_use),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OHE())]), cat_cols),
    ], verbose_feature_names_out=False)
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=20, max_features="sqrt",
        class_weight="balanced_subsample", random_state=SEED, n_jobs=-1
    )
    steps += [("pre", pre), ("clf", clf)]
    return Pipeline(steps)




results = []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for scaler_name in ["standard", "minmax"]:
    for outlier_method in ["none", "iforest", "lof"]:
        pipe = build_pipeline(scaler_name, outlier_method)
        cv = cross_validate(pipe, X, y, cv=skf, scoring={"f1": "f1"}, n_jobs=-1, return_train_score=False)
        f1_mean = cv["test_f1"].mean()
        f1_std  = cv["test_f1"].std(ddof=1)
        print(f"{scaler_name:8s} | {outlier_method:7s} -> F1 {f1_mean:.4f} ± {f1_std:.4f}")
        results.append({
            "scaler": scaler_name,
            "outlier": outlier_method,
            "f1_mean": float(f1_mean),
            "f1_std": float(f1_std),
        })

import pandas as pd
ranked = pd.DataFrame(results).sort_values(["f1_mean","f1_std"], ascending=[False, True]).reset_index(drop=True)
print("\n=== Ranked variants (best first) ===")
print(ranked)

best = ranked.iloc[0].to_dict()
print(f"\nSelected: scaler={best['scaler']} | outlier={best['outlier']} "
      f"with F1 {best['f1_mean']:.4f} ± {best['f1_std']:.4f}")
