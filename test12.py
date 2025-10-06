
import numpy as np, pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

SEED = 42
LABEL = "Target (Col44)"

# --- Version-agnostic OHE factory ---
def make_ohe(min_freq=0.01, dtype=np.float32, force_dense=False):
    params = dict(handle_unknown="ignore", dtype=dtype, min_frequency=min_freq)
    if force_dense:
        # Prefer new param; fall back for older sklearn
        try:
            return OneHotEncoder(sparse_output=False, **params)
        except TypeError:
            return OneHotEncoder(sparse=False, **params)
    else:
        try:
            return OneHotEncoder(sparse_output=True, **params)
        except TypeError:
            return OneHotEncoder(sparse=True, **params)

# --- Data ---
df = pd.read_csv("train.csv")
y = df[LABEL].astype(int)
X = df.drop(columns=[LABEL])
num = [c for c in X.columns if c.startswith("Num_Col")]
cat = [c for c in X.columns if c.startswith("Nom_Col")]

# ---------------- Preprocessors ----------------
# Sparse (good for RF/kNN)
cat_sparse = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe(force_dense=False))
])
prep_noscale_sparse = ColumnTransformer(
    [("num", SimpleImputer(strategy="median"), num),
     ("cat", cat_sparse, cat)]
)
prep_scaled_sparse = ColumnTransformer(
    [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())]), num),
     ("cat", cat_sparse, cat)]
)

# Dense (for NB)
cat_dense = Pipeline([
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", make_ohe(force_dense=True))  # force dense OHE
])
prep_scaled_dense = ColumnTransformer(
    [("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                       ("sc", StandardScaler())]), num),
     ("cat", cat_dense, cat)],
    sparse_threshold=0.0,  # force overall dense output
)

# ---------------- Pipelines ----------------
pipelines = {
    "RF_simple": Pipeline([
        ("prep", prep_noscale_sparse),
        ("clf", RandomForestClassifier(
            n_estimators=300, max_depth=None,
            class_weight="balanced", random_state=SEED, n_jobs=-1))
    ]),
    "KNN_scaled": Pipeline([
        ("prep", prep_scaled_sparse),
        ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", p=2))
    ]),
    "NB_scaled": Pipeline([
        ("prep", prep_scaled_dense),      # DENSE for NB
        ("clf", GaussianNB(var_smoothing=1e-8))
    ]),
}

# ---------------- CV ----------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scorer = {"f1": make_scorer(f1_score), "acc": "accuracy"}

for name, pipe in pipelines.items():
    cv = cross_validate(pipe, X, y, scoring=scorer, cv=skf, n_jobs=-1, error_score="raise")
    print(f"{name}: mean F1={cv['test_f1'].mean():.3f} (std={cv['test_f1'].std():.3f}), "
          f"mean Acc={cv['test_acc'].mean():.3f}")
