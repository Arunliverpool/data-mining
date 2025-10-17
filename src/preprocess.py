from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
import numpy as np
import pandas as pd

def OHE():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.4
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)         # older sklearn

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
        score = np.zeros((len(X), 1)) if self.model_ is None else self.model_.score_samples(Z).reshape(-1, 1)
        X2 = X.copy()
        X2["outlier_score"] = score
        return X2

def build_preprocessor(num_cols, cat_cols, scaler_name="minmax", outlier_method="none", seed=42):
    scaler = StandardScaler() if scaler_name == "standard" else MinMaxScaler()
    steps = []
    if outlier_method != "none":
        steps.append(("outlier", OutlierScoreAppender(num_cols=num_cols, method=outlier_method, random_state=seed)))
    num_cols_use = list(num_cols) + (["outlier_score"] if outlier_method != "none" else [])
    ct = ColumnTransformer(
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
    return Pipeline(steps + [("ct", ct)]) if steps else Pipeline([("ct", ct)])
