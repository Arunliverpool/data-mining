
### INFS4203 – Classification Project (s4755276)

This repository contains everything needed to **reproduce my submitted report** `s4755276.infs4203`, plus the full code for **preprocessing**, **model selection**, **hyperparameter tuning**, **training**, **testing**, and **report generation**.

---

## Final choices (used to produce the report)

**Preprocessing**
- Numeric: `SimpleImputer(strategy="median", add_indicator=True")` → **MinMaxScaler**
- Categorical: `SimpleImputer(strategy="most_frequent")` --> `OneHotEncoder(handle_unknown="ignore")`
- Extra feature: **Local Outlier Factor (LOF)** score appended as `outlier_score`  
  *(LOF is fit inside each CV fold on imputed+scaled numerics to avoid leakage)*

**Classifier**
- **RandomForestClassifier** with:
  - `n_estimators=200`
  - `max_depth=20`
  - `min_samples_leaf=4`
  - `min_samples_split=10`
  - `class_weight="balanced_subsample"`
  - `max_features="sqrt"`

**Evaluation**
- 5-fold **Stratified** CV, seed **42**  
- Metrics: **Accuracy** and **F1 (binary, pos_label = 1)**  
- The mean CV Accuracy & F1 (binary) are placed on the final line of the report file.

** Change CV to 10?

Five folds is perfectly acceptable and common for this level. Going to 10:
- Pros: slightly tighter estimate (lower standard error).
        With your fold std ≈ 0.0143, the standard error of the mean is
        
        5-fold: ~0.0143/√5 ≈ 0.0064
        10-fold: ~0.0143/√10 ≈ 0.0045

That’s a small stability gain.

- Cons: ~2× runtime everywhere (preprocess scans + every model grid). 
        No guarantee it changes your ranking or your final score in a meaningful way.
---

## Environment description

- OS: Linux / macOS / Windows (tested locally)
- Language: **Python 3.10+**
- Packages:
  ```bash
  pip install numpy pandas scikit-learn
  ```
  *(Worked with: numpy ≥ 1.24, pandas ≥ 2.0, scikit-learn ≥ 1.2)*

- Reproducibility:
  - Random seed fixed to **42**
  - CV folds fixed to **5**
  - All transforms/models are fit **within** CV (no global fits before CV)

---

## Repository / ZIP layout

Your ZIP should look like:

```
s4755276.zip
├─ s4755276.infs4203          # the same report submitted to “Report” (MUST be at ZIP root)
├─ README.md                  # or README.txt
├─ datasets/
│  ├─ train.csv
│  └─ test_data.csv
└─ src/
   ├─ main.py                 # FINAL locked pipeline (has a main())
   ├─ model.py          # full comparison: preprocess variants × model grids
   ├─ preprocess.py     # build_preprocessor(), outlier_score, OHE helper
   └─ best_model_config.json  # saved winner from model.py (loaded by main.py)
   
```
---

## Reproduction instructions

### 0) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install numpy pandas scikit-learn
```

### 1) Regenerate my report (final locked pipeline)

From the project root:
```bash
python src/main.py
```

This will:
1. Load `datasets/train.csv` and `datasets/test_data.csv`  
2. Build the **final preprocessor + model** (MinMax + LOF + RF with the params above)  
3. Run **5-fold CV** → **Accuracy** and **F1 (binary, pos=1)**  
4. Fit on the full training data  
5. Predict integer labels for all **2713** test rows  
6. Write the report **`s4755276.infs4203`** to the **project root** (outside `src/`)

**Report format (strict)**  
- Total **2,714** lines:
  - Lines **1–2713**: predictions as `0,` or `1,`
  - Line **2714**: `accuracy,f1`


### 2) (Optional) Re-run the full comparison

```bash
python src/models.py
```

Artifacts produced:
- `preprocess_cv_results.csv` – ranked preprocessing variants by CV F1  
- `model_cv_results.csv` – ranked models (on the best preprocessor) by CV F1  
- `src/best_model_config.json` – locked preprocessor + model + hyperparameters

Then regenerate the report (main will auto-load `best_model_config.json`):
```bash
python src/main.py
```

---

## Selection & tuning summary (actual results)

**Model ranking (on best preprocessor)**

| model         | mean F1 | std F1  | best params (key) |
|---------------|--------:|--------:|-------------------|
| RandomForest  | 0.6615  | 0.0143  | `n_estimators=200, max_depth=20, min_samples_leaf=4, min_samples_split=10, class_weight=balanced_subsample, max_features=sqrt` |
| DecisionTree  | 0.6070  | 0.0089  | `criterion=gini, max_depth=10, class_weight=balanced` |
| GaussianNB    | 0.4782  | 0.0063  | `var_smoothing=1e-7` |
| kNN           | 0.4683  | 0.0144  | `n_neighbors=5, weights=uniform, metric=euclidean` |

**Preprocessing variant scan (RF baseline)**

| scaler   | outlier | mean F1 | std F1 |
|----------|---------|--------:|-------:|
| minmax   | lof     | 0.6488  | 0.0153 |
| standard | lof     | 0.6475  | 0.0149 |
| standard | none    | 0.6451  | 0.0169 |
| minmax   | none    | 0.6435  | 0.0168 |
| standard | iforest | 0.6433  | 0.0228 |
| minmax   | iforest | 0.6428  | 0.0230 |

**Locked config (saved in `src/best_model_config.json`)**

```json
{
  "preprocess_choice": { "scale": "minmax", "outlier": "lof" },
  "winner_model": "RandomForest",
  "best_params": {
    "clf__class_weight": "balanced_subsample",
    "clf__max_depth": 20,
    "clf__max_features": "sqrt",
    "clf__min_samples_leaf": 4,
    "clf__min_samples_split": 10,
    "clf__n_estimators": 200
  },
  "cv_mean_f1": 0.6615266364432273,
  "cv_std_f1": 0.014318237147483523,
  "cv": { "n_splits": 5, "seed": 42 }
}
```

---

## Additional justifications

- **Why LOF?**  
  LOF (Local Outlier Factor) is an anomaly detection method based on the density-based technique
  In the variant scan, adding an **LOF outlier score** to numerics improved mean F1 versus `none` and `iforest` for both scalers. LOF + MinMax produced the top F1 with good stability (std ≈ 0.015).

- **Leakage avoidance**  
  The outlier model (IF/LOF) is fit **inside** each CV fold (after numeric imputation+scaling) and only its **scores** are appended for that fold. No fitting on full data before CV.
  If the outlier model were fitted using the entire dataset before the CV split, the calculation of the outlier score feature for the validation set would implicitly incorporate information derived from the distribution of that validation data itself. This contamination leads to an overly optimistic performance estimate because the model evaluates features that benefited from exposure to the test data during their creation

- **Why RandomForest?**  
  It achieved the highest mean F1 on the best preprocessor and had low variance across folds. The grid was intentionally compact and reproducible.
  Achieving better generalization performance compared to single learners.Random Forest inherently reduces the risk of overfitting compared to a single, deeply grown decision tree

- **Binary F1 in marking**  
  All evaluation uses **F1 (binary)** with **positive class = 1**, aligning with marking guidance.

---

## Training, evaluation, and testing code (where things are)

- **Main entry** (`src/main.py`) – **has a `main()`** and executes the overall process:  
  Load → Preprocess → Train → **5-fold CV (Accuracy & F1 binary)** → Fit full train → Predict test → Write report  
  Seeds fixed: **42**; CV: **5**

- **Preprocessing utilities** (`src/preprocess.py`):  
  `build_preprocessor()` – ColumnTransformer for numerics/categoricals (+ optional `outlier_score`)  
  `outlier_score` – safely computes IF/LOF scores within CV

- **Selection & tuning** (`src/model.py`):  
  Preprocess variant scan (RF baseline) → Model grid search on the best preprocessor → Save `src/best_model_config.json`

---

## AI-assistance disclosure

- Tool: **GPT-5 Thinking (ChatGPT)**  
- Used for: repository scaffolding; path help in file; formating issues. ; README structure.  
- research on best practices for flow
- All code was reviewed locally; experiments and final outputs were produced on my machine.

---

## Submission checklist

- [x] **Report** `s4755276.infs4203` is at the **ZIP root** (and also submitted to the Report link)  
- [x] **README** (this file) at the ZIP root  
- [x] **All code** (`src/main.py`, `src/preprocess.py`, `src/model.py`, and any helpers) – **.py only**  
- [x] **Data** – `datasets/train.csv`, `datasets/test_data.csv`  
- [x] **Seeds fixed** (42) and **CV** (5) declared for reproducibility

---





