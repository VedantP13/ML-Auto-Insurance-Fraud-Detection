import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent   # parent of src/ or app/ is project root
DATA_PATH = BASE_DIR / "data" / "insurance_claims.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# -------------------------
# Load data
# -------------------------
df = pd.read_csv(DATA_PATH)

# Drop obvious ID/leakage-like columns
drop_cols = ['_c39', 'incident_location', 'policy_number', 'insured_zip']
for c in drop_cols:
    if c in df.columns:
        df.drop(columns=c, inplace=True)

# Target
y = df['fraud_reported'].astype(str).str.upper().map({'Y': 1, 'N': 0}).astype(int)
X = df.drop(columns=['fraud_reported'])

# -------------------------
# Handle date-like columns
# -------------------------
date_like = [c for c in X.select_dtypes(include=['object']).columns if "date" in c.lower()]
for c in date_like:
    try:
        dt = pd.to_datetime(X[c], errors='coerce')
        X[c + "_year"] = dt.dt.year
        X[c + "_month"] = dt.dt.month
    except Exception:
        pass
X.drop(columns=date_like, inplace=True, errors='ignore')

# -------------------------
# Feature lists
# -------------------------
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

# -------------------------
# Preprocessing pipelines
# -------------------------
num_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

cat_tf = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

pre = ColumnTransformer([
    ("num", num_tf, num_cols),
    ("cat", cat_tf, cat_cols)
])

# -------------------------
# Model
# -------------------------
clf = LogisticRegression(max_iter=500, solver="lbfgs")

pipe = ImbPipeline(steps=[
    ("pre", pre),
    ("smote", SMOTE(random_state=42)),
    ("clf", clf)
])

# -------------------------
# Train/Test split
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

pipe.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
y_prob = pipe.predict_proba(X_test)[:, 1]
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))

# -------------------------
# Save model
# -------------------------
joblib.dump(pipe, MODELS_DIR / "pipeline.pkl")
print(f"✅ Saved pipeline to {MODELS_DIR / 'pipeline.pkl'}")
