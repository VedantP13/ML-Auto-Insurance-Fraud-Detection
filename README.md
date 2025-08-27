# Auto Insurance Fraud Detection

This project includes:
- A basic training script (`train_model.py`) and Flask app (`app.py`) that use label-encoded features.
- A friendlier **pipeline** variant that accepts raw text for categorical fields and handles preprocessing internally.

## Quickstart

### 1) Setup a virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -r requirements.txt
```

> If you see an error loading `model.pkl` due to NumPy/Sklearn versions, ensure you use the same versions pinned in `requirements.txt`.

### 3) Train (original approach)
```bash
python train_model.py
```
This saves `model.pkl`, `scaler.pkl`, and `feature_names.pkl`.

### 4) Run the Flask app (original)
```bash
python app.py
```
Visit `http://127.0.0.1:5000`. The form expects **numeric values for all features** (categoricals are label-encoded).

### 5) OPTIONAL: Train the pipeline model (recommended)
```bash
python train_model_pipeline.py
```
This saves `pipeline.pkl` which encapsulates preprocessing + model.

### 6) Run the pipeline Flask app (friendlier UI)
```bash
python app_pipeline.py
```
Visit `http://127.0.0.1:5000`. This version accepts raw text for categoricals (e.g., `incident_state=IL`).

### 7) Batch predictions (CLI)
Prepare a CSV file with **exactly** the columns from `feature_names.pkl` (for the original approach), then run:
```bash
python predict_cli.py input_rows.csv --out preds.csv
```

## Notes & Improvements
- The original `train_model.py` label-encodes many columns (including some numeric ones) and does not persist per-column encoders. The web form works because it asks for numeric inputs, but this is brittle.
- The pipeline variant fixes this using `ColumnTransformer` + `OneHotEncoder`, and trains a logistic regression with SMOTE. You can swap in other models (e.g., `XGBClassifier`) inside the pipeline.
- To add confidence scores, use `predict_proba` and show the probability in the UI (already implemented in `app_pipeline.py`).

## Repository Layout
```
.
├─ app.py                   # original Flask app (numeric inputs only)
├─ train_model.py           # original trainer (label encodes categoricals)
├─ app_pipeline.py          # pipeline-based Flask app
├─ train_model_pipeline.py  # pipeline-based trainer
├─ predict_cli.py           # batch CLI for predictions
├─ requirements.txt
├─ templates/
│  ├─ index.html            # form for app.py
│  └─ index_pipeline.html   # form for app_pipeline.py
├─ insurance_claims.csv
└─ (generated) model.pkl, scaler.pkl, feature_names.pkl, pipeline.pkl
```
