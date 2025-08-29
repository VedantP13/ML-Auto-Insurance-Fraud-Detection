# Auto Insurance Fraud Detection

## Overview

This project provides two end-to-end variants for auto insurance fraud detection: a recommended pipeline-based Logistic Regression system that accepts raw text categories and performs preprocessing internally, and a legacy label-encoded Decision Tree system that expects numeric inputs for all fields.[^3][^4]

The Flask application exposes a friendly web UI with dropdowns, an analytics dashboard, and JSON APIs, and it supports batch predictions via a CLI utility for CSV files.[^2][^5]

## Environment setup

- Create and activate a virtual environment, then install dependencies pinned in requirements.txt to avoid serialization/runtime mismatches across scikit-learn and imbalanced-learn versions.[^6]
- Ensure the working directory is the project root so relative paths to data/, models/, and templates/ resolve correctly when running the apps and training scripts.[^2]

Commands:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Train models

Two training flows are provided; train at least one before running the corresponding app.

- Pipeline (recommended): trains a Logistic Regression inside an imbalanced-learn Pipeline with a ColumnTransformer that imputes, scales numeric columns, and one-hot encodes categoricals, while engineering year/month from date-like columns and dropping ID/leakage fields.[^4]
- Legacy Decision Tree: trains a Decision Tree with SMOTE and StandardScaler after label-encoding many columns (including dates and some fields that would be better treated as categoricals), saving separate artifacts expected by the legacy app.[^3]

Commands:

```bash
# Recommended pipeline trainer (writes models/pipeline.pkl and models/engineered_feature_columns.pkl)
python src/train_model_pipeline.py

# Legacy trainer (writes models/dt_model.pkl, models/dt_scaler.pkl, models/dt_feature_names.pkl)
python src/train_model.py
```

Artifacts produced:

- Pipeline: pipeline.pkl and engineered_feature_columns.pkl in models/ used by the web app and JSON APIs.[^2][^4]
- Legacy: dt_model.pkl, dt_scaler.pkl, dt_feature_names.pkl in models/ used by the numeric-only web form and CLI when the pipeline is absent.[^5][^1][^3]

## Run the app

Two Flask apps exist; run the one that matches the model artifacts trained.

- Pipeline web app (recommended UI): provides dropdowns, combined City, State selection, date pickers, feature importance, and consolidated preprocessing at inference time.[^2]
- Legacy web app (numeric-only): exposes a plain form requiring numeric-encoded inputs for all fields, matching the label-encoded training approach.[^1]

Commands from project root:

```bash
# Pipeline UI with smarter preprocessing and JSON APIs
python app/app.py

# Legacy numeric-only UI
python app/app_old.py
```

After the server starts, open the local server on port 5000 to use the UI or APIs for predictions and analytics.[^2]

## UI and templates

- Pipeline UI template (templates/index_pipeline.html) shows the prediction, risk band, probability, and top contributing factor groups aggregated back to the original fields for interpretability.[^7]
- Legacy UI template (templates/index.html) renders dynamic fields based on feature names when running with the legacy model and shows the prediction with optional probability for the pipeline path.[^8][^1]
- A shared base template (templates/base.html) is used to layout pages consistently across the app.[^9]
- The dashboard template (templates/dashboard.html) displays model metadata, a sample of live predictions from the trained pipeline, and simple counts of fraudulent vs. non-fraudulent predictions to help verify integration.[^10][^2]

## Data and features

- The app and trainers read data from data/insurance_claims.csv and drop ID/leakage-like columns such as _c39, incident_location, policy_number, and insured_zip before training or inference alignment, mirroring the training-time preprocessing.[^4][^2]
- In the pipeline, date-like columns are split into engineered parts (e.g., policy_bind_date_year/month and incident_date_year/month), and the original date strings are removed before modeling, exactly matching the training logic.[^4][^2]
- The pipeline app curates a practical subset of form fields, builds dropdown choices from the raw CSV, enforces valid state-city combinations, and fills unspecified numeric and categorical fields with neutral defaults that align with the pipeline’s imputers and encoders.[^2]

## Making predictions (web)

- On the pipeline UI, fill in selected fields or leave them blank to rely on the pipeline’s imputers and encoders; the app converts date inputs to engineered year/month parts and safely handles unknown categories via OneHotEncoder(handle_unknown="ignore").[^4][^2]
- The result includes the predicted label (Fraudulent / Not Fraudulent), probability, and a UX-oriented risk band: Low (<0.3), Medium (0.3–<0.7), or High (≥0.7), derived from the model’s predicted probability.[^2]
- The “Top factors” list aggregates absolute model coefficients from the one-hot space back to base input fields (e.g., insured_hobbies) to provide a concise interpretability summary aligned with the fields supplied in the form.[^2]

## API endpoints

The pipeline app exposes JSON endpoints for programmatic use.

- POST /api/predict: accepts either a full engineered row or a partial payload with raw fields and ISO dates, returning fraud_probability, prediction, and risk_band.[^2]
- GET /api/cities/<state>: returns a list of valid cities for the requested state to support dependent dropdowns in the UI.[^2]
- GET /api/analytics: returns model metadata, feature categories, and a few sample rows to help verify the end-to-end wiring.[^2]

Example:

```bash
# Minimal example payload structure (fields may be omitted; pipeline imputers apply)
curl -X POST localhost:5000/api/predict   -H "Content-Type: application/json"   -d '{"age": 45, "policy_state": "IL", "incident_type": "Single Vehicle Collision", "policy_bind_date": "2015-06-01", "incident_date": "2015-11-23", "total_claim_amount": 4500}'
```

## Dashboard

- Open /dashboard to view model details, latest load status, number of engineered features, and example predictions generated by passing training samples through the same build_full_row_from_form logic used by the web form and APIs.[^2]
- Fraudulent vs. Not Fraudulent counts are computed from those sample predictions to provide a quick, visual sanity check that the pipeline is loaded and producing probabilities as expected.[^10][^2]

## Batch predictions (CLI)

- The CLI auto-detects which artifacts are present and uses the pipeline when available, otherwise falling back to the legacy Decision Tree with its scaler and explicit feature column order.[^5]
- Provide a CSV of inputs aligned to the expected model, and the CLI will emit a predictions CSV with labels and probabilities for the pipeline variant or labels for the legacy variant.[^5]

Command:

```bash
python src/predict_cli.py path/to/input.csv --out preds.csv
```

## Project structure

A representative layout aligned with this repository:

```
.
├─ app/
│  ├─ app.py                 # Flask app (pipeline UI + APIs + dashboard)
│  └─ app_old.py             # Legacy numeric-only Flask app
├─ data/
│  └─ insurance_claims.csv   # Training/inference data
├─ models/
│  ├─ pipeline.pkl
│  ├─ engineered_feature_columns.pkl
│  ├─ dt_model.pkl
│  ├─ dt_scaler.pkl
│  └─ dt_feature_names.pkl
├─ src/
│  ├─ train_model_pipeline.py
│  ├─ train_model.py
│  └─ predict_cli.py
├─ templates/
│  ├─ base.html
│  ├─ index.html
│  ├─ index_pipeline.html
│  └─ dashboard.html
├─ tests/
│  ├─ test_load.py
│  └─ test_predict.py
└─ README.md
```

## Testing

- Unit tests are organized under tests/ and can be executed after installing requirements; running tests after training validates that artifacts load and basic predictions execute end-to-end.[^6]
- Use a clean virtual environment to avoid dependency conflicts that could cause model deserialization issues during testing.[^6]

## Notes and improvements

- Prefer the pipeline path for production-like usage because it encapsulates preprocessing, handles unknown categories robustly, and mirrors training-time feature engineering at inference.[^4][^2]
- The legacy label-encoding flow is kept for comparison and educational purposes but is brittle when unseen labels appear or when the UI inputs diverge from training encodings, so it should be treated as a baseline only.[^3][^1]
- For security and privacy, this demo is not production-hardened; sanitize inputs, manage PII carefully, and add proper logging, authentication, and monitoring before deployment.[^2]

## Quick commands 

```bash
# Install
pip install -r requirements.txt

# Train (recommended)
python src/train_model_pipeline.py

# Run (recommended UI)
python app/app.py

# Legacy (optional)
python src/train_model.py
python app/app_old.py

# CLI
python src/predict_cli.py input.csv --out preds.csv
```

<div style="text-align: center">⁂</div>

[^1]: app_old.py
[^2]: app.py
[^3]: train_model.py
[^4]: train_model_pipeline.py
[^5]: predict_cli.py
[^6]: README.md
[^7]: index_pipeline.html
[^8]: index.html
[^9]: base.html
[^10]: dashboard.html
