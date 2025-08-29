from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "insurance_claims.csv"
MODELS_DIR = BASE_DIR / "models"
PIPE_PATH = MODELS_DIR / "pipeline.pkl"
FEAT_PATH = MODELS_DIR / "engineered_feature_columns.pkl"

app = Flask(__name__)

# -------------------------
# Load artifacts
# -------------------------
pipe = joblib.load(PIPE_PATH)
engineered_cols = joblib.load(FEAT_PATH)  # list[str]

# -------------------------
# Build template schema from training data
# (so we know which columns are numeric vs categorical and can provide dropdowns)
# -------------------------
df_raw = pd.read_csv(DATA_PATH)
# Drop leakage-like columns (same as training)
for c in ['_c39', 'incident_location', 'policy_number', 'insured_zip']:
    if c in df_raw.columns:
        df_raw.drop(columns=c, inplace=True, errors="ignore")

# Derive date engineered columns (as training did)
def add_date_parts(df, col):
    dt = pd.to_datetime(df[col], errors='coerce')
    df[col + "_year"] = dt.dt.year
    df[col + "_month"] = dt.dt.month

for c in [c for c in df_raw.columns if "date" in c.lower()]:
    add_date_parts(df_raw, c)
    df_raw.drop(columns=c, inplace=True, errors='ignore')

# Split numeric/categorical exactly as training would "see" them
num_cols_template = df_raw.select_dtypes(include=[np.number]).columns.tolist()
cat_cols_template = df_raw.select_dtypes(include=['object']).columns.tolist()

# For the UI we’ll present a curated set of fields (keeps the form usable)
FORM_FIELDS = [
    # policy / insured
    "months_as_customer", "age", "policy_state", "policy_csl", "policy_deductable",
    "policy_annual_premium", "umbrella_limit", "insured_sex", "insured_education_level",
    "insured_occupation", "insured_hobbies", "insured_relationship",
    "capital-gains", "capital-loss",
    # incident
    "incident_type", "collision_type", "incident_severity", "authorities_contacted",
    "incident_state", "incident_city", "incident_hour_of_the_day",
    "number_of_vehicles_involved", "property_damage", "bodily_injuries",
    "witnesses", "police_report_available",
    # claim
    "total_claim_amount", "injury_claim", "property_claim", "vehicle_claim",
    # vehicle
    "auto_make", "auto_model", "auto_year",
    # dates (UI) -> we convert to engineered cols below
    "policy_bind_date", "incident_date",
]

# Helper: unique options for dropdowns (limit to keep UI fast)
def uniq(series, limit=50):
    vals = (
        series.dropna()
        .astype(str)
        .replace("", np.nan)
        .dropna()
        .value_counts()
        .index.tolist()
    )
    return vals[:limit]

def format_choice_value(value):
    """Format choice values for better UI display."""
    if pd.isna(value) or value == "":
        return ""
    
    # Convert to string and clean up
    value = str(value).strip()
    
    # Handle specific formatting cases
    if value.upper() in ["YES", "NO", "UNKNOWN"]:
        return value.title()
    elif value.upper() in ["MALE", "FEMALE"]:
        return value.title()
    elif value.upper() in ["HUSBAND", "WIFE", "OTHER"]:
        return value.title()
    elif value.upper() in ["BACHELOR", "MASTER", "PHD", "HIGH SCHOOL"]:
        return value.title()
    elif value.upper() in ["MAJOR DAMAGE", "MINOR DAMAGE", "TOTAL LOSS"]:
        return value.title()
    elif value.upper() in ["SINGLE VEHICLE COLLISION", "VEHICLE THEFT", "PARKED CAR"]:
        return value.title()
    elif value.upper() in ["REAR COLLISION", "FRONT COLLISION", "SIDE COLLISION"]:
        return value.title()
    elif value.upper() in ["POLICE", "AMBULANCE", "FIRE", "NONE"]:
        return value.title()
    else:
        # Default: title case for other values
        return value.title()

# Build choices from original raw file (pre-engineering for nicer labels)
df_for_choices = pd.read_csv(DATA_PATH)

# Create state-city mapping for logical consistency
STATE_CITY_MAPPING = {}
for _, row in df_for_choices.iterrows():
    state = row.get('incident_state', '')
    city = row.get('incident_city', '')
    if pd.notna(state) and pd.notna(city) and state != '' and city != '':
        if state not in STATE_CITY_MAPPING:
            STATE_CITY_MAPPING[state] = set()
        STATE_CITY_MAPPING[state].add(city)

# Convert sets to sorted lists
for state in STATE_CITY_MAPPING:
    STATE_CITY_MAPPING[state] = sorted(list(STATE_CITY_MAPPING[state]))

# Build combined City, State list for single dropdown to guarantee valid pairs
COMBINED_CITY_STATE = []
for state, cities in STATE_CITY_MAPPING.items():
    for city in cities:
        COMBINED_CITY_STATE.append(f"{city}, {state}")
COMBINED_CITY_STATE = sorted(list(dict.fromkeys(COMBINED_CITY_STATE)))


CHOICES = {
    "policy_state": [format_choice_value(v) for v in uniq(df_for_choices.get("policy_state", pd.Series(dtype=str)))],
    "policy_csl": [format_choice_value(v) for v in uniq(df_for_choices.get("policy_csl", pd.Series(dtype=str)))],
    "insured_sex": [format_choice_value(v) for v in uniq(df_for_choices.get("insured_sex", pd.Series(dtype=str)))],
    "insured_education_level": [format_choice_value(v) for v in uniq(df_for_choices.get("insured_education_level", pd.Series(dtype=str)))],
    "insured_occupation": [format_choice_value(v) for v in uniq(df_for_choices.get("insured_occupation", pd.Series(dtype=str)))],
    "insured_hobbies": [format_choice_value(v) for v in uniq(df_for_choices.get("insured_hobbies", pd.Series(dtype=str)))],
    "insured_relationship": [format_choice_value(v) for v in uniq(df_for_choices.get("insured_relationship", pd.Series(dtype=str)))],
    "incident_type": [format_choice_value(v) for v in uniq(df_for_choices.get("incident_type", pd.Series(dtype=str)))],
    "collision_type": [format_choice_value(v) for v in uniq(df_for_choices.get("collision_type", pd.Series(dtype=str)))],
    "incident_severity": [format_choice_value(v) for v in uniq(df_for_choices.get("incident_severity", pd.Series(dtype=str)))],
    "authorities_contacted": [format_choice_value(v) for v in uniq(df_for_choices.get("authorities_contacted", pd.Series(dtype=str)))],
    # Use RAW state codes/values here so they match STATE_CITY_MAPPING keys (e.g., 'NY', 'OH')
    "incident_state": uniq(df_for_choices.get("incident_state", pd.Series(dtype=str))),
    # Present only city options to the UI to avoid confusion about states
    "incident_city": [format_choice_value(v) for v in uniq(df_for_choices.get("incident_city", pd.Series(dtype=str)))],
    "property_damage": ["Yes", "No", "Unknown"],
    "police_report_available": ["Yes", "No", "Unknown"],
    "auto_make": [format_choice_value(v) for v in uniq(df_for_choices.get("auto_make", pd.Series(dtype=str)))],
    "auto_model": [format_choice_value(v) for v in uniq(df_for_choices.get("auto_model", pd.Series(dtype=str)), limit=100)],
}

NUM_DEFAULTS = {
    # sensible defaults for numeric fields if user leaves blank
    "months_as_customer": np.nan,
    "age": np.nan,
    "policy_deductable": np.nan,
    "policy_annual_premium": np.nan,
    "umbrella_limit": 0,
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_hour_of_the_day": np.nan,
    "number_of_vehicles_involved": 1,
    "bodily_injuries": 0,
    "witnesses": 0,
    "total_claim_amount": np.nan,
    "injury_claim": np.nan,
    "property_claim": np.nan,
    "vehicle_claim": np.nan,
    "auto_year": np.nan,
}
CAT_DEFAULTS = {
    # neutral defaults for categoricals
    "policy_state": "",
    "policy_csl": "",
    "insured_sex": "",
    "insured_education_level": "",
    "insured_occupation": "",
    "insured_hobbies": "",
    "insured_relationship": "",
    "incident_type": "",
    "collision_type": "",
    "incident_severity": "",
    "authorities_contacted": "",
    "incident_state": "",
    "incident_city": "",
    "property_damage": "",
    "police_report_available": "",
    "auto_make": "",
    "auto_model": "",
}

def parse_date_to_parts(date_str):
    """Return (year, month) from an ISO yyyy-mm-dd string, else (np.nan, np.nan)."""
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        return dt.year, dt.month
    except Exception:
        return np.nan, np.nan

def build_full_row_from_form(form):
    """
    Create a single-row DataFrame with EXACTLY the engineered_cols,
    converting date fields and filling unspecified features with neutral defaults.
    """
    # 1) Start with an empty dict for all engineered columns
    row = {}

    # 2) Add engineered date parts (expected by the pipeline)
    pb_year, pb_month = parse_date_to_parts(form.get("policy_bind_date", ""))
    in_year, in_month = parse_date_to_parts(form.get("incident_date", ""))

    date_map = {
        "policy_bind_date_year": pb_year,
        "policy_bind_date_month": pb_month,
        "incident_date_year": in_year,
        "incident_date_month": in_month,
    }

    # 3) Prepare raw inputs from the form (typed)
    typed = {}
    for k, v in form.items():
        if k in NUM_DEFAULTS:
            typed[k] = float(v) if v not in ("", None) else NUM_DEFAULTS[k]
        else:
            typed[k] = v.strip()

    # 4) Fill row values for every engineered column
    # decide numeric vs categorical using our template inference
    num_set = set(num_cols_template)
    cat_set = set(cat_cols_template)

    for col in engineered_cols:
        if col in date_map:
            row[col] = date_map[col]
        elif col in typed:
            # form supplied the value
            if col in num_set:
                try:
                    row[col] = float(typed[col])
                except Exception:
                    row[col] = np.nan
            else:
                row[col] = typed[col]
        else:
            # not collected in the form — fall back on defaults by type
            if col in num_set:
                row[col] = np.nan  # let SimpleImputer(median) handle it
            else:
                row[col] = ""      # let OneHotEncoder(handle_unknown="ignore") handle it

    df_row = pd.DataFrame([row], columns=engineered_cols)
    return df_row

def predict_row(df_row):
    proba = float(pipe.predict_proba(df_row)[0, 1])
    pred = int(proba >= 0.5)
    label = "Fraudulent" if pred == 1 else "Not Fraudulent"
    # simple banding for UX
    if proba < 0.3:
        risk = "Low"
    elif proba < 0.7:
        risk = "Medium"
    else:
        risk = "High"
    return proba, label, risk

def get_feature_importance(df_row, allowed_fields=None):
    """
    Aggregate model coefficients back to original input fields.
    - Groups OneHotEncoder columns by their source feature (e.g., all values of insured_hobbies).
    - Sums absolute coefficients per source feature.
    - Filters to fields the user actually provided (allowed_fields), plus engineered date parts.
    Returns: list[(pretty_feature_name, importance_score)] sorted desc.
    """
    try:
        lr_model = pipe.named_steps['clf']
        preprocessor = pipe.named_steps['pre']

        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = list(preprocessor.get_feature_names_out())
        else:
            feature_names = df_row.columns.tolist()

        coefficients = np.abs(lr_model.coef_[0])

        # Identify original numeric and categorical feature lists from the ColumnTransformer
        original_numeric_features = []
        original_categorical_features = []
        try:
            for name, transformer, columns in preprocessor.transformers_:
                if columns is None:
                    continue
                if name.startswith('num'):
                    original_numeric_features = list(columns)
                elif name.startswith('cat'):
                    original_categorical_features = list(columns)
        except Exception:
            # Fallback: infer from engineered templates
            original_numeric_features = [c for c in df_row.columns if c in set(df_row.select_dtypes(include=[np.number]).columns.tolist())]
            original_categorical_features = [c for c in df_row.columns if c not in set(original_numeric_features)]

        # Allowed base features: those the user provided or date engineered parts
        engineered_date_parts = {
            'policy_bind_date_year': 'Policy Bind Date (Year)',
            'policy_bind_date_month': 'Policy Bind Date (Month)',
            'incident_date_year': 'Incident Date (Year)',
            'incident_date_month': 'Incident Date (Month)'
        }
        allowed_bases = set()
        if allowed_fields is not None:
            allowed_bases.update(set(allowed_fields))
            # If raw date fields were provided, also allow engineered parts
            if 'policy_bind_date' in allowed_bases:
                allowed_bases.update({'policy_bind_date_year', 'policy_bind_date_month'})
            if 'incident_date' in allowed_bases:
                allowed_bases.update({'incident_date_year', 'incident_date_month'})
        else:
            # Fallback to UI-declared fields
            allowed_bases.update(set(FORM_FIELDS))
            allowed_bases.update(engineered_date_parts.keys())

        # Helper to map transformed feature name back to its base/original feature
        def base_from_transformed(name: str) -> str:
            if '__' in name:
                prefix, rest = name.split('__', 1)
            else:
                rest = name
            # Numeric features typically emit names like "num__age"
            for num_col in original_numeric_features:
                if rest == num_col:
                    return num_col
            # Categorical with OHE typically: "cat__insured_hobbies_<category>"
            for cat_col in original_categorical_features:
                marker = f"{cat_col}_"
                if rest.startswith(marker) or rest == cat_col:
                    return cat_col
            # If nothing matched, return the tail as-is
            return rest

        aggregated = defaultdict(float)
        for name, coef in zip(feature_names, coefficients):
            base = base_from_transformed(name)
            aggregated[base] += float(coef)

        # Filter to allowed bases only
        filtered_items = []
        for base, score in aggregated.items():
            if base in allowed_bases:
                # Pretty label for engineered date parts
                pretty = engineered_date_parts.get(base, base)
                filtered_items.append((pretty, score))

        # Sort and take top 10
        filtered_items.sort(key=lambda x: x[1], reverse=True)
        return filtered_items[:10] if filtered_items else []

    except Exception as e:
        print(f"Error extracting feature importance: {e}")
        return []

# -------------------------
# Routes
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index_pipeline.html",
        choices=CHOICES,
        form_fields=FORM_FIELDS,
        result=None,
        state_city_mapping=STATE_CITY_MAPPING,
    )

@app.route("/dashboard", methods=["GET"])
def dashboard():
    """
    Dashboard to show model analytics and verification.
    """
    try:
        # Get model performance metrics
        model_metrics = {
            "model_type": "LogisticRegression Pipeline",
            "features_count": len(engineered_cols),
            "model_loaded": pipe is not None,
            "last_updated": "Model loaded successfully"
        }
        
        # Get sample predictions for verification
        sample_data = df_raw.head(10)
        sample_predictions = []
        
        for _, row in sample_data.iterrows():
            try:
                # Convert row to string format for build_full_row_from_form
                row_dict = {}
                for col, val in row.items():
                    if pd.isna(val):
                        row_dict[col] = ""
                    elif isinstance(val, (int, float)):
                        row_dict[col] = str(val)
                    else:
                        row_dict[col] = str(val)
                
                # Prepare the row for prediction
                sample_row = build_full_row_from_form(row_dict)
                proba, label, risk = predict_row(sample_row)
                sample_predictions.append({
                    "data": row.to_dict(),
                    "prediction": label,
                    "probability": proba,
                    "risk": risk
                })
            except Exception as e:
                print(f"Error predicting sample row: {e}")
                continue
        
        # Calculate prediction counts for charts
        fraudulent_count = sum(1 for pred in sample_predictions if pred['prediction'] == 'Fraudulent')
        not_fraudulent_count = sum(1 for pred in sample_predictions if pred['prediction'] == 'Not Fraudulent')
        
        return render_template(
            "dashboard.html",
            model_metrics=model_metrics,
            sample_predictions=sample_predictions,
            feature_categories={
                "numeric_features": num_cols_template,
                "categorical_features": cat_cols_template,
                "total_features": len(engineered_cols)
            },
            fraudulent_count=fraudulent_count,
            not_fraudulent_count=not_fraudulent_count
        )
        
    except Exception as e:
        return f"Error loading dashboard: {str(e)}", 500

@app.route("/predict", methods=["POST"])
def predict():
    # Support combined City, State selection to avoid invalid pairs
    form_data = request.form.to_dict(flat=True)
    combined = form_data.get("incident_location_combined", "").strip()
    if combined:
        if "," in combined:
            city_part, state_part = combined.rsplit(",", 1)
            form_data["incident_city"] = city_part.strip()
            form_data["incident_state"] = state_part.strip()
        else:
            form_data["incident_city"] = combined
    df_row = build_full_row_from_form(form_data)
    proba, label, risk = predict_row(df_row)
    
    # Determine fields actually provided by the user (non-empty)
    provided_fields = [
        k for k, v in form_data.items()
        if (isinstance(v, str) and v.strip() != "") or (not isinstance(v, str) and v is not None)
    ]

    # Get aggregated and filtered feature importance
    feature_importance = get_feature_importance(df_row, allowed_fields=provided_fields)
    
    return render_template(
        "index_pipeline.html",
        choices=CHOICES,
        form_fields=FORM_FIELDS,
        result={
            "prob": proba,
            "label": label,
            "risk": risk,
            "payload": df_row.to_dict(orient="records")[0],
            "feature_importance": feature_importance,
        },
        state_city_mapping=STATE_CITY_MAPPING,
    )

# JSON API (optional)
@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    # accept either full engineered row or partial with date strings
    class DummyForm(dict): pass
    form_like = DummyForm(payload)
    df_row = build_full_row_from_form(form_like)
    proba, label, risk = predict_row(df_row)
    return jsonify({
        "fraud_probability": proba,
        "prediction": label,
        "risk_band": risk
    })

@app.route("/api/cities/<state>", methods=["GET"])
def get_cities_for_state(state):
    """Get cities for a specific state."""
    if state in STATE_CITY_MAPPING:
        cities = [format_choice_value(city) for city in STATE_CITY_MAPPING[state]]
        return jsonify({"cities": cities})
    else:
        return jsonify({"cities": []})

@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """
    Provide analytics and model information for the frontend.
    This helps verify that the model is being used correctly.
    """
    try:
        # Get model info
        model_info = {
            "model_type": "LogisticRegression Pipeline",
            "features_count": len(engineered_cols),
            "model_loaded": pipe is not None,
            "last_updated": "Model loaded successfully"
        }
        
        # Get sample data for demonstration
        sample_data = df_raw.head(5).to_dict(orient="records")
        
        # Get feature categories
        feature_categories = {
            "numeric_features": num_cols_template,
            "categorical_features": cat_cols_template,
            "total_features": len(engineered_cols)
        }
        
        return jsonify({
            "model_info": model_info,
            "feature_categories": feature_categories,
            "sample_data": sample_data,
            "status": "success"
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

if __name__ == "__main__":
    # For local dev
    app.run(debug=True)
