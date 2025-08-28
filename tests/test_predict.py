import pandas as pd
import joblib
from pathlib import Path

# -------------------------
# Paths
# -------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# -------------------------
# Load pipeline + features
# -------------------------
pipe = joblib.load(MODELS_DIR / "pipeline.pkl")
cols = joblib.load(MODELS_DIR / "engineered_feature_columns.pkl")
print(f"Loaded engineered feature columns: {len(cols)}")

# -------------------------
# Example input
# -------------------------
# Fill with a realistic single-row dictionary
sample = {
    "months_as_customer": 24,
    "age": 45,
    "policy_state": "OH",
    "policy_csl": "250/500",
    "policy_deductable": 500,
    "policy_annual_premium": 1200,
    "umbrella_limit": 0,
    "insured_sex": "MALE",
    "insured_education_level": "Bachelor",
    "insured_occupation": "engineer",
    "insured_hobbies": "reading",
    "insured_relationship": "husband",
    "capital-gains": 0,
    "capital-loss": 0,
    "incident_type": "Single Vehicle Collision",
    "collision_type": "Rear Collision",
    "incident_severity": "Major Damage",
    "authorities_contacted": "Police",
    "incident_state": "OH",
    "incident_city": "Columbus",
    "incident_hour_of_the_day": 14,
    "number_of_vehicles_involved": 1,
    "property_damage": "YES",
    "bodily_injuries": 1,
    "witnesses": 2,
    "police_report_available": "YES",
    "total_claim_amount": 12000,
    "injury_claim": 3000,
    "property_claim": 5000,
    "vehicle_claim": 4000,
    "auto_make": "Toyota",
    "auto_model": "Camry",
    "auto_year": 2015,
    "policy_bind_date_year": 2015,
    "policy_bind_date_month": 6,
    "incident_date_year": 2017,
    "incident_date_month": 8
}

df = pd.DataFrame([sample], columns=cols)
print(f"Input shape: {df.shape}")
print("Missing in DF:", set(cols) - set(df.columns))

# -------------------------
# Predict
# -------------------------
proba = pipe.predict_proba(df)[0, 1]
pred = "FRAUDULENT" if proba >= 0.5 else "NOT FRAUDULENT"

print(f"Fraud probability: {proba:.4f}")
print(f"Prediction: {pred}")
