from flask import Flask, request, render_template
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Flask app
app = Flask(__name__)

# Model detection
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
DT_MODEL_PATH = MODELS_DIR / "dt_model.pkl"
DT_SCALER_PATH = MODELS_DIR / "dt_scaler.pkl"
DT_FEATURES_PATH = MODELS_DIR / "dt_feature_names.pkl"

model_type = None
pipe = None
model = None
scaler = None
feature_names = None

if PIPELINE_PATH.exists():
    # Logistic Regression full pipeline
    pipe = joblib.load(PIPELINE_PATH)
    model_type = "pipeline"
    print("Loaded Logistic Regression pipeline model.")
elif DT_MODEL_PATH.exists() and DT_SCALER_PATH.exists() and DT_FEATURES_PATH.exists():
    # Decision Tree + Scaler + Feature Names
    model = joblib.load(DT_MODEL_PATH)
    scaler = joblib.load(DT_SCALER_PATH)
    feature_names = joblib.load(DT_FEATURES_PATH)
    model_type = "decision_tree"
    print("Loaded Decision Tree model with scaler + feature names.")
else:
    raise RuntimeError("No valid model files found in models/")

@app.route("/")
def home():
    if model_type == "pipeline":
        return render_template("index.html", feature_names=None,
                               message="Pipeline model loaded. Fill the form or send JSON.")
    else:
        return render_template("index.html", feature_names=feature_names,
                               message="Decision Tree model loaded. Fill the form.")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if model_type == "pipeline":
            # JSON or form input
            data = request.form.to_dict() or request.get_json(force=True)
            df = pd.DataFrame([data])
            proba = pipe.predict_proba(df)[0, 1]
            pred = int(proba >= 0.5)
            label = "Fraud" if pred == 1 else "Not Fraud"
            return render_template("index.html",
                                   prediction_text=f"Prediction: {label} (p={proba:.3f})",
                                   feature_names=None)
        else:
            # Decision Tree
            input_data = {feature: float(request.form[feature]) for feature in feature_names}
            df = pd.DataFrame([input_data])[feature_names]
            X_scaled = scaler.transform(df)
            pred = model.predict(X_scaled)[0]
            label = "Fraud" if pred == 1 else "Not Fraud"
            return render_template("index.html",
                                   prediction_text=f"Prediction: {label}",
                                   feature_names=feature_names)
    except Exception as e:
        return render_template("index.html",
                               prediction_text=f"Error: {str(e)}",
                               feature_names=feature_names)

if __name__ == "__main__":
    app.run(debug=True)
