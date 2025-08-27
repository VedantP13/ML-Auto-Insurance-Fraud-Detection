from flask import Flask, request, render_template
import pandas as pd
import joblib
from pathlib import Path

# project root (assumes file is in src/ or app/)
BASE_DIR = Path(__file__).resolve().parent.parent   # parent of src/ or app/ is project root
DATA_PATH = BASE_DIR / "data" / "insurance_claims.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
pipe = joblib.load(MODELS_DIR / "pipeline.pkl")

# Infer input schema from pipeline's preprocessor columns if available
def get_input_columns():
    try:
        cols = joblib.load(MODELS_DIR / "original_feature_columns.pkl")
        return cols
    except Exception:
        # fallback: request JSON input
        return None

@app.route("/", methods=["GET"])
def home():
    cols = get_input_columns()
    if cols is None:
        return "<h2>Pipeline API ready.</h2><p>POST JSON to /predict with your raw feature columns.</p>"
    return render_template("index_pipeline.html", feature_names=cols)

@app.route("/predict", methods=["POST"])
def predict():
    cols = get_input_columns()
    if cols is None:
        # Accept JSON
        data = request.get_json(force=True)
        df = pd.DataFrame([data])
    else:
        row = {c: request.form.get(c, "") for c in cols}
        df = pd.DataFrame([row])
    proba = pipe.predict_proba(df)[0, 1]
    pred = int(proba >= 0.5)
    label = "Fraud" if pred == 1 else "Not Fraud"
    return render_template(
        "index_pipeline.html",
        feature_names=cols,
        prediction_text=f"Prediction: {label} (p={proba:.3f})"
    )

if __name__ == "__main__":
    app.run(debug=True)
