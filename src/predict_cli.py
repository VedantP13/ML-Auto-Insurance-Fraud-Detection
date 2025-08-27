import argparse
import pandas as pd
import joblib
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent   # project root
MODELS_DIR = BASE_DIR / "models"

PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
DT_MODEL_PATH = MODELS_DIR / "dt_model.pkl"
DT_SCALER_PATH = MODELS_DIR / "dt_scaler.pkl"
DT_FEATURES_PATH = MODELS_DIR / "dt_feature_names.pkl"


def main():
    parser = argparse.ArgumentParser(description="Fraud detection on CSV input")
    parser.add_argument("csv", help="Path to CSV input file")
    parser.add_argument("--out", default="predictions.csv", help="Path to save predictions")
    args = parser.parse_args()

    # Detect model type
    if PIPELINE_PATH.exists():
        print("Using Logistic Regression pipeline...")
        pipe = joblib.load(PIPELINE_PATH)

        df = pd.read_csv(args.csv)
        preds = pipe.predict(df)
        proba = pipe.predict_proba(df)[:, 1]
        output = pd.DataFrame({
            "prediction": preds,
            "probability": proba
        })
    elif DT_MODEL_PATH.exists() and DT_SCALER_PATH.exists() and DT_FEATURES_PATH.exists():
        print("Using Decision Tree model...")
        model = joblib.load(DT_MODEL_PATH)
        scaler = joblib.load(DT_SCALER_PATH)
        feature_names = joblib.load(DT_FEATURES_PATH)

        df = pd.read_csv(args.csv)
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise SystemExit(f"ERROR: Missing columns in CSV: {missing}")

        df = df[feature_names]
        X_scaled = scaler.transform(df)
        preds = model.predict(X_scaled)
        output = pd.DataFrame({"prediction": preds})
    else:
        raise RuntimeError("No valid model found in models/")

    output.to_csv(args.out, index=False)
    print(f"✅ Predictions saved to {args.out}")


if __name__ == "__main__":
    main()
