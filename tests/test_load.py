# tests/test_load.py
from pathlib import Path
import joblib

MODELS_DIR = Path("models")
p = MODELS_DIR / "pipeline.pkl"

print("pipeline exists:", p.exists())
if p.exists():
    pipe = joblib.load(p)
    print("Loaded pipeline. Has predict_proba:", hasattr(pipe, "predict_proba"))
else:
    print("pipeline.pkl not found — did you run src/train_model_pipeline.py?")
