from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# --------------------------
# LOAD MODEL ARTIFACT
# --------------------------
artifact_path = "pd_24h_model.joblib"
artifact = joblib.load(artifact_path)
model = artifact["model"]
feature_cols = artifact["feature_cols"]

app = FastAPI(
    title="Predictive Maintenance API",
    description="Predicts failure probability within next 24 hours based on engineered features.",
    version="1.0.0",
)


# --------------------------
# REQUEST SCHEMA
# --------------------------
class FeatureVector(BaseModel):
    """
    Generic feature payload.
    In a real system, these would be the engineered rolling features (mean/std/max/...) per sensor.
    For demo, we accept a dict keyed by feature name.
    """
    features: dict


# --------------------------
# HEALTH CHECK
# --------------------------
@app.get("/")
def read_root():
    return {
        "message": "Predictive Maintenance API is running.",
        "model_features_count": len(feature_cols),
    }


# --------------------------
# PREDICTION ENDPOINT
# --------------------------
@app.post("/predict_24h")
def predict_failure_24h(payload: FeatureVector):
    """
    Expects: JSON with a 'features' dict where keys match the model's feature_cols.
    Any missing feature will be filled with 0.0 (for demo purposes).
    """

    # Build ordered feature vector matching training feature_cols
    x = np.array([[float(payload.features.get(col, 0.0)) for col in feature_cols]])

    # Probability of failure (class 1)
    proba = model.predict_proba(x)[0, 1]

    # Simple rule-based recommendation
    if proba < 0.3:
        recommendation = "Low risk: continue normal operation, routine monitoring."
    elif proba < 0.7:
        recommendation = "Moderate risk: schedule inspection in the next maintenance window."
    else:
        recommendation = "High risk: schedule maintenance as soon as possible to avoid unplanned downtime."

    return {
        "failure_probability_24h": round(float(proba), 3),
        "recommendation": recommendation,
    }


# --------------------------
# OPTIONAL: SAMPLE PAYLOAD TEMPLATE
# --------------------------
@app.get("/sample_payload")
def sample_payload():
    """
    Returns a minimal example showing how to call /predict_24h
    with the correct feature names.
    """

    # Take first up to 10 feature names as example
    example_features = {col: 0.0 for col in feature_cols[:10]}

    return {
        "note": "Use these feature keys in the 'features' dict when POSTing to /predict_24h.",
        "features_example": example_features,
    }