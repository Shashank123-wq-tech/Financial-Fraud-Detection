import json
import logging
import os
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from api.risk_engine import THRESHOLDS, is_fraud_decision, recommend_action
from api.schemas import HealthResponse, PredictionResponse, Transaction
from src.features import SELECTED_FEATURES, engineer_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

MODEL_PATH = os.getenv("MODEL_PATH", "models/xgboost_fraud_model.pkl")
FEATURE_COLUMNS_PATH = os.getenv("FEATURE_COLUMNS_PATH", "models/feature_columns.json")
# Set by CI/CD to the Git commit SHA the image was built from, so every
# prediction can be traced back to exact source code (see ci-cd.yml).
MODEL_VERSION = os.getenv("MODEL_VERSION", "dev-local")

ml_state = {"model": None, "feature_columns": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load once at container startup, not per-request — avoids repeated
    # disk I/O and deserialization on every prediction.
    try:
        ml_state["model"] = joblib.load(MODEL_PATH)
        with open(FEATURE_COLUMNS_PATH) as f:
            ml_state["feature_columns"] = json.load(f)

        if ml_state["feature_columns"] != SELECTED_FEATURES:
            # Hard failure on purpose: a mismatch here means the deployed
            # artifact and the deployed feature-engineering code have
            # drifted apart, which silently corrupts every prediction.
            raise RuntimeError(
                "feature_columns.json does not match SELECTED_FEATURES in "
                "src/features.py — model artifact and code are out of sync."
            )
        logger.info("Model and feature columns loaded from %s", MODEL_PATH)
    except Exception:
        logger.exception("Failed to load model at startup")
        ml_state["model"] = None
    yield
    ml_state.clear()


app = FastAPI(
    title="Fraud Detection API",
    description="Scores TRANSFER and CASH_OUT transactions for fraud before funds leave the system.",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Used by the ECS/ALB health check. Returns 200 with model_loaded
    reflecting real state — never fakes a healthy model."""
    loaded = ml_state["model"] is not None
    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=MODEL_VERSION,
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    model = ml_state["model"]
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Check /health.",
        )

    raw = pd.DataFrame([transaction.model_dump(exclude={"transaction_id", "nameOrig", "nameDest"})])

    try:
        X = engineer_features(raw)
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Missing field required for feature engineering: {e}",
        )

    probability = float(model.predict_proba(X)[:, 1][0])
    action = recommend_action(probability)

    logger.info(
        "prediction transaction_id=%s type=%s amount=%s probability=%.4f action=%s",
        transaction.transaction_id, transaction.type, transaction.amount, probability, action,
    )

    return PredictionResponse(
        transaction_id=transaction.transaction_id,
        fraud_probability=probability,
        is_fraud=is_fraud_decision(probability),
        recommended_action=action,
        threshold_used=THRESHOLDS.review,
        model_version=MODEL_VERSION,
    )