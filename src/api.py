from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field


ARTIFACT_PATH = Path("artifacts/week4_candidate_calibrated_soft_platt.joblib")
LOG_DIR = Path("logs")
LOG_PATH = LOG_DIR / "predictions.jsonl"

# From Week 4: cost-sensitive threshold (FP=1, FN=5) chosen on validation
DECISION_THRESHOLD = 0.11

app = FastAPI(title="Churn Risk API", version="1.0.0")

_model_bundle = None  # cached {"model": ..., "model_version": ...}


class CustomerRecord(BaseModel):
    gender: str
    SeniorCitizen: int = Field(ge=0, le=1)
    Partner: str
    Dependents: str
    tenure: int = Field(ge=0)
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: str | float  # accept either, we'll coerce to numeric


class PredictRequest(BaseModel):
    record: CustomerRecord


class PredictResponse(BaseModel):
    prob_churn: float
    pred_churn: int
    threshold: float
    confidence: float
    latency_ms: float
    model_version: str


def _ensure_model_loaded() -> dict:
    global _model_bundle
    if _model_bundle is not None:
        return _model_bundle

    if not ARTIFACT_PATH.exists():
        # Build artifact on-demand for local convenience
        from src.build_week4_model import main as build_model
        build_model()

    _model_bundle = joblib.load(ARTIFACT_PATH)
    return _model_bundle


def _log_event(event: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(event) + "\n")


@app.get("/health")
def health() -> dict:
    bundle = _ensure_model_loaded()
    return {
        "status": "ok",
        "model_version": bundle.get("model_version", "unknown"),
        "threshold": DECISION_THRESHOLD,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, request: Request) -> PredictResponse:
    t0 = time.perf_counter()
    bundle = _ensure_model_loaded()
    model = bundle["model"]
    model_version = bundle.get("model_version", "unknown")

    # Convert request into a 1-row DataFrame
    try:
        row = req.record.model_dump()
        df = pd.DataFrame([row])

        # Coerce numeric types exactly how training expected
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["MonthlyCharges"] = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        df["tenure"] = pd.to_numeric(df["tenure"], errors="coerce")
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce")
    except Exception as e:
        _log_event({
            "ts": datetime.now(timezone.utc).isoformat(),
            "schema_valid": False,
            "error": str(e),
            "path": str(request.url.path),
            "model_version": model_version,
        })
        raise HTTPException(status_code=400, detail="Invalid input record")

    prob = float(model.predict_proba(df)[:, 1][0])
    pred = int(prob >= DECISION_THRESHOLD)
    confidence = float(max(prob, 1 - prob))
    latency_ms = (time.perf_counter() - t0) * 1000.0

    _log_event({
        "ts": datetime.now(timezone.utc).isoformat(),
        "schema_valid": True,
        "prob_churn": prob,
        "pred_churn": pred,
        "confidence": confidence,
        "threshold": DECISION_THRESHOLD,
        "latency_ms": latency_ms,
        "model_version": model_version,
    })

    return PredictResponse(
        prob_churn=prob,
        pred_churn=pred,
        threshold=float(DECISION_THRESHOLD),
        confidence=confidence,
        latency_ms=float(latency_ms),
        model_version=str(model_version),
    )