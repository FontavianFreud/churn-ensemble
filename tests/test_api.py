import numpy as np
from fastapi.testclient import TestClient

import src.api as api


class DummyModel:
    """A tiny stand-in model so tests don't depend on real artifacts."""
    def __init__(self, p: float = 0.20):
        self.p = p

    def predict_proba(self, df):
        n = len(df)
        p1 = np.full(n, self.p, dtype=float)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])


def make_client(p: float = 0.20) -> TestClient:
    # Prevent tests from writing logs to disk
    api._log_event = lambda event: None  # noqa: E731

    # Inject dummy model bundle so /health and /predict don't try to load artifacts
    api._model_bundle = {"model": DummyModel(p=p), "model_version": "test_dummy_v1"}

    return TestClient(api.app)


def test_health_ok():
    client = make_client()
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert "model_version" in body
    assert "threshold" in body


def test_predict_returns_valid_response():
    client = make_client(p=0.20)  # > 0.11 threshold, so pred should be 1
    payload = {
        "record": {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 1,
            "PhoneService": "No",
            "MultipleLines": "No phone service",
            "InternetService": "DSL",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 29.85,
            "TotalCharges": "29.85",
        }
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    assert 0.0 <= body["prob_churn"] <= 1.0
    assert body["pred_churn"] in (0, 1)
    assert body["model_version"] == "test_dummy_v1"

    # With dummy p=0.20 and threshold=0.11, we expect churn predicted
    assert body["pred_churn"] == 1