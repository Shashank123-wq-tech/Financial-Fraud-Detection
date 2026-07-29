import numpy as np
import pytest
from fastapi.testclient import TestClient

from api import main as main_module
from api.main import app
from src.features import SELECTED_FEATURES


class DummyModel:
    """Stands in for the real XGBoost model in CI, so tests don't
    require the 470MB dataset or a committed .pkl artifact to run."""

    def predict_proba(self, X):
        # Always return a mid-range probability; individual tests can
        # monkeypatch this per-case if they need a specific score.
        return np.array([[0.7, 0.3]] * len(X))


@pytest.fixture
def client():
    main_module.ml_state["model"] = DummyModel()
    main_module.ml_state["feature_columns"] = SELECTED_FEATURES
    with TestClient(app) as c:
        yield c
    main_module.ml_state["model"] = None
    main_module.ml_state["feature_columns"] = None


def test_health_reports_model_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["model_loaded"] is True


def test_health_reports_degraded_when_model_missing(client):
    main_module.ml_state["model"] = None
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "degraded"


def test_predict_returns_valid_response(client):
    payload = {
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
        "transaction_id": "txn-123",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["transaction_id"] == "txn-123"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert body["recommended_action"] in {"approve", "otp", "review", "block"}


def test_predict_rejects_invalid_type(client):
    payload = {
        "type": "PAYMENT",  # not scored — should fail schema validation
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_returns_503_when_model_not_loaded(client):
    main_module.ml_state["model"] = None
    payload = {
        "type": "CASH_OUT",
        "amount": 500.0,
        "oldbalanceOrg": 1000.0,
        "newbalanceOrig": 500.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 500.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 503