import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import (
    app,
    get_prediction_service,
)
from app.services.prediction_service import (
    PredictionService,
)


class FakeHighRiskModel:

    def predict_proba(self, customer_data):
        return np.array(
            [
                [0.10, 0.90],
            ]
        )


@pytest.fixture
def client(monkeypatch):
    """
    Her test için fake model kullanan
    bir FastAPI TestClient oluşturur.
    """

    monkeypatch.setenv(
        "MODEL_VERSION",
        "integration-test-1.0.0",
    )

    fake_service = PredictionService(
        model=FakeHighRiskModel(),
    )

    app.dependency_overrides[
        get_prediction_service
    ] = lambda: fake_service

    test_client = TestClient(app)

    yield test_client

    app.dependency_overrides.clear()


def test_predict_endpoint_returns_success(client):
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 89.90,
            "contract_type": "month-to-month",
            "has_internet_service": True,
            "support_calls": 3,
        },
    )

    assert response.status_code == 200

    response_body = response.json()

    assert response_body["prediction"] == 1
    assert response_body["churn_probability"] == 0.90
    assert response_body["risk_level"] == "high"
    assert (
        response_body["model_version"]
        == "integration-test-1.0.0"
    )


def test_predict_endpoint_rejects_negative_tenure(
    client,
):
    response = client.post(
        "/predict",
        json={
            "tenure_months": -5,
            "monthly_charges": 89.90,
            "contract_type": "month-to-month",
            "has_internet_service": True,
            "support_calls": 3,
        },
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_missing_field(
    client,
):
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "contract_type": "month-to-month",
            "has_internet_service": True,
            "support_calls": 3,
        },
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_contract(
    client,
):
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 89.90,
            "contract_type": "monthly",
            "has_internet_service": True,
            "support_calls": 3,
        },
    )

    assert response.status_code == 422


def test_predict_endpoint_uses_default_support_calls(
    client,
):
    response = client.post(
        "/predict",
        json={
            "tenure_months": 12,
            "monthly_charges": 89.90,
            "contract_type": "month-to-month",
            "has_internet_service": True,
        },
    )

    assert response.status_code == 200
    assert response.json()["risk_level"] == "high"


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "application": "Customer Churn AI API",
        "version": "1.0.0",
        "status": "running",
    }


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }