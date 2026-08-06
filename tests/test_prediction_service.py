import numpy as np

from app.schemas.prediction import PredictionRequest
from app.services.prediction_service import PredictionService


class FakeHighRiskModel:

    def predict_proba(self, customer_data):
        return np.array(
            [
                [0.10, 0.90],
            ]
        )


class FakeMediumRiskModel:

    def predict_proba(self, customer_data):
        return np.array(
            [
                [0.45, 0.55],
            ]
        )


class FakeLowRiskModel:

    def predict_proba(self, customer_data):
        return np.array(
            [
                [0.80, 0.20],
            ]
        )


def create_request() -> PredictionRequest:
    return PredictionRequest(
        tenure_months=12,
        monthly_charges=89.90,
        contract_type="month-to-month",
        has_internet_service=True,
        support_calls=3,
    )


def test_predict_returns_high_risk(
    monkeypatch,
):
    monkeypatch.setenv(
        "MODEL_VERSION",
        "test-1.0.0",
    )

    service = PredictionService(
        model=FakeHighRiskModel(),
    )

    response = service.predict(
        create_request()
    )

    assert response.prediction == 1
    assert response.churn_probability == 0.90
    assert response.risk_level == "high"
    assert response.model_version == "test-1.0.0"


def test_predict_returns_medium_risk(
    monkeypatch,
):
    monkeypatch.setenv(
        "MODEL_VERSION",
        "test-1.0.0",
    )

    service = PredictionService(
        model=FakeMediumRiskModel(),
    )

    response = service.predict(
        create_request()
    )

    assert response.prediction == 1
    assert response.churn_probability == 0.55
    assert response.risk_level == "medium"


def test_predict_returns_low_risk(
    monkeypatch,
):
    monkeypatch.setenv(
        "MODEL_VERSION",
        "test-1.0.0",
    )

    service = PredictionService(
        model=FakeLowRiskModel(),
    )

    response = service.predict(
        create_request()
    )

    assert response.prediction == 0
    assert response.churn_probability == 0.20
    assert response.risk_level == "low"
    assert response.model_version == "test-1.0.0"