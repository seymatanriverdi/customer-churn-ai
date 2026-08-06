import logging
import os

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)


logger = logging.getLogger(__name__)


class PredictionService:

    def __init__(self, model=None):
        # Unit ve integration testlerde fake model dışarıdan verilir.
        if model is not None:
            self.model = model

            self.model_version = os.getenv(
                "MODEL_VERSION",
                "test-version",
            )

            logger.info(
                "Injected model is being used. "
                "Model version: %s",
                self.model_version,
            )

            return

        # Normal uygulamada model MLflow Registry'den yüklenir.
        tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI"
        )

        if tracking_uri is None:
            raise ValueError(
                "MLFLOW_TRACKING_URI environment variable "
                "tanımlı değil."
            )

        registered_model_name = os.getenv(
            "REGISTERED_MODEL_NAME"
        )

        if registered_model_name is None:
            raise ValueError(
                "REGISTERED_MODEL_NAME environment variable "
                "tanımlı değil."
            )

        model_alias = os.getenv(
            "MODEL_ALIAS"
        )

        if model_alias is None:
            raise ValueError(
                "MODEL_ALIAS environment variable "
                "tanımlı değil."
            )

        # FastAPI ve training script aynı MLflow store'u kullanır.
        mlflow.set_tracking_uri(
            tracking_uri
        )

        model_uri = (
            f"models:/{registered_model_name}"
            f"@{model_alias}"
        )

        logger.info(
            "Loading model from MLflow Registry: %s",
            model_uri,
        )

        mlflow_client = MlflowClient()

        # Alias'ın hangi gerçek model versiyonuna bağlı
        # olduğunu Registry'den öğreniyoruz.
        model_version_info = (
            mlflow_client.get_model_version_by_alias(
                name=registered_model_name,
                alias=model_alias,
            )
        )

        self.model_version = str(
            model_version_info.version
        )

        # Champion alias'ına bağlı sklearn pipeline'ını yükler.
        self.model = mlflow.sklearn.load_model(
            model_uri
        )

        logger.info(
            "Model loaded successfully from MLflow Registry. "
            "Model name: %s | Alias: %s | Version: %s",
            registered_model_name,
            model_alias,
            self.model_version,
        )

    def predict(
        self,
        request: PredictionRequest,
    ) -> PredictionResponse:

        logger.info(
            "Prediction request received."
        )

        customer_data = pd.DataFrame(
            [
                {
                    "tenure_months": (
                        request.tenure_months
                    ),
                    "monthly_charges": (
                        request.monthly_charges
                    ),
                    "contract_type": (
                        request.contract_type
                    ),
                    "has_internet_service": (
                        request.has_internet_service
                    ),
                    "support_calls": (
                        request.support_calls
                    ),
                }
            ]
        )

        probabilities = self.model.predict_proba(
            customer_data
        )

        churn_probability = float(
            probabilities[0][1]
        )

        prediction = (
            1
            if churn_probability >= 0.50
            else 0
        )

        if churn_probability >= 0.70:
            risk_level = "high"
        elif churn_probability >= 0.40:
            risk_level = "medium"
        else:
            risk_level = "low"

        logger.info(
            "Risk Level: %s",
            risk_level,
        )

        logger.info(
            "Probability: %.4f",
            churn_probability,
        )

        logger.info(
            "Prediction completed successfully. "
            "Model version: %s",
            self.model_version,
        )

        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(
                churn_probability,
                4,
            ),
            risk_level=risk_level,
            model_version=self.model_version,
        )