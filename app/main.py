import logging
import os
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import Depends, FastAPI
import logging
import os
from fastapi import Depends, FastAPI
from dotenv import load_dotenv
from fastapi import FastAPI

from functools import lru_cache
from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
)
from app.services.prediction_service import (
    PredictionService,
)


load_dotenv()


log_level_name = os.getenv(
    "LOG_LEVEL",
    "INFO",
).upper()

log_level = getattr(
    logging,
    log_level_name.upper(),
    logging.INFO,
)


api_version = os.getenv("API_VERSION")

if api_version is None:
    raise ValueError(
        "API_VERSION environment variable tanımlı değil."
    )


logging.basicConfig(
    level=log_level,
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(name)s | "
        "%(message)s"
    ),
)


app = FastAPI(
    title="Customer Churn AI API",
    description=(
        "Customer churn prediction için "
        "geliştirilen Production AI API."
    ),
    version=api_version,
)

@lru_cache
def get_prediction_service() -> PredictionService:
    return PredictionService()


@app.get("/")
def root():
    return {
        "application": "Customer Churn AI API",
        "version": api_version,
        "status": "running",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
)
def predict(
    request: PredictionRequest,
    service: PredictionService = Depends(
        get_prediction_service
    ),
) -> PredictionResponse:
    return service.predict(request)