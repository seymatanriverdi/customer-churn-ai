import logging
import os
import time
import uuid
from functools import lru_cache

from dotenv import load_dotenv
from fastapi import (
    Depends,
    FastAPI,
    Request,
    Response,
)
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    generate_latest,
)

from app.logging_config import (
    request_id_var,
    setup_logging,
)
from app.monitoring.metrics import (
    HTTP_ERRORS_TOTAL,
    HTTP_REQUEST_DURATION_SECONDS,
    HTTP_REQUESTS_IN_PROGRESS,
    HTTP_REQUESTS_TOTAL,
    PREDICTIONS_TOTAL,
)
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

api_version = os.getenv(
    "API_VERSION",
    "1.0.0",
)


# Tum loglar JSON formatinda stdout'a yazilir.
setup_logging(log_level_name)

logger = logging.getLogger(__name__)


app = FastAPI(
    title="Customer Churn AI API",
    description=(
        "Customer churn prediction icin "
        "gelistirilen Production AI API."
    ),
    version=api_version,
)


@app.middleware("http")
async def prometheus_metrics_middleware(
    request: Request,
    call_next,
):
    method = request.method
    endpoint = request.url.path

    # Prometheus'un kendi scrape isteklerini
    # uygulama request metriklerine dahil etmiyoruz.
    if endpoint == "/metrics":
        return await call_next(request)

    start_time = time.perf_counter()
    status_code = "500"

    HTTP_REQUESTS_IN_PROGRESS.labels(
        method=method,
        endpoint=endpoint,
    ).inc()

    try:
        response = await call_next(request)
        status_code = str(response.status_code)

        if response.status_code >= 400:
            HTTP_ERRORS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

        return response

    except Exception:
        HTTP_ERRORS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code="500",
        ).inc()
        raise

    finally:
        duration_seconds = (
            time.perf_counter() - start_time
        )

        HTTP_REQUESTS_TOTAL.labels(
            method=method,
            endpoint=endpoint,
            status_code=status_code,
        ).inc()

        HTTP_REQUEST_DURATION_SECONDS.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration_seconds)

        HTTP_REQUESTS_IN_PROGRESS.labels(
            method=method,
            endpoint=endpoint,
        ).dec()


# Probe ve scrape istekleri cok sik gelir; INFO seviyesinde
# loglanirsa gercek istekleri bogar. Bunlar DEBUG'a iner.
NOISY_PATHS = {"/health", "/ready", "/metrics"}


@app.middleware("http")
async def request_logging_middleware(
    request: Request,
    call_next,
):
    # Istemci X-Request-ID gonderdiyse onu kullan,
    # yoksa yeni bir kimlik uret.
    request_id = (
        request.headers.get("X-Request-ID")
        or uuid.uuid4().hex[:12]
    )
    token = request_id_var.set(request_id)

    path = request.url.path
    level = (
        logging.DEBUG
        if path in NOISY_PATHS
        else logging.INFO
    )
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

    except Exception:
        logger.exception(
            "request failed",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": path,
                    "status_code": 500,
                    "duration_ms": round(
                        (time.perf_counter() - start_time)
                        * 1000,
                        2,
                    ),
                }
            },
        )
        raise

    else:
        logger.log(
            level,
            "request completed",
            extra={
                "extra_fields": {
                    "method": request.method,
                    "path": path,
                    "status_code": response.status_code,
                    "duration_ms": round(
                        (time.perf_counter() - start_time)
                        * 1000,
                        2,
                    ),
                }
            },
        )
        response.headers["X-Request-ID"] = request_id
        return response

    finally:
        request_id_var.reset(token)


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


@app.get("/ready")
def readiness_check(
    service: PredictionService = Depends(
        get_prediction_service
    ),
):
    return {
        "status": "ready",
        "model_version": service.model_version,
    }


@app.get(
    "/metrics",
    include_in_schema=False,
)
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST,
    )


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
    result = service.predict(request)

    PREDICTIONS_TOTAL.labels(
        prediction=str(result.prediction),
        risk_level=result.risk_level,
        model_version=str(result.model_version),
    ).inc()

    # Musteri verisini (girdi) loglamiyoruz, sadece sonucu.
    logger.info(
        "prediction completed",
        extra={
            "extra_fields": {
                "prediction": str(result.prediction),
                "risk_level": result.risk_level,
                "model_version": str(result.model_version),
            }
        },
    )

    return result