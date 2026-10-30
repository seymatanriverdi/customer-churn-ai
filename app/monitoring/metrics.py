from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
)


HTTP_REQUESTS_TOTAL = Counter(
    "customer_churn_http_requests_total",
    "Total number of HTTP requests",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)


HTTP_ERRORS_TOTAL = Counter(
    "customer_churn_http_errors_total",
    "Total number of HTTP errors",
    [
        "method",
        "endpoint",
        "status_code",
    ],
)


HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "customer_churn_http_request_duration_seconds",
    "HTTP request duration in seconds",
    [
        "method",
        "endpoint",
    ],
    buckets=(
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
    ),
)


HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "customer_churn_http_requests_in_progress",
    "Number of HTTP requests currently being processed",
    [
        "method",
        "endpoint",
    ],
)


PREDICTIONS_TOTAL = Counter(
    "customer_churn_predictions_total",
    "Total number of churn predictions",
    [
        "prediction",
        "risk_level",
        "model_version",
    ],
)