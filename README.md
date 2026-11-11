# Customer Churn AI — Production MLOps on Kubernetes

An end-to-end, production-style machine learning service that predicts customer churn risk. The focus of this project is **operating a model in production**, not only training it: experiment tracking and a model registry, a tested and containerized API, CI/CD, Kubernetes deployment with autoscaling and persistent storage, and metrics-based monitoring.

> **Note on data:** the model is trained on a **synthetic dataset** (2,000 customers generated with a fixed random seed and a known churn formula). This keeps the project self-contained and lets the work concentrate on the MLOps lifecycle. The same pipeline accepts a real dataset by replacing the data-generation step in `train_model.py`.

## Highlights

- **MLflow** experiment tracking, **Model Registry** and **alias-based promotion** (`champion` / `challenger`)
- **FastAPI** inference service with Pydantic validation, dependency injection and environment-based configuration
- **Structured JSON logging** with per-request IDs; customer input data is never logged
- **Prometheus metrics** (request count, errors, latency histogram, in-flight requests, predictions by risk level and model version)
- **10 automated tests** (unit + API integration) with model injection, no MLflow server needed
- **GitHub Actions CI/CD**: tests → Docker build → container smoke test → push to Docker Hub
- **Kubernetes**: Deployment with rolling updates and startup/liveness/readiness probes, ConfigMap, Service, **Horizontal Pod Autoscaler**, MLflow server on a **PersistentVolumeClaim**
- **Monitoring** with kube-prometheus-stack (Prometheus + Grafana) and a ServiceMonitor
- **Controlled resource experiments**: OOMKilled (exit code 137), CPU throttling, and storage persistence after pod recreation

## Architecture

```text
                    train_model.py
                          │  params, metrics, model
                          v
                ┌────────────────────┐   alias: champion / challenger
                │  MLflow Tracking + │ <── registry_tools.py
                │   Model Registry   │
                └─────────┬──────────┘
                          │  models:/CustomerChurnModel@champion
                          v
 Client ──> Service ──> FastAPI pods (2–5, HPA) ──> /predict  /health  /ready  /metrics
                          │                                                  │
                          │ JSON logs (stdout)                               │ scrape every 15s
                          v                                                  v
                       kubectl logs                           Prometheus ──> Grafana

 GitHub push ──> GitHub Actions: pytest ──> docker build ──> smoke test ──> Docker Hub
```

## Tech Stack

Python · scikit-learn · MLflow · FastAPI · Pydantic · Uvicorn · pytest · httpx · prometheus-client · Docker · GitHub Actions · Kubernetes · Helm · kube-prometheus-stack (Prometheus, Grafana) · AWS EC2 (earlier Docker deployment)

## Project Structure

```text
customer-churn-ai/
├── app/
│   ├── main.py                    # FastAPI app, middlewares, endpoints
│   ├── logging_config.py          # JSON log formatter + request-id context
│   ├── monitoring/metrics.py      # Prometheus metric definitions
│   ├── schemas/prediction.py      # Pydantic request/response models
│   └── services/prediction_service.py  # model loading (MLflow Registry) + prediction logic
├── k8s/
│   ├── configmap.yaml             # runtime config (MLflow URI, model name, alias)
│   ├── deployment.yaml            # API Deployment, probes, requests/limits
│   ├── service.yaml               # ClusterIP Service
│   ├── hpa.yaml                   # CPU-based autoscaling, 2–5 replicas
│   ├── mlflow-deployment.yaml     # MLflow server inside the cluster
│   ├── mlflow-service.yaml
│   ├── mlflow-pvc.yaml            # persistent storage for MLflow
│   ├── servicemonitor.yaml        # Prometheus scrape config for /metrics
│   ├── load-generator.yaml        # load test for the HPA
│   ├── oom-demo.yaml              # memory-limit experiment
│   ├── cpu-demo.yaml              # CPU-throttling experiment
│   └── storage-demo-pod.yaml      # persistence experiment
├── monitoring/
│   └── kube-prometheus-values.yaml  # Helm values for Prometheus + Grafana
├── tests/
│   ├── test_prediction_service.py
│   └── test_prediction_api.py
├── .github/workflows/ci.yml
├── train_model.py
├── registry_tools.py
├── Dockerfile
├── requirements.txt
└── .env.example
```

## 1. Training, Tracking and Model Registry

`train_model.py` builds a scikit-learn `Pipeline` (StandardScaler + OneHotEncoder in a `ColumnTransformer`, then the classifier) and logs everything to MLflow:

- **Algorithm is selectable** with the `MODEL_ALGORITHM` environment variable: `logreg` (Logistic Regression, default) or `gbm` (Gradient Boosting).
- Parameters, test accuracy / precision / recall / F1, tags and the model artifact are logged to the run.
- Each run registers a new version of `CustomerChurnModel` in the Model Registry.

`registry_tools.py` manages promotion through **aliases** instead of hard-coded version numbers:

```bash
python registry_tools.py list                    # versions, aliases, algorithm, recall, F1
python registry_tools.py set-alias challenger 2  # put a new version under evaluation
python registry_tools.py set-alias champion 2    # promote it to production
```

The API always loads `models:/CustomerChurnModel@champion`, so promoting or rolling back a model means **moving an alias**. No code change or image rebuild is needed.

## 2. Inference API

| Endpoint | Purpose |
|---|---|
| `POST /predict` | Returns `prediction`, `churn_probability`, `risk_level` (low / medium / high) and `model_version` |
| `GET /health` | Liveness: the process is up |
| `GET /ready` | Readiness: the model is loaded; returns the served model version |
| `GET /metrics` | Prometheus metrics (hidden from the OpenAPI docs) |

Design choices:

- **Validation at the edge:** Pydantic rejects negative values, missing fields and unknown contract types before the model is called.
- **Dependency injection:** `PredictionService` is created once (`lru_cache`) and injected with `Depends`. Tests pass a fake model into the same class.
- **Model version in every response:** each prediction can be traced back to a registry version.
- **Privacy-aware logging:** only the prediction result is logged, never the customer input.
- **Optional local artifacts:** with `LOCAL_MODEL_ARTIFACT_ROOT` set, the service resolves the registry version but loads the artifact from inside the container image.

Example request:

```json
POST /predict
{
  "tenure_months": 2,
  "monthly_charges": 95.0,
  "contract_type": "month-to-month",
  "has_internet_service": true,
  "support_calls": 5
}
```

## 3. Observability

**Structured logs:** every log line is JSON written to stdout (`timestamp`, `level`, `logger`, `message`, `request_id`, plus extra fields). The request ID is taken from the `X-Request-ID` header or generated, stored in a `ContextVar`, and returned in the response header, so one request can be followed across log lines. Probe and scrape requests (`/health`, `/ready`, `/metrics`) are logged at DEBUG so they do not flood the logs.

**Prometheus metrics** (from a FastAPI middleware):

| Metric | Type |
|---|---|
| `customer_churn_http_requests_total` | Counter (method, endpoint, status) |
| `customer_churn_http_errors_total` | Counter (method, endpoint, status) |
| `customer_churn_http_request_duration_seconds` | Histogram (method, endpoint) |
| `customer_churn_http_requests_in_progress` | Gauge |
| `customer_churn_predictions_total` | Counter (prediction, risk level, model version) |

In Kubernetes, `kube-prometheus-stack` is installed with Helm, and a `ServiceMonitor` scrapes `/metrics` every 15 seconds. The metrics are then explored in Grafana.

## 4. Testing and CI/CD

`pytest` covers the service layer (high / medium / low risk thresholds) and the API (successful prediction, negative tenure, missing field, invalid contract type, default values, `/` and `/health`). A fake model is injected, so the tests run without an MLflow server.

`.github/workflows/ci.yml` runs on every push and pull request to `main`:

1. **test**: installs dependencies and runs `pytest -v`
2. **build** (only if tests pass): builds the image tagged with the commit SHA
3. **smoke test**: starts the container, waits for `/health`, and checks that `/metrics` exposes `customer_churn_*` metrics; prints container logs on failure
4. **push** (on push to `main`): pushes `:<commit-sha>` and `:latest` to Docker Hub

## 5. Kubernetes Deployment

- **Deployment:** 2 replicas, `RollingUpdate` with `maxUnavailable: 0` (no capacity loss during updates), CPU/memory requests and limits (Burstable QoS).
- **Probes:** a `startupProbe` gives the model time to load; the `livenessProbe` uses `/health`; the `readinessProbe` uses `/ready`, so a pod receives traffic only after the model is loaded.
- **Configuration:** a ConfigMap supplies the MLflow URI, model name and alias.
- **MLflow in the cluster:** a separate Deployment and Service run the MLflow server with a SQLite backend and served artifacts, stored on a 2 Gi **PersistentVolumeClaim**.
- **Autoscaling:** the HPA targets 50% average CPU and scales between 2 and 5 replicas, with a 60-second scale-down stabilization window. `load-generator.yaml` sends traffic from 20 threads. The deployment scaled from 2 to 5 replicas under load and back to 2 after the load stopped.

### Resource experiments

| Manifest | What it shows |
|---|---|
| `oom-demo.yaml` | A process that keeps allocating memory under a 64 Mi limit is killed by the kernel: **OOMKilled, exit code 137** |
| `cpu-demo.yaml` | A busy loop under a 100m CPU limit is **throttled**, visible in the cgroup CPU statistics |
| `storage-demo-pod.yaml` | Data written to the PVC is still there after the pod is deleted and recreated |

## Running the Project

### Locally

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt

cp .env.example .env                 # then set MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME, MODEL_ALIAS
python train_model.py                # train and register a version
python registry_tools.py set-alias champion 1

uvicorn app.main:app --reload        # Swagger UI: http://127.0.0.1:8000/docs
pytest -v
```

### Docker

```bash
docker build -t customer-churn-ai .
docker run -p 8000:8000 --env-file .env customer-churn-ai
```

### Kubernetes

```bash
kubectl create namespace customer-churn-ai

kubectl apply -f k8s/mlflow-pvc.yaml
kubectl apply -f k8s/mlflow-deployment.yaml -f k8s/mlflow-service.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
kubectl apply -f k8s/hpa.yaml               # requires Metrics Server

# Monitoring
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace -f monitoring/kube-prometheus-values.yaml
kubectl apply -f k8s/servicemonitor.yaml

# Load test for the HPA
kubectl apply -f k8s/load-generator.yaml
kubectl get hpa -n customer-churn-ai -w
```

## Limitations and Next Steps

- The training data is synthetic. The next step is to plug in a real banking dataset.
- Model quality is monitored only through prediction counts. Data drift monitoring (for example PSI on input features) is not implemented yet.
- MLflow uses SQLite on a single-replica PVC, which suits a demo but not a multi-user setup (PostgreSQL + object storage would be the production choice).
