from pathlib import Path
import mlflow.sklearn
import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


# Sonuçların her çalıştırmada aynı olması için.
random_generator = np.random.default_rng(seed=42)

number_of_customers = 2_000


# Örnek müşteri verileri oluşturuyoruz.
data = pd.DataFrame(
    {
        "tenure_months": random_generator.integers(
            low=0,
            high=73,
            size=number_of_customers,
        ),
        "monthly_charges": random_generator.uniform(
            low=20,
            high=150,
            size=number_of_customers,
        ).round(2),
        "contract_type": random_generator.choice(
            [
                "month-to-month",
                "one-year",
                "two-year",
            ],
            size=number_of_customers,
            p=[0.55, 0.25, 0.20],
        ),
        "has_internet_service": random_generator.choice(
            [True, False],
            size=number_of_customers,
            p=[0.80, 0.20],
        ),
        "support_calls": random_generator.integers(
            low=0,
            high=11,
            size=number_of_customers,
        ),
    }
)


# Sentetik churn ihtimali oluşturuyoruz.
risk_score = (
    -0.05 * data["tenure_months"]
    + 0.025 * data["monthly_charges"]
    + 0.45 * data["support_calls"]
    + 1.80
    * (
        data["contract_type"]
        == "month-to-month"
    ).astype(int)
    - 1.20
    * (
        data["contract_type"]
        == "two-year"
    ).astype(int)
    + 0.50
    * data["has_internet_service"].astype(int)
    - 2.50
)


churn_probability = 1 / (
    1 + np.exp(-risk_score)
)


data["churn"] = random_generator.binomial(
    n=1,
    p=churn_probability,
)


features = data[
    [
        "tenure_months",
        "monthly_charges",
        "contract_type",
        "has_internet_service",
        "support_calls",
    ]
]

target = data["churn"]


# Veriyi eğitim ve test olarak ayırıyoruz.
X_train, X_test, y_train, y_test = (
    train_test_split(
        features,
        target,
        test_size=0.20,
        random_state=42,
        stratify=target,
    )
)


numeric_features = [
    "tenure_months",
    "monthly_charges",
    "support_calls",
]

categorical_features = [
    "contract_type",
    "has_internet_service",
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            StandardScaler(),
            numeric_features,
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
            ),
            categorical_features,
        ),
    ]
)


model_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor,
        ),
        (
            "model",
            LogisticRegression(
                max_iter=1_000,
                random_state=42,
            ),
        ),
    ]
)


models_directory = Path("models")

models_directory.mkdir(
    parents=True,
    exist_ok=True,
)

model_path = (
    models_directory
    / "customer_churn_model.joblib"
)


# MLflow experiment'ını seçiyoruz.
mlflow.set_experiment(
    "Customer Churn"
)


# Her çalıştırmada yeni bir MLflow Run oluşur.
with mlflow.start_run(
    run_name="Logistic Regression Baseline"
):

    # Parametreleri MLflow'a kaydediyoruz.
    mlflow.log_param(
        "algorithm",
        "Logistic Regression",
    )

    mlflow.log_param(
        "number_of_customers",
        number_of_customers,
    )

    mlflow.log_param(
        "test_size",
        0.20,
    )

    mlflow.log_param(
        "random_state",
        42,
    )

    mlflow.log_param(
        "max_iter",
        1_000,
    )

    # Modeli yalnızca training verisiyle eğitiyoruz.
    model_pipeline.fit(
        X_train,
        y_train,
    )

    # Test verisi için tahmin üretiyoruz.
    predictions = model_pipeline.predict(
        X_test
    )

    # Metrikleri hesaplıyoruz.
    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # Metrikleri MLflow'a kaydediyoruz.
    mlflow.log_metric(
        "test_accuracy",
        accuracy,
    )

    mlflow.log_metric(
        "test_precision",
        precision,
    )

    mlflow.log_metric(
        "test_recall",
        recall,
    )

    mlflow.log_metric(
        "test_f1",
        f1,
    )

    # Model pipeline'ını dosyaya kaydediyoruz.
    joblib.dump(
        model_pipeline,
        model_path,
    )

    # Model dosyasını MLflow artifact'ı olarak ekliyoruz.
    mlflow.log_artifact(
        str(model_path),
        artifact_path="model",
    )

    registered_model_name = "CustomerChurnModel"

    input_example = X_train.head(3)

    model_info = mlflow.sklearn.log_model(
    sk_model=model_pipeline,
    name="customer-churn-model",
    input_example=input_example,
    registered_model_name=registered_model_name,
)

    print(
        "Registry model URI:",
        model_info.model_uri,
    )


    # Run hakkında ek bilgiler.
    mlflow.set_tag(
        "project",
        "Customer Churn AI",
    )

    mlflow.set_tag(
        "model_type",
        "binary-classification",
    )

    mlflow.set_tag(
        "dataset_type",
        "synthetic",
    )


print("Model başarıyla eğitildi.")
print(f"Model yolu: {model_path}")
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test F1 score: {f1:.4f}")