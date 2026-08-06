from typing import Literal

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    tenure_months: int = Field(
        ...,
        ge=0,
        description="Müşterinin şirkette kaldığı toplam ay sayısı",
        examples=[12],
    )

    monthly_charges: float = Field(
        ...,
        ge=0,
        description="Müşterinin aylık ödeme tutarı",
        examples=[89.90],
    )

    contract_type: Literal[
        "month-to-month",
        "one-year",
        "two-year",
    ] = Field(
        ...,
        description="Müşterinin sözleşme türü",
        examples=["month-to-month"],
    )

    has_internet_service: bool = Field(
        ...,
        description="Müşterinin internet hizmeti kullanıp kullanmadığı",
        examples=[True],
    )

    support_calls: int = Field(
        default=0,
        ge=0,
        description="Müşterinin destek hattını arama sayısı",
        examples=[3],
    )




class PredictionResponse(BaseModel):
    prediction: int = Field(
        ...,
        description="Modelin tahmin sınıfı: 0 müşterinin kalması, 1 ayrılması",
        examples=[1],
    )

    churn_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Müşterinin ayrılma olasılığı",
        examples=[0.82],
    )

    risk_level: Literal["low", "medium", "high"] = Field(
        ...,
        description="Müşterinin churn risk seviyesi",
        examples=["high"],
    )

    model_version: str = Field(
        ...,
        description="Tahminde kullanılan model sürümü",
        examples=["1.0.0"],
    )