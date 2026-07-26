from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Callable

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from src.predict import AnomalyPredictor


class SensorInput(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "temperature": 30.0,
                "vibration": 0.35,
                "pressure": 100.0,
                "humidity": 45.0,
            }
        }
    )

    temperature: float = Field(ge=-100, le=300)
    vibration: float = Field(ge=0, le=100)
    pressure: float = Field(ge=0, le=1000)
    humidity: float = Field(ge=0, le=100)


class PredictionResponse(BaseModel):
    prediction: str
    reconstruction_error: float
    threshold: float
    error_margin: float
    feature_errors: dict[str, float]
    model_version: str
    input: dict[str, float]


def create_app(
    predictor_factory: Callable[[], AnomalyPredictor] = AnomalyPredictor,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.predictor = predictor_factory()
        yield

    application = FastAPI(
        title="Sensor Anomaly Detection API",
        description=(
            "Normal-only PyTorch autoencoder inference with a checkpoint-bound "
            "validation threshold."
        ),
        version="2.0.0",
        lifespan=lifespan,
    )

    @application.get("/")
    def root(request: Request) -> dict[str, str]:
        predictor: AnomalyPredictor = request.app.state.predictor
        return {
            "service": "Sensor Anomaly Detection API",
            "model_version": predictor.model_version,
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs",
        }

    @application.get("/health")
    def health(request: Request) -> dict[str, object]:
        predictor: AnomalyPredictor = request.app.state.predictor
        return {
            "status": "ok",
            "model_loaded": True,
            "model_version": predictor.model_version,
            "threshold": round(predictor.threshold, 6),
        }

    @application.post("/predict", response_model=PredictionResponse)
    def predict(input_data: SensorInput, request: Request) -> dict[str, object]:
        predictor: AnomalyPredictor = request.app.state.predictor
        return predictor.predict(input_data.model_dump())

    return application


app = create_app()
