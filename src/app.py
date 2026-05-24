from pathlib import Path
import pickle

import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.model import SensorAutoEncoder, reconstruction_error


MODEL_PATH = Path("models/autoencoder.pt")
SCALER_PATH = Path("models/scaler.pkl")
THRESHOLD = 1.569104552268982


class SensorInput(BaseModel):
    temperature: float = Field(..., example=30.0)
    vibration: float = Field(..., example=0.35)
    pressure: float = Field(..., example=100.0)
    humidity: float = Field(..., example=45.0)


class PredictionResponse(BaseModel):
    prediction: str
    reconstruction_error: float
    threshold: float
    input: SensorInput


app = FastAPI(
    title="Sensor Anomaly Detection API",
    description="PyTorch AutoEncoder 기반 센서 데이터 이상탐지 추론 API",
    version="1.0.0",
)


def load_model_and_scaler():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(f"Scaler file not found: {SCALER_PATH}")

    checkpoint = torch.load(MODEL_PATH, map_location="cpu")

    model = SensorAutoEncoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


model, scaler = load_model_and_scaler()


@app.get("/")
def root():
    return {
        "message": "Sensor Anomaly Detection API",
        "model": "PyTorch AutoEncoder",
        "endpoint": "/predict",
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(input_data: SensorInput):
    sample = np.array(
        [[
            input_data.temperature,
            input_data.vibration,
            input_data.pressure,
            input_data.humidity,
        ]],
        dtype=np.float32,
    )

    sample_scaled = scaler.transform(sample).astype(np.float32)

    with torch.no_grad():
        sample_tensor = torch.tensor(sample_scaled)
        reconstructed = model(sample_tensor)
        error = reconstruction_error(sample_tensor, reconstructed).item()

    is_anomaly = error > THRESHOLD

    return PredictionResponse(
        prediction="anomaly" if is_anomaly else "normal",
        reconstruction_error=round(error, 6),
        threshold=round(THRESHOLD, 6),
        input=input_data,
    )
