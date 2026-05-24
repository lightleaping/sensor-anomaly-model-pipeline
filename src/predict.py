import argparse
from pathlib import Path
import pickle

import numpy as np
import torch

from model import SensorAutoEncoder, reconstruction_error


FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "humidity"]


def predict_anomaly(
    temperature: float,
    vibration: float,
    pressure: float,
    humidity: float,
    model_path: str = "models/autoencoder.pt",
    scaler_path: str = "models/scaler.pkl",
    threshold: float = 1.569104552268982,
) -> None:
    model_file = Path(model_path)
    scaler_file = Path(scaler_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")

    sample = np.array([[temperature, vibration, pressure, humidity]], dtype=np.float32)

    with open(scaler_file, "rb") as f:
        scaler = pickle.load(f)

    sample_scaled = scaler.transform(sample).astype(np.float32)

    checkpoint = torch.load(model_file, map_location="cpu")

    model = SensorAutoEncoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        sample_tensor = torch.tensor(sample_scaled)
        reconstructed = model(sample_tensor)
        error = reconstruction_error(sample_tensor, reconstructed).item()

    is_anomaly = error > threshold

    result = {
        "input": {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "humidity": humidity,
        },
        "reconstruction_error": round(error, 6),
        "threshold": round(threshold, 6),
        "prediction": "anomaly" if is_anomaly else "normal",
    }

    print(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--vibration", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--threshold", type=float, default=1.569104552268982)
    args = parser.parse_args()

    predict_anomaly(
        temperature=args.temperature,
        vibration=args.vibration,
        pressure=args.pressure,
        humidity=args.humidity,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
