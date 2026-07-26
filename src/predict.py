from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from src.artifacts import load_model_bundle
from src.config import (
    DEFAULT_MODEL_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_COLUMNS,
)


class AnomalyPredictor:
    """Reusable in-process predictor shared by the CLI and FastAPI."""

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
        scaler_path: str | Path = DEFAULT_SCALER_PATH,
        threshold_override: float | None = None,
    ) -> None:
        self.model, self.scaler, self.checkpoint = load_model_bundle(
            model_path, scaler_path
        )
        self.threshold = (
            float(threshold_override)
            if threshold_override is not None
            else float(self.checkpoint["threshold"])
        )
        if not math.isfinite(self.threshold) or self.threshold < 0:
            raise ValueError("threshold must be a finite non-negative number")
        self.model_version = str(self.checkpoint.get("model_version", "unknown"))

    def predict(self, values: Mapping[str, float]) -> dict[str, object]:
        missing = [feature for feature in FEATURE_COLUMNS if feature not in values]
        if missing:
            raise ValueError(f"Missing sensor features: {missing}")

        numeric_values = [float(values[feature]) for feature in FEATURE_COLUMNS]
        if not all(math.isfinite(value) for value in numeric_values):
            raise ValueError("All sensor values must be finite numbers")

        sample = np.asarray([numeric_values], dtype=np.float32)
        sample_scaled = self.scaler.transform(sample).astype(np.float32)
        sample_tensor = torch.from_numpy(sample_scaled)

        with torch.no_grad():
            reconstructed = self.model(sample_tensor)
            squared_error = (sample_tensor - reconstructed).pow(2)
            error = float(squared_error.mean(dim=1).item())

        feature_errors = {
            feature: round(float(squared_error[0, index].item()), 6)
            for index, feature in enumerate(FEATURE_COLUMNS)
        }
        return {
            "prediction": "anomaly" if error > self.threshold else "normal",
            "reconstruction_error": round(error, 6),
            "threshold": round(self.threshold, 6),
            "error_margin": round(error - self.threshold, 6),
            "feature_errors": feature_errors,
            "model_version": self.model_version,
            "input": {
                feature: numeric_values[index]
                for index, feature in enumerate(FEATURE_COLUMNS)
            },
        }


def predict_anomaly(
    temperature: float,
    vibration: float,
    pressure: float,
    humidity: float,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    threshold: float | None = None,
) -> dict[str, object]:
    predictor = AnomalyPredictor(
        model_path=model_path,
        scaler_path=scaler_path,
        threshold_override=threshold,
    )
    return predictor.predict(
        {
            "temperature": temperature,
            "vibration": vibration,
            "pressure": pressure,
            "humidity": humidity,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one sensor observation.")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--vibration", type=float, required=True)
    parser.add_argument("--pressure", type=float, required=True)
    parser.add_argument("--humidity", type=float, required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER_PATH))
    parser.add_argument(
        "--threshold",
        type=float,
        help="Optional operational override; defaults to the trained checkpoint.",
    )
    args = parser.parse_args()

    result = predict_anomaly(
        temperature=args.temperature,
        vibration=args.vibration,
        pressure=args.pressure,
        humidity=args.humidity,
        model_path=args.model,
        scaler_path=args.scaler,
        threshold=args.threshold,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
