from __future__ import annotations

import json
import math
import pickle
from pathlib import Path
from typing import Any

import torch

from src.config import FEATURE_COLUMNS, project_path
from src.model import SensorAutoEncoder


CHECKPOINT_SCHEMA_VERSION = 1


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    checkpoint_path = project_path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint_path}. "
            "Run `python -m src.pipeline` first."
        )

    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    required = {
        "model_state_dict",
        "input_dim",
        "latent_dim",
        "threshold",
        "feature_columns",
    }
    missing = required.difference(checkpoint)
    if missing:
        raise ValueError(
            f"Invalid checkpoint {checkpoint_path}; missing keys: {sorted(missing)}"
        )

    threshold = float(checkpoint["threshold"])
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError(f"Invalid anomaly threshold in checkpoint: {threshold}")

    feature_columns = tuple(checkpoint["feature_columns"])
    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Checkpoint feature order does not match the application: "
            f"{feature_columns} != {FEATURE_COLUMNS}"
        )

    return checkpoint


def load_model_bundle(
    model_path: str | Path,
    scaler_path: str | Path,
) -> tuple[SensorAutoEncoder, Any, dict[str, Any]]:
    checkpoint = load_checkpoint(model_path)
    scaler_file = project_path(scaler_path)
    if not scaler_file.exists():
        raise FileNotFoundError(
            f"Scaler not found: {scaler_file}. Run `python -m src.pipeline` first."
        )

    model = SensorAutoEncoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 8)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with scaler_file.open("rb") as file:
        scaler = pickle.load(file)

    scaler_features = getattr(scaler, "n_features_in_", None)
    if scaler_features != len(FEATURE_COLUMNS):
        raise ValueError(
            f"Scaler expects {scaler_features} features; "
            f"{len(FEATURE_COLUMNS)} are required."
        )

    return model, scaler, checkpoint


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output_path = project_path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
