from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.artifacts import CHECKPOINT_SCHEMA_VERSION, write_json
from src.config import (
    DEFAULT_HISTORY_PATH,
    DEFAULT_MODEL_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREPROCESSED_PATH,
    DEFAULT_TRAINING_CURVE_PATH,
    FEATURE_COLUMNS,
    project_path,
)
from src.model import SensorAutoEncoder, reconstruction_error


def _state_fingerprint(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def train_model(
    data_path: str | Path = DEFAULT_PREPROCESSED_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    history_path: str | Path = DEFAULT_HISTORY_PATH,
    metadata_path: str | Path = DEFAULT_MODEL_METADATA_PATH,
    training_curve_path: str | Path = DEFAULT_TRAINING_CURVE_PATH,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    latent_dim: int = 2,
    hidden_dim: int = 8,
    threshold_percentile: float = 95.0,
    patience: int = 20,
    min_delta: float = 1e-5,
    random_seed: int = 42,
) -> dict[str, Any]:
    if epochs < 1:
        raise ValueError("epochs must be positive")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if not 50 <= threshold_percentile < 100:
        raise ValueError("threshold_percentile must be in [50, 100)")

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    torch.use_deterministic_algorithms(True)

    data_file = project_path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_file}. "
            "Run `python -m src.preprocess` first."
        )

    with np.load(data_file) as data:
        X_train = data["X_train"].astype(np.float32)
        X_val = data["X_val"].astype(np.float32)
        feature_columns = tuple(data["feature_columns"].tolist())

    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            f"Unexpected feature order: {feature_columns}; expected {FEATURE_COLUMNS}"
        )
    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Training and validation splits must not be empty")

    train_dataset = TensorDataset(torch.from_numpy(X_train))
    loader_generator = torch.Generator().manual_seed(random_seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, len(train_dataset)),
        shuffle=True,
        generator=loader_generator,
    )
    val_tensor = torch.from_numpy(X_val)

    model = SensorAutoEncoder(
        input_dim=X_train.shape[1],
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    history: list[dict[str, float | int]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_val_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0

        for (batch_x,) in train_loader:
            optimizer.zero_grad(set_to_none=True)
            reconstructed = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * len(batch_x)
            sample_count += len(batch_x)

        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(val_tensor), val_tensor).item())
        train_loss = loss_sum / sample_count
        history.append(
            {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        )

        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if patience > 0 and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_reconstructed = model(val_tensor)
        val_errors = reconstruction_error(val_tensor, val_reconstructed).numpy()
    threshold = float(np.percentile(val_errors, threshold_percentile))

    fingerprint = _state_fingerprint(best_state)
    model_version = f"sensor-ae-{fingerprint[:12]}"
    trained_at = datetime.now(timezone.utc).isoformat()
    training_config = {
        "requested_epochs": epochs,
        "completed_epochs": len(history),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "threshold_percentile": threshold_percentile,
        "patience": patience,
        "min_delta": min_delta,
        "random_seed": random_seed,
    }

    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model_state_dict": best_state,
        "input_dim": int(X_train.shape[1]),
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "feature_columns": list(FEATURE_COLUMNS),
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "model_version": model_version,
        "trained_at": trained_at,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "training_config": training_config,
        "validation_error_summary": {
            "minimum": float(val_errors.min()),
            "mean": float(val_errors.mean()),
            "maximum": float(val_errors.max()),
            "standard_deviation": float(val_errors.std()),
        },
    }

    model_file = project_path(model_path)
    history_file = project_path(history_path)
    curve_file = project_path(training_curve_path)
    model_file.parent.mkdir(parents=True, exist_ok=True)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    curve_file.parent.mkdir(parents=True, exist_ok=True)

    torch.save(checkpoint, model_file)
    history_frame = pd.DataFrame(history)
    history_frame.to_csv(history_file, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(7, 4))
    plt.plot(history_frame["epoch"], history_frame["train_loss"], label="Train")
    plt.plot(history_frame["epoch"], history_frame["val_loss"], label="Validation")
    plt.axvline(best_epoch, color="black", linestyle="--", alpha=0.5, label="Best")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(curve_file, dpi=150)
    plt.close()

    metadata = {
        key: value
        for key, value in checkpoint.items()
        if key != "model_state_dict"
    }
    metadata["model_path"] = str(model_file)
    metadata["history_path"] = str(history_file)
    metadata["training_curve_path"] = str(curve_file)
    metadata_file = write_json(metadata_path, metadata)

    print(f"Model: {model_file}")
    print(f"Model version: {model_version}")
    print(f"Threshold: {threshold:.6f} ({threshold_percentile}th percentile)")
    print(f"Metadata: {metadata_file}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a normal-only autoencoder.")
    parser.add_argument("--data", default=str(DEFAULT_PREPROCESSED_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--history", default=str(DEFAULT_HISTORY_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_MODEL_METADATA_PATH))
    parser.add_argument("--training-curve", default=str(DEFAULT_TRAINING_CURVE_PATH))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--latent-dim", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=8)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_model(
        data_path=args.data,
        model_path=args.model,
        history_path=args.history,
        metadata_path=args.metadata,
        training_curve_path=args.training_curve,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        threshold_percentile=args.threshold_percentile,
        patience=args.patience,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
