import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from model import SensorAutoEncoder, reconstruction_error


def evaluate_model(
    data_path: str = "outputs/preprocessed_data.npz",
    model_path: str = "models/autoencoder.pt",
    metrics_path: str = "outputs/evaluation_metrics.csv",
    predictions_path: str = "outputs/test_predictions.csv",
    confusion_matrix_path: str = "outputs/confusion_matrix.png",
    threshold_percentile: float = 95.0,
) -> None:
    data_file = Path(data_path)
    model_file = Path(model_path)

    if not data_file.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {data_file}")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")

    data = np.load(data_file)
    X_val = data["X_val"].astype(np.float32)
    X_test = data["X_test"].astype(np.float32)
    y_test = data["y_test"].astype(np.int64)

    checkpoint = torch.load(model_file, map_location="cpu")
    model = SensorAutoEncoder(
        input_dim=checkpoint["input_dim"],
        latent_dim=checkpoint["latent_dim"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        val_tensor = torch.tensor(X_val)
        test_tensor = torch.tensor(X_test)

        val_reconstructed = model(val_tensor)
        test_reconstructed = model(test_tensor)

        val_errors = reconstruction_error(val_tensor, val_reconstructed).numpy()
        test_errors = reconstruction_error(test_tensor, test_reconstructed).numpy()

    threshold = float(np.percentile(val_errors, threshold_percentile))
    y_pred = (test_errors > threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        y_pred,
        average="binary",
        zero_division=0,
    )

    cm = confusion_matrix(y_test, y_pred)

    metrics = pd.DataFrame([
        {
            "threshold_percentile": threshold_percentile,
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "test_size": len(y_test),
            "normal_count": int((y_test == 0).sum()),
            "anomaly_count": int((y_test == 1).sum()),
        }
    ])

    predictions = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
        "reconstruction_error": test_errors,
    })

    metrics_file = Path(metrics_path)
    predictions_file = Path(predictions_path)
    cm_file = Path(confusion_matrix_path)

    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    predictions_file.parent.mkdir(parents=True, exist_ok=True)
    cm_file.parent.mkdir(parents=True, exist_ok=True)

    metrics.to_csv(metrics_file, index=False, encoding="utf-8-sig")
    predictions.to_csv(predictions_file, index=False, encoding="utf-8-sig")

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks([0, 1], ["Normal", "Anomaly"])
    plt.yticks([0, 1], ["Normal", "Anomaly"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(cm_file)
    plt.close()

    print("Evaluation completed")
    print(f"Threshold: {threshold:.6f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, target_names=["normal", "anomaly"], zero_division=0))
    print(f"Saved metrics: {metrics_file}")
    print(f"Saved predictions: {predictions_file}")
    print(f"Saved confusion matrix: {cm_file}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="outputs/preprocessed_data.npz")
    parser.add_argument("--model", default="models/autoencoder.pt")
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    args = parser.parse_args()

    evaluate_model(
        data_path=args.data,
        model_path=args.model,
        threshold_percentile=args.threshold_percentile,
    )


if __name__ == "__main__":
    main()
