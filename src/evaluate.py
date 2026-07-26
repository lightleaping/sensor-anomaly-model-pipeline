from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.artifacts import load_checkpoint, write_json
from src.config import (
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_ERROR_DISTRIBUTION_PATH,
    DEFAULT_METRICS_JSON_PATH,
    DEFAULT_METRICS_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    DEFAULT_PREPROCESSED_PATH,
    DEFAULT_PR_CURVE_PATH,
    DEFAULT_README_PATH,
    DEFAULT_REPORT_METRICS_PATH,
    DEFAULT_REPORT_PATH,
    FEATURE_COLUMNS,
    PROJECT_ROOT,
    project_path,
)
from src.model import SensorAutoEncoder, reconstruction_error


def _plot_confusion_matrix(cm: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5.5, 4.5))
    image = axis.imshow(cm, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_title("Confusion matrix")
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_xticks([0, 1], labels=["Normal", "Anomaly"])
    axis.set_yticks([0, 1], labels=["Normal", "Anomaly"])
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                str(cm[row, column]),
                ha="center",
                va="center",
                color="white" if cm[row, column] > cm.max() / 2 else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_error_distribution(
    errors: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.hist(errors[labels == 0], bins=35, alpha=0.65, label="Normal")
    axis.hist(errors[labels == 1], bins=35, alpha=0.65, label="Anomaly")
    axis.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold ({threshold:.3f})",
    )
    axis.set_xlabel("Reconstruction error")
    axis.set_ylabel("Samples")
    axis.set_title("Test reconstruction-error distribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_precision_recall(
    labels: np.ndarray,
    errors: np.ndarray,
    average_precision: float,
    output_path: Path,
) -> None:
    precision, recall, _ = precision_recall_curve(labels, errors)
    baseline = float(labels.mean())
    figure, axis = plt.subplots(figsize=(6, 4.5))
    axis.plot(recall, precision, label=f"AP={average_precision:.3f}")
    axis.axhline(
        baseline,
        color="gray",
        linestyle="--",
        label=f"Positive rate={baseline:.3f}",
    )
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title("Precision-recall curve")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _write_model_card(
    output_path: str | Path,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    anomaly_type_metrics: dict[str, dict[str, Any]],
    report_metrics_path: Path,
) -> Path:
    path = project_path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    confusion = metrics["confusion_matrix"]
    try:
        report_metrics_display = report_metrics_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        report_metrics_display = report_metrics_path.as_posix()
    anomaly_rows = "\n".join(
        f"| `{name}` | {values['support']} | {values['detected']} | "
        f"{values['recall']:.4f} | {values['mean_error']:.4f} |"
        for name, values in anomaly_type_metrics.items()
    )
    content = f"""# Sensor Autoencoder Model Card

## Model

- Version: `{checkpoint.get("model_version", "unknown")}`
- Architecture: `{checkpoint["input_dim"]} → {checkpoint.get("hidden_dim", 8)} → {checkpoint["latent_dim"]} → {checkpoint.get("hidden_dim", 8)} → {checkpoint["input_dim"]}`
- Features: {", ".join(checkpoint["feature_columns"])}
- Training data: normal samples only
- Decision rule: reconstruction error > validation {checkpoint["threshold_percentile"]}th percentile
- Threshold: `{metrics["threshold"]:.6f}`

## Held-out test results

| Metric | Value |
|---|---:|
| Accuracy | {metrics["accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1_score"]:.4f} |
| Balanced accuracy | {metrics["balanced_accuracy"]:.4f} |
| Specificity | {metrics["specificity"]:.4f} |
| ROC AUC | {metrics["roc_auc"]:.4f} |
| Average precision | {metrics["average_precision"]:.4f} |
| Matthews correlation coefficient | {metrics["matthews_correlation_coefficient"]:.4f} |

Confusion matrix: TN={confusion["true_negative"]}, FP={confusion["false_positive"]}, FN={confusion["false_negative"]}, TP={confusion["true_positive"]}.

## Anomaly-type detection

| Anomaly type | Support | Detected | Recall | Mean error |
|---|---:|---:|---:|---:|
{anomaly_rows}

The machine-readable evaluation summary is stored in `{report_metrics_display}`.

## Intended use

This model is a reproducible row-level baseline for detecting unusual combinations of temperature, vibration, pressure, and humidity. It is suitable for demos and pipeline verification.

## Limitations

- The checked-in evaluation is based on synthetic data, not a production machine.
- The model does not use temporal windows, so it cannot learn trend or sequence anomalies.
- The threshold must be recalibrated after sensor replacement, operating-regime changes, or data drift.
- A prediction is an operational signal, not a diagnosis of equipment failure.
"""
    path.write_text(content, encoding="utf-8")
    return path


def _calculate_anomaly_type_metrics(
    anomaly_types: np.ndarray,
    labels: np.ndarray,
    predictions: np.ndarray,
    errors: np.ndarray,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for anomaly_type in sorted(set(anomaly_types[labels == 1].tolist())):
        mask = (labels == 1) & (anomaly_types == anomaly_type)
        support = int(mask.sum())
        detected = int(predictions[mask].sum())
        result[str(anomaly_type)] = {
            "support": support,
            "detected": detected,
            "missed": support - detected,
            "recall": detected / support if support else 0.0,
            "mean_error": float(errors[mask].mean()) if support else 0.0,
            "median_error": float(np.median(errors[mask])) if support else 0.0,
        }
    return result


def _update_readme_evaluation(
    readme_path: str | Path | None,
    checkpoint: dict[str, Any],
    metrics: dict[str, Any],
    anomaly_type_metrics: dict[str, dict[str, Any]],
) -> Path | None:
    if readme_path is None:
        return None
    path = project_path(readme_path)
    if not path.exists():
        return None

    start_marker = "<!-- EVALUATION_RESULTS_START -->"
    end_marker = "<!-- EVALUATION_RESULTS_END -->"
    content = path.read_text(encoding="utf-8")
    if start_marker not in content or end_marker not in content:
        raise ValueError(
            f"README evaluation markers are missing from {path}: "
            f"{start_marker}, {end_marker}"
        )

    confusion = metrics["confusion_matrix"]
    anomaly_rows = "\n".join(
        f"| `{name}` | {values['support']} | {values['detected']} | "
        f"{values['recall']:.4f} |"
        for name, values in anomaly_type_metrics.items()
    )
    generated = f"""{start_marker}
> 이 표는 `python -m src.pipeline` 실행 시 `reports/evaluation_summary.json`에서 자동 갱신됩니다.

| 항목 | 결과 |
|---|---:|
| Model version | `{checkpoint.get("model_version", "unknown")}` |
| Test samples | {metrics["test_size"]} |
| Threshold | {metrics["threshold"]:.6f} |
| Accuracy | {metrics["accuracy"]:.4f} |
| Balanced Accuracy | {metrics["balanced_accuracy"]:.4f} |
| Precision | {metrics["precision"]:.4f} |
| Recall | {metrics["recall"]:.4f} |
| F1 | {metrics["f1_score"]:.4f} |
| Specificity | {metrics["specificity"]:.4f} |
| ROC AUC | {metrics["roc_auc"]:.4f} |
| Average Precision | {metrics["average_precision"]:.4f} |
| MCC | {metrics["matthews_correlation_coefficient"]:.4f} |
| Confusion Matrix | TN {confusion["true_negative"]} · FP {confusion["false_positive"]} · FN {confusion["false_negative"]} · TP {confusion["true_positive"]} |

이상 유형별 탐지 결과:

| 유형 | Test 수 | 탐지 | Recall |
|---|---:|---:|---:|
{anomaly_rows}
{end_marker}"""
    before = content.split(start_marker, 1)[0]
    after = content.split(end_marker, 1)[1]
    path.write_text(before + generated + after, encoding="utf-8")
    return path


def evaluate_model(
    data_path: str | Path = DEFAULT_PREPROCESSED_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    metrics_path: str | Path = DEFAULT_METRICS_PATH,
    metrics_json_path: str | Path = DEFAULT_METRICS_JSON_PATH,
    predictions_path: str | Path = DEFAULT_PREDICTIONS_PATH,
    confusion_matrix_path: str | Path = DEFAULT_CONFUSION_MATRIX_PATH,
    error_distribution_path: str | Path = DEFAULT_ERROR_DISTRIBUTION_PATH,
    pr_curve_path: str | Path = DEFAULT_PR_CURVE_PATH,
    report_path: str | Path = DEFAULT_REPORT_PATH,
    report_metrics_path: str | Path = DEFAULT_REPORT_METRICS_PATH,
    readme_path: str | Path | None = None,
    threshold_override: float | None = None,
) -> dict[str, Any]:
    data_file = project_path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(
            f"Preprocessed data not found: {data_file}. "
            "Run `python -m src.pipeline` first."
        )

    with np.load(data_file) as data:
        X_test = data["X_test"].astype(np.float32)
        X_test_raw = (
            data["X_test_raw"].astype(np.float32)
            if "X_test_raw" in data
            else X_test.copy()
        )
        y_test = data["y_test"].astype(np.int64)
        test_indices = (
            data["test_indices"].astype(np.int64)
            if "test_indices" in data
            else np.arange(len(y_test))
        )
        test_sample_ids = (
            data["test_sample_ids"].astype(str)
            if "test_sample_ids" in data
            else test_indices.astype(str)
        )
        test_anomaly_types = (
            data["test_anomaly_types"].astype(str)
            if "test_anomaly_types" in data
            else np.where(y_test == 1, "anomaly", "normal")
        )

    if set(np.unique(y_test)) != {0, 1}:
        raise ValueError("The test split must contain both normal and anomaly labels")

    checkpoint = load_checkpoint(model_path)
    model = SensorAutoEncoder(
        input_dim=int(checkpoint["input_dim"]),
        latent_dim=int(checkpoint["latent_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 8)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_tensor = torch.from_numpy(X_test)
    with torch.no_grad():
        test_errors = reconstruction_error(test_tensor, model(test_tensor)).numpy()

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else float(checkpoint["threshold"])
    )
    if not np.isfinite(threshold) or threshold < 0:
        raise ValueError("threshold must be a finite non-negative value")
    y_pred = (test_errors > threshold).astype(np.int64)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in cm.ravel()
    )
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive
        else 0.0
    )
    negative_predictive_value = (
        true_negative / (true_negative + false_negative)
        if true_negative + false_negative
        else 0.0
    )
    metrics = {
        "threshold": threshold,
        "threshold_percentile": float(checkpoint["threshold_percentile"]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": precision,
        "recall": recall,
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "negative_predictive_value": float(negative_predictive_value),
        "false_positive_rate": float(1.0 - specificity),
        "false_negative_rate": float(1.0 - recall),
        "roc_auc": float(roc_auc_score(y_test, test_errors)),
        "average_precision": float(average_precision_score(y_test, test_errors)),
        "matthews_correlation_coefficient": float(
            matthews_corrcoef(y_test, y_pred)
        ),
        "test_size": int(len(y_test)),
        "normal_count": int((y_test == 0).sum()),
        "anomaly_count": int((y_test == 1).sum()),
        "confusion_matrix": {
            "true_negative": true_negative,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_positive": true_positive,
        },
    }
    anomaly_type_metrics = _calculate_anomaly_type_metrics(
        test_anomaly_types, y_test, y_pred, test_errors
    )
    report = classification_report(
        y_test,
        y_pred,
        target_names=["normal", "anomaly"],
        output_dict=True,
        zero_division=0,
    )
    payload = {
        "model_version": checkpoint.get("model_version", "unknown"),
        "model_trained_at": checkpoint.get("trained_at"),
        "metrics": metrics,
        "anomaly_type_metrics": anomaly_type_metrics,
        "classification_report": report,
    }

    metrics_file = project_path(metrics_path)
    predictions_file = project_path(predictions_path)
    cm_file = project_path(confusion_matrix_path)
    error_file = project_path(error_distribution_path)
    pr_file = project_path(pr_curve_path)
    for path in (metrics_file, predictions_file, cm_file, error_file, pr_file):
        path.parent.mkdir(parents=True, exist_ok=True)

    flat_metrics = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrix"
    }
    flat_metrics.update(metrics["confusion_matrix"])
    pd.DataFrame([flat_metrics]).to_csv(
        metrics_file, index=False, encoding="utf-8-sig"
    )

    predictions = pd.DataFrame(X_test_raw, columns=FEATURE_COLUMNS)
    predictions.insert(0, "source_index", test_indices)
    predictions.insert(1, "sample_id", test_sample_ids)
    predictions.insert(2, "anomaly_type", test_anomaly_types)
    predictions["y_true"] = y_test
    predictions["y_pred"] = y_pred
    predictions["reconstruction_error"] = test_errors
    predictions["threshold"] = threshold
    predictions["error_margin"] = test_errors - threshold
    predictions.to_csv(predictions_file, index=False, encoding="utf-8-sig")

    metrics_json_file = write_json(metrics_json_path, payload)
    report_metrics_file = write_json(report_metrics_path, payload)
    _plot_confusion_matrix(cm, cm_file)
    _plot_error_distribution(test_errors, y_test, threshold, error_file)
    _plot_precision_recall(
        y_test, test_errors, metrics["average_precision"], pr_file
    )
    model_card_file = _write_model_card(
        report_path,
        checkpoint,
        metrics,
        anomaly_type_metrics,
        report_metrics_file,
    )
    updated_readme = _update_readme_evaluation(
        readme_path, checkpoint, metrics, anomaly_type_metrics
    )

    print("Evaluation completed")
    print(
        f"Accuracy={metrics['accuracy']:.4f} | Precision={precision:.4f} | "
        f"Recall={recall:.4f} | F1={metrics['f1_score']:.4f}"
    )
    print(
        f"Confusion matrix: TN={true_negative}, FP={false_positive}, "
        f"FN={false_negative}, TP={true_positive}"
    )
    print(f"Metrics: {metrics_json_file}")
    print(f"Model card: {model_card_file}")
    if updated_readme is not None:
        print(f"README metrics updated: {updated_readme}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained sensor model.")
    parser.add_argument("--data", default=str(DEFAULT_PREPROCESSED_PATH))
    parser.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--threshold", type=float)
    parser.add_argument(
        "--update-readme",
        action="store_true",
        help="Update the generated evaluation block in README.md.",
    )
    args = parser.parse_args()

    evaluate_model(
        data_path=args.data,
        model_path=args.model,
        readme_path=DEFAULT_README_PATH if args.update_readme else None,
        threshold_override=args.threshold,
    )


if __name__ == "__main__":
    main()
