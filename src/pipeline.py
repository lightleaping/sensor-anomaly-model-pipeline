from __future__ import annotations

import argparse
from datetime import datetime, timezone

from src.artifacts import write_json
from src.config import DEFAULT_RUN_SUMMARY_PATH
from src.evaluate import evaluate_model
from src.generate_data import generate_sensor_data
from src.predict import AnomalyPredictor
from src.preprocess import preprocess_data
from src.train import train_model


def run_pipeline(
    normal_count: int = 1500,
    anomaly_count: int = 300,
    epochs: int = 120,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    threshold_percentile: float = 95.0,
    random_seed: int = 42,
    minimum_recall: float = 0.85,
    minimum_f1: float = 0.80,
    enforce_quality_gate: bool = True,
) -> dict[str, object]:
    started_at = datetime.now(timezone.utc).isoformat()
    print("Step 1/5 - Generate data")
    data_summary = generate_sensor_data(
        normal_count=normal_count,
        anomaly_count=anomaly_count,
        random_seed=random_seed,
    )

    print("\nStep 2/5 - Preprocess")
    preprocess_summary = preprocess_data(random_seed=random_seed)

    print("\nStep 3/5 - Train")
    model_summary = train_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        threshold_percentile=threshold_percentile,
        random_seed=random_seed,
    )

    print("\nStep 4/5 - Evaluate")
    evaluation = evaluate_model()
    metrics = evaluation["metrics"]

    print("\nStep 5/5 - Smoke-test inference")
    predictor = AnomalyPredictor()
    normal_prediction = predictor.predict(
        {
            "temperature": 30.0,
            "vibration": 0.35,
            "pressure": 100.0,
            "humidity": 45.0,
        }
    )
    anomaly_prediction = predictor.predict(
        {
            "temperature": 55.0,
            "vibration": 1.6,
            "pressure": 135.0,
            "humidity": 80.0,
        }
    )

    failures = []
    if float(metrics["recall"]) < minimum_recall:
        failures.append(
            f"recall {metrics['recall']:.4f} is below {minimum_recall:.4f}"
        )
    if float(metrics["f1_score"]) < minimum_f1:
        failures.append(
            f"F1 {metrics['f1_score']:.4f} is below {minimum_f1:.4f}"
        )
    if normal_prediction["prediction"] != "normal":
        failures.append("normal smoke sample was classified as anomaly")
    if anomaly_prediction["prediction"] != "anomaly":
        failures.append("anomaly smoke sample was classified as normal")

    summary = {
        "status": "failed" if failures else "passed",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "data": data_summary,
        "preprocessing": preprocess_summary,
        "model": {
            "model_version": model_summary["model_version"],
            "threshold": model_summary["threshold"],
            "best_epoch": model_summary["best_epoch"],
            "completed_epochs": model_summary["training_config"]["completed_epochs"],
        },
        "evaluation": evaluation,
        "quality_gate": {
            "minimum_recall": minimum_recall,
            "minimum_f1": minimum_f1,
            "failures": failures,
        },
        "smoke_predictions": {
            "normal": normal_prediction,
            "anomaly": anomaly_prediction,
        },
    }
    summary_path = write_json(DEFAULT_RUN_SUMMARY_PATH, summary)
    print(f"Run summary: {summary_path}")

    if failures and enforce_quality_gate:
        raise RuntimeError("Pipeline quality gate failed: " + "; ".join(failures))

    print("Pipeline completed successfully.")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run data generation, training, evaluation, and smoke inference."
    )
    parser.add_argument("--normal-count", type=int, default=1500)
    parser.add_argument("--anomaly-count", type=int, default=300)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.85)
    parser.add_argument("--minimum-f1", type=float, default=0.80)
    parser.add_argument("--skip-quality-gate", action="store_true")
    args = parser.parse_args()

    run_pipeline(
        normal_count=args.normal_count,
        anomaly_count=args.anomaly_count,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        threshold_percentile=args.threshold_percentile,
        random_seed=args.seed,
        minimum_recall=args.minimum_recall,
        minimum_f1=args.minimum_f1,
        enforce_quality_gate=not args.skip_quality_gate,
    )


if __name__ == "__main__":
    main()
