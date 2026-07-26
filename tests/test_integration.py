from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app import create_app
from src.evaluate import evaluate_model
from src.generate_data import generate_sensor_data
from src.predict import AnomalyPredictor
from src.preprocess import preprocess_data
from src.train import train_model


@pytest.fixture(scope="session")
def trained_bundle(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("trained_bundle")
    data_path = root / "sensor_data.csv"
    preprocessed_path = root / "preprocessed_data.npz"
    preprocess_metadata_path = root / "preprocessing_metadata.json"
    scaler_path = root / "scaler.pkl"
    model_path = root / "autoencoder.pt"

    generate_sensor_data(
        output_path=data_path,
        normal_count=500,
        anomaly_count=100,
        random_seed=7,
    )
    preprocess_data(
        input_path=data_path,
        output_path=preprocessed_path,
        scaler_path=scaler_path,
        metadata_path=preprocess_metadata_path,
        random_seed=7,
    )
    training = train_model(
        data_path=preprocessed_path,
        model_path=model_path,
        history_path=root / "history.csv",
        metadata_path=root / "model_metadata.json",
        training_curve_path=root / "training_curve.png",
        epochs=60,
        patience=12,
        random_seed=7,
    )
    evaluation = evaluate_model(
        data_path=preprocessed_path,
        model_path=model_path,
        metrics_path=root / "metrics.csv",
        metrics_json_path=root / "metrics.json",
        predictions_path=root / "predictions.csv",
        confusion_matrix_path=root / "confusion.png",
        error_distribution_path=root / "errors.png",
        pr_curve_path=root / "pr.png",
        report_path=root / "model_card.md",
        report_metrics_path=root / "evaluation_summary.json",
    )
    predictor = AnomalyPredictor(model_path=model_path, scaler_path=scaler_path)
    return {
        "root": root,
        "training": training,
        "evaluation": evaluation,
        "predictor": predictor,
    }


def test_training_outputs_and_metrics(trained_bundle) -> None:
    root: Path = trained_bundle["root"]
    metrics = trained_bundle["evaluation"]["metrics"]

    assert trained_bundle["training"]["threshold"] > 0
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1_score"] <= 1
    assert metrics["test_size"] == 120
    assert (root / "model_card.md").exists()
    assert (root / "predictions.csv").exists()
    assert (root / "confusion.png").exists()


def test_predictor_uses_checkpoint_threshold(trained_bundle) -> None:
    predictor: AnomalyPredictor = trained_bundle["predictor"]
    assert predictor.threshold == pytest.approx(
        trained_bundle["training"]["threshold"]
    )

    normal_values = predictor.scaler.mean_
    normal = predictor.predict(
        {
            "temperature": float(normal_values[0]),
            "vibration": float(normal_values[1]),
            "pressure": float(normal_values[2]),
            "humidity": float(normal_values[3]),
        }
    )
    anomaly = predictor.predict(
        {
            "temperature": 55.0,
            "vibration": 1.6,
            "pressure": 135.0,
            "humidity": 80.0,
        }
    )

    assert normal["prediction"] == "normal"
    assert anomaly["prediction"] == "anomaly"
    assert anomaly["reconstruction_error"] > normal["reconstruction_error"]
    assert set(anomaly["feature_errors"]) == {
        "temperature",
        "vibration",
        "pressure",
        "humidity",
    }


def test_api_health_prediction_and_validation(trained_bundle) -> None:
    predictor: AnomalyPredictor = trained_bundle["predictor"]
    application = create_app(lambda: predictor)

    with TestClient(application) as client:
        health_response = client.get("/health")
        prediction_response = client.post(
            "/predict",
            json={
                "temperature": 30.0,
                "vibration": 0.35,
                "pressure": 100.0,
                "humidity": 45.0,
            },
        )
        invalid_response = client.post(
            "/predict",
            json={
                "temperature": 30.0,
                "vibration": 0.35,
                "pressure": 100.0,
                "humidity": 120.0,
            },
        )

    assert health_response.status_code == 200
    assert health_response.json()["status"] == "ok"
    assert prediction_response.status_code == 200
    assert prediction_response.json()["prediction"] in {"normal", "anomaly"}
    assert invalid_response.status_code == 422
