from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_COLUMNS = ("temperature", "vibration", "pressure", "humidity")
LABEL_COLUMN = "label"

DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "sensor_data.csv"
DEFAULT_PREPROCESSED_PATH = PROJECT_ROOT / "outputs" / "preprocessed_data.npz"
DEFAULT_PREPROCESS_METADATA_PATH = (
    PROJECT_ROOT / "outputs" / "preprocessing_metadata.json"
)
DEFAULT_SCALER_PATH = PROJECT_ROOT / "models" / "scaler.pkl"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "autoencoder.pt"
DEFAULT_MODEL_METADATA_PATH = PROJECT_ROOT / "models" / "model_metadata.json"
DEFAULT_HISTORY_PATH = PROJECT_ROOT / "outputs" / "train_history.csv"
DEFAULT_TRAINING_CURVE_PATH = PROJECT_ROOT / "outputs" / "training_curve.png"
DEFAULT_METRICS_PATH = PROJECT_ROOT / "outputs" / "evaluation_metrics.csv"
DEFAULT_METRICS_JSON_PATH = PROJECT_ROOT / "outputs" / "evaluation_metrics.json"
DEFAULT_PREDICTIONS_PATH = PROJECT_ROOT / "outputs" / "test_predictions.csv"
DEFAULT_CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrix.png"
DEFAULT_ERROR_DISTRIBUTION_PATH = (
    PROJECT_ROOT / "outputs" / "error_distribution.png"
)
DEFAULT_PR_CURVE_PATH = PROJECT_ROOT / "outputs" / "precision_recall_curve.png"
DEFAULT_RUN_SUMMARY_PATH = PROJECT_ROOT / "outputs" / "run_summary.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "model_card.md"
DEFAULT_REPORT_METRICS_PATH = PROJECT_ROOT / "reports" / "evaluation_summary.json"


def project_path(path: str | Path) -> Path:
    """Resolve relative artifact paths from the project root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
