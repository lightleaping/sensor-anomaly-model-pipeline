# Sensor Autoencoder Model Card

## Model

- Version: `sensor-ae-669717a0dab2`
- Architecture: `4 → 8 → 2 → 8 → 4`
- Features: temperature, vibration, pressure, humidity
- Training data: normal samples only
- Decision rule: reconstruction error > validation 95.0th percentile
- Threshold: `0.394629`

## Held-out test results

| Metric | Value |
|---|---:|
| Accuracy | 0.9500 |
| Precision | 0.7838 |
| Recall | 0.9667 |
| F1 | 0.8657 |
| Balanced accuracy | 0.9567 |
| Specificity | 0.9467 |
| ROC AUC | 0.9893 |
| Average precision | 0.9798 |
| Matthews correlation coefficient | 0.8423 |

Confusion matrix: TN=284, FP=16, FN=2, TP=58.

## Anomaly-type detection

| Anomaly type | Support | Detected | Recall | Mean error |
|---|---:|---:|---:|---:|
| `combined_fault` | 12 | 10 | 0.8333 | 1.7771 |
| `leak` | 12 | 12 | 1.0000 | 9.1958 |
| `mechanical_fault` | 12 | 12 | 1.0000 | 19.1581 |
| `pressure_event` | 12 | 12 | 1.0000 | 10.7744 |
| `thermal_overload` | 12 | 12 | 1.0000 | 11.2141 |

The machine-readable evaluation summary is stored in `reports/evaluation_summary.json`.

## Intended use

This model is a reproducible row-level baseline for detecting unusual combinations of temperature, vibration, pressure, and humidity. It is suitable for demos and pipeline verification.

## Limitations

- The checked-in evaluation is based on synthetic data, not a production machine.
- The model does not use temporal windows, so it cannot learn trend or sequence anomalies.
- The threshold must be recalibrated after sensor replacement, operating-regime changes, or data drift.
- A prediction is an operational signal, not a diagnosis of equipment failure.
