# Sensor Autoencoder Model Card

## Model

- Version: `sensor-ae-26732b4bb7a1`
- Architecture: `4 → 8 → 2 → 8 → 4`
- Features: temperature, vibration, pressure, humidity
- Training data: normal samples only
- Decision rule: reconstruction error > validation 95.0th percentile
- Threshold: `0.411495`

## Held-out test results

| Metric | Value |
|---|---:|
| Accuracy | 0.9333 |
| Precision | 0.7571 |
| Recall | 0.8833 |
| F1 | 0.8154 |
| Specificity | 0.9433 |
| ROC AUC | 0.9640 |
| Average precision | 0.9351 |

Confusion matrix: TN=283, FP=17, FN=7, TP=53.

The machine-readable evaluation summary is stored in `reports/evaluation_summary.json`.

## Intended use

This model is a reproducible row-level baseline for detecting unusual combinations of temperature, vibration, pressure, and humidity. It is suitable for demos and pipeline verification.

## Limitations

- The checked-in evaluation is based on synthetic data, not a production machine.
- The model does not use temporal windows, so it cannot learn trend or sequence anomalies.
- The threshold must be recalibrated after sensor replacement, operating-regime changes, or data drift.
- A prediction is an operational signal, not a diagnosis of equipment failure.
