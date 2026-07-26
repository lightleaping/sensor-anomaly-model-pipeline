# 다변량 센서 이상 탐지

**Normal-only AutoEncoder, Checkpoint Threshold, Evaluation, and FastAPI**

> 온도, 진동, 압력, 습도 센서 데이터를 전처리하고, 정상 데이터 기반 PyTorch AutoEncoder를 학습해 Reconstruction Error로 이상 여부를 판단하는 모델 파이프라인입니다.

<p>
  <img src="https://img.shields.io/badge/Python-3.11-3561D8?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-AutoEncoder-151F32?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/scikit--learn-StandardScaler-21AFC4?style=flat-square&logo=scikitlearn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-3%20Endpoints-2FA66A?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/GitHub%20Actions-E2E%20Pipeline-5F6675?style=flat-square&logo=githubactions&logoColor=white" alt="GitHub Actions">
</p>

---

## Why This Project

제조 설비는 온도, 진동, 압력, 습도처럼 단위와 범위가 다른 여러 센서값을 동시에 생성합니다. 하지만 실제 환경에서는 모든 고장 유형의 이상 데이터를 충분히 확보하기 어렵고, 학습 시점에 없던 새로운 조합의 이상도 발생할 수 있습니다.

이 프로젝트는 이상 유형을 직접 분류하는 대신 **정상 센서 조합을 학습**하고, 입력을 잘 복원하지 못할 때 발생하는 Reconstruction Error를 이상 신호로 사용합니다.

```text
Sensor Data
→ Split and Scaling
→ Normal-only Training
→ Validation Threshold
→ Held-out Test Evaluation
→ CLI and FastAPI Inference
```

모델 작성에서 끝내지 않고, Scaler와 Threshold를 Checkpoint와 연결하고, 동일한 Artifact를 CLI와 API가 사용하도록 구성했습니다.

---

## Project Overview

| 항목 | 내용 |
|---|---|
| **기간** | 2026.05 |
| **형태** | 개인 프로젝트 |
| **목표** | 다변량 센서 입력을 Reconstruction Error로 변환해 정상과 이상을 구분 |
| **범위** | 합성 데이터 생성, 데이터 검증, Train/Validation/Test 분리, Scaling, 정상 데이터 학습, Early Stopping, Threshold 설정, 평가, Model Card, CLI, FastAPI, pytest, GitHub Actions |
| **입력 Feature** | `temperature`, `vibration`, `pressure`, `humidity` |
| **모델** | PyTorch AutoEncoder, `4 → 8 → 2 → 8 → 4` |
| **판단 기준** | Validation 정상 Reconstruction Error의 95 Percentile |
| **기술** | Python, pandas, NumPy, scikit-learn, PyTorch, Matplotlib, FastAPI, Uvicorn, pytest, GitHub Actions |
| **중요 범위** | 시간 Window를 학습하는 Sequence 모델이 아니라 한 행의 센서 조합을 처리하는 다변량 이상 탐지 |

---

## Problem → Implementation → Result

| Problem | Implementation | Result |
|---|---|---|
| 모든 이상 유형을 학습 데이터로 확보하기 어려움 | 정상 Train 데이터만 사용해 AutoEncoder 학습 | 정상 패턴과 다른 센서 조합을 Error로 비교 |
| 센서마다 단위와 값의 범위가 다름 | Train 기준 `StandardScaler`를 Validation, Test, Inference에 재사용 | Feature Scale 차이가 Loss를 지배하는 문제 완화 |
| Test 결과에 맞춘 Threshold는 평가를 왜곡할 수 있음 | Validation 정상 Error의 95 Percentile 사용 | Threshold 결정과 Held-out Test 평가 분리 |
| 추론 코드의 Threshold가 학습 결과와 달라질 수 있음 | Threshold와 Feature 순서를 Model Checkpoint에 저장 | CLI와 API가 학습 시 결정한 기준을 자동 로드 |
| 전체 Error만으로 원인 Feature를 확인하기 어려움 | Feature별 Squared Error 반환 | 어떤 센서가 Error에 크게 기여했는지 확인 |
| 모델 평가가 한 번의 수동 실행에 의존할 수 있음 | Test, E2E Pipeline, Quality Gate, GitHub Actions 구성 | 회귀 검증과 전체 실행 경로 자동화 |

---

## System Overview

<img src="./docs/assets/sensor-system-overview.png" alt="다변량 센서 이상 탐지 시스템 구성도" width="100%">

### Responsibility Separation

| Layer | Responsibility |
|---|---|
| **Data Generation** | 정상 상관 구조와 5개 이상 유형을 가진 합성 센서 데이터 생성 |
| **Preprocessing** | Column 검증, 결측치 제거, Stratified Test Split, 정상 Train/Validation 분리, Scaling |
| **Training** | 정상 데이터 기반 AutoEncoder 학습과 Early Stopping |
| **Checkpoint** | Weight, Feature 순서, Threshold, Model Version, Training Metadata 저장 |
| **Evaluation** | Held-out Test의 Precision, Recall, F1, CM, PR Curve, 유형별 Recall 계산 |
| **Inference** | 저장된 Model, Scaler, Threshold로 단일 입력 판단 |
| **FastAPI** | Root, Health, Predict Endpoint와 입력 범위 검증 |
| **CI** | pytest와 End-to-End Pipeline 실행 |

---

## Practical Evaluation Criteria

이상 탐지는 Accuracy 하나보다 **미탐, 오탐, Threshold 독립성, Artifact 일관성, Drift 대응 범위**를 함께 확인해야 합니다.

| 실무 관점 | 평가 기준 | 프로젝트에서 확인한 근거 |
|---|---|---|
| **데이터 분리** | Test가 학습과 Threshold 결정에 사용되지 않는가 | Stratified Test Split 후 정상 Train과 Validation 분리 |
| **정상 패턴 학습** | AutoEncoder가 정상 데이터만 학습하는가 | `y_train_full == 0` Filter |
| **정규화 일관성** | 학습과 추론이 같은 Feature 순서와 Scaler를 사용하는가 | `FEATURE_COLUMNS` 검증, `scaler.pkl` 재사용 |
| **Threshold 독립성** | Test 결과를 보고 Threshold를 정하지 않는가 | Validation 정상 Error 95 Percentile |
| **Checkpoint 일관성** | Model과 Threshold가 같은 학습 결과에 묶이는가 | Checkpoint에 Weight, Threshold, Feature 순서 저장 |
| **미탐과 오탐** | Anomaly Recall과 Normal False Positive를 함께 보는가 | Precision, Recall, F1, Specificity, Confusion Matrix |
| **추론 설명성** | 어떤 Feature가 Error에 기여했는가 | `feature_errors` Response |
| **회귀 검증** | 전체 Pipeline이 자동 실행되는가 | pytest, Pipeline Smoke Prediction, GitHub Actions |
| **범위 설명** | 이상 신호를 고장 원인 진단으로 과장하지 않는가 | Synthetic, Row-level, Non-diagnostic 한계 명시 |

> Pipeline의 Recall과 F1 Quality Gate는 이 합성 데이터 Pipeline의 회귀 검증 기준입니다. 실제 설비의 운영 기준은 미탐과 오탐 비용, 운전 조건, Sensor Drift에 맞춰 별도로 정해야 합니다.

---

## Key Results

현재 저장소의 `reports/evaluation_summary.json`과 `reports/model_card.md`에 기록된 Held-out Test 결과입니다.

<!-- EVALUATION_RESULTS_START -->
| 항목 | 결과 |
|---|---:|
| Model Version | `sensor-ae-669717a0dab2` |
| Test Samples | **360** |
| Normal / Anomaly | 300 / 60 |
| Threshold | **0.394629** |
| Accuracy | **0.9500** |
| Balanced Accuracy | **0.9567** |
| Precision | **0.7838** |
| Recall | **0.9667** |
| F1 | **0.8657** |
| Specificity | **0.9467** |
| ROC AUC | **0.9893** |
| Average Precision | **0.9798** |
| Confusion Matrix | TN 284, FP 16, FN 2, TP 58 |
<!-- EVALUATION_RESULTS_END -->

### Reading the Result

- Anomaly 60개 중 58개를 탐지해 Recall은 96.67%입니다.
- False Negative는 2개, False Positive는 16개입니다.
- Precision 78.38%는 이상 경보 중 일부가 정상 Sample을 포함한다는 뜻입니다.
- 이 결과는 합성 데이터에 대한 결과이며 실제 설비 성능으로 일반화할 수 없습니다.

### Anomaly Type Detection

| Anomaly Type | Test | Detected | Recall |
|---|---:|---:|---:|
| `combined_fault` | 12 | 10 | 0.8333 |
| `leak` | 12 | 12 | 1.0000 |
| `mechanical_fault` | 12 | 12 | 1.0000 |
| `pressure_event` | 12 | 12 | 1.0000 |
| `thermal_overload` | 12 | 12 | 1.0000 |

`combined_fault`에서 2개를 놓쳤으며, 다른 4개 합성 이상 유형은 해당 Test Split에서 모두 탐지했습니다.

---

## Data Pipeline

### Default Synthetic Data

| Class | Default Count | 구성 |
|---|---:|---|
| Normal | 1,500 | 부하와 주변 조건을 반영한 센서 상관 구조 |
| Anomaly | 300 | 5개 이상 유형을 균등하게 섞은 센서 변형 |

Anomaly Types:

```text
thermal_overload
mechanical_fault
pressure_event
leak
combined_fault
```

합성 데이터는 Pipeline 검증을 위한 데이터입니다. 실제 공정의 고장 확률이나 Sensor 분포를 재현한다고 해석하지 않습니다.

### Preprocessing Order

```text
CSV Load
→ Required Column Validation
→ Numeric Conversion
→ NaN and Inf Removal
→ Stratified Test Split
→ Normal Train Filtering
→ Train and Validation Split
→ StandardScaler Fit on Train
→ Transform Validation, Test, Inference
```

Default Split:

```text
Test = 전체 데이터의 20%
Validation = 정상 Train 후보의 20%
Training = 남은 정상 데이터
```

---

## Model and Threshold

### AutoEncoder Architecture

```text
Input 4
→ Linear 8
→ LeakyReLU
→ Latent 2
→ Linear 8
→ LeakyReLU
→ Output 4
```

### Default Training Configuration

| Item | Value |
|---|---:|
| Requested Epochs | 120 |
| Batch Size | 32 |
| Learning Rate | 0.001 |
| Optimizer | Adam |
| Loss | Mean Squared Error |
| Early Stopping Patience | 20 |
| Threshold Percentile | 95 |
| Random Seed | 42 |
| Deterministic Algorithms | Enabled |

학습 중 Validation Loss가 가장 낮은 Weight를 복원한 뒤, 정상 Validation Sample의 Reconstruction Error 95 Percentile을 Threshold로 계산합니다.

---

## Anomaly Decision Flow

<img src="./docs/assets/sensor-anomaly-decision-flow.png" alt="다변량 센서 이상 판단 흐름" width="100%">

```text
reconstruction_error > threshold  → anomaly
reconstruction_error <= threshold → normal
```

현재 Threshold는 상수가 아니라 학습 Checkpoint에 저장됩니다. `AnomalyPredictor`는 Checkpoint의 Model Weight, Threshold, Feature 순서와 저장된 Scaler를 함께 검증해 로드합니다.

Response에는 다음 정보가 포함됩니다.

```text
prediction
reconstruction_error
threshold
error_margin
feature_errors
model_version
input
```

---

## Technical Details

<details open>
<summary><b>01 | Artifacts and Model Version</b></summary>

<br>

| Artifact | Purpose |
|---|---|
| `models/autoencoder.pt` | Weight, Threshold, Feature 순서, Model Version |
| `models/scaler.pkl` | Train 기준 StandardScaler |
| `models/model_metadata.json` | Checkpoint Metadata |
| `outputs/preprocessed_data.npz` | Train, Validation, Test Array와 Index |
| `outputs/preprocessing_metadata.json` | Row Count, Split, Data Hash |
| `outputs/train_history.csv` | Epoch별 Train과 Validation Loss |
| `outputs/training_curve.png` | 학습 곡선 |
| `outputs/evaluation_metrics.json` | 전체 평가 지표 |
| `outputs/test_predictions.csv` | Sample별 Label, Prediction, Error |
| `outputs/confusion_matrix.png` | Normal과 Anomaly Confusion Matrix |
| `outputs/error_distribution.png` | Class별 Reconstruction Error 분포 |
| `outputs/precision_recall_curve.png` | Precision Recall Curve |
| `outputs/run_summary.json` | Pipeline, Quality Gate, Smoke Prediction 결과 |
| `reports/model_card.md` | Model, 성능, Intended Use, Limitations |
| `reports/evaluation_summary.json` | 공개용 Machine-readable 평가 결과 |

Model Version은 Best Weight의 State Fingerprint로 생성됩니다.

</details>

<details>
<summary><b>02 | Quality Gate and Smoke Inference</b></summary>

<br>

기본 Pipeline Quality Gate:

```text
Recall >= 0.85
F1 >= 0.80
Normal Smoke Sample → normal
Anomaly Smoke Sample → anomaly
```

GitHub Actions에서는 실행 시간을 고려해 다음 조건으로 End-to-End Pipeline을 검증합니다.

```text
Normal 800
Anomaly 160
Epochs 80
Minimum Recall 0.75
Minimum F1 0.70
```

이 Threshold는 CI 회귀 검증 기준이며 실제 설비 운영 기준이 아닙니다.

</details>

<details>
<summary><b>03 | Validation</b></summary>

<br>

현재 공개 Test:

```text
tests/test_integration.py    3 cases
tests/test_model.py          3 cases
```

검증 범위:

- Training Artifact와 Evaluation Metric 생성
- Predictor가 Checkpoint Threshold 사용
- 정상과 이상 Smoke Input 비교
- Feature Error 반환
- `/health`, `/predict`, Pydantic Validation
- AutoEncoder Output와 Reconstruction Error Shape
- 잘못된 Tensor Shape 예외 처리

</details>

---

## FastAPI

| Method | Endpoint | Role |
|---|---|---|
| `GET` | `/` | Service, Model Version, Endpoint 안내 |
| `GET` | `/health` | Model Load 상태와 Threshold 확인 |
| `POST` | `/predict` | 4개 센서값의 이상 여부와 Feature Error 반환 |

### Request

```json
{
  "temperature": 30.0,
  "vibration": 0.35,
  "pressure": 100.0,
  "humidity": 45.0
}
```

### Response Structure

```json
{
  "prediction": "normal",
  "reconstruction_error": 0.182315,
  "threshold": 0.394629,
  "error_margin": -0.212314,
  "feature_errors": {
    "temperature": 0.041201,
    "vibration": 0.014210,
    "pressure": 0.581002,
    "humidity": 0.092847
  },
  "model_version": "sensor-ae-669717a0dab2",
  "input": {
    "temperature": 30.0,
    "vibration": 0.35,
    "pressure": 100.0,
    "humidity": 45.0
  }
}
```

> Response의 Error 값은 구조 예시입니다. 실제 값은 입력과 저장된 Model Artifact에 따라 달라집니다.

Input Validation:

```text
temperature: -100 to 300
vibration: 0 to 100
pressure: 0 to 1000
humidity: 0 to 100
```

---

## Run and Verify

### 1. Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r .\requirements.txt
```

### 2. Full Pipeline

```powershell
python -m src.pipeline
```

또는:

```powershell
.\scripts\run_pipeline.ps1
```

### 3. CLI Inference

```powershell
python -m src.predict `
  --temperature 30 `
  --vibration 0.35 `
  --pressure 100 `
  --humidity 45
```

### 4. FastAPI

```powershell
python -m uvicorn src.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

> API는 시작 시 Model과 Scaler를 로드합니다. Fresh Clone에서는 먼저 `python -m src.pipeline`을 실행해야 합니다.

### 5. Tests

```powershell
python -m pytest -q
```

---

## Project Structure

```text
sensor-anomaly-model-pipeline/
├── .github/
│   └── workflows/
│       └── ci.yml
├── docs/
│   └── assets/
├── reports/
│   ├── evaluation_summary.json
│   └── model_card.md
├── scripts/
│   └── run_pipeline.ps1
├── src/
│   ├── app.py
│   ├── artifacts.py
│   ├── config.py
│   ├── evaluate.py
│   ├── generate_data.py
│   ├── model.py
│   ├── pipeline.py
│   ├── predict.py
│   ├── preprocess.py
│   └── train.py
├── tests/
│   ├── test_integration.py
│   └── test_model.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

Pipeline 실행 시 생성:

```text
data/
models/
outputs/
```

---

## Current Scope and Limitations

### Current Scope

- 합성 다변량 Sensor Data
- 5개 합성 이상 유형
- 한 행 단위 4개 Feature 입력
- 정상 데이터 기반 AutoEncoder
- Validation Percentile Threshold
- Checkpoint 기반 Model, Scaler, Threshold 연결
- Held-out Test와 유형별 Recall
- CLI와 FastAPI 추론
- Model Card와 Machine-readable Summary
- pytest와 GitHub Actions E2E Pipeline

### Limitations

- 실제 제조 설비 데이터가 아님
- 시간 순서를 가진 Sequence Window를 사용하지 않음
- 설비 상태 전환과 장기 추세를 학습하지 않음
- 합성 이상 유형과 실제 고장 원인의 대응을 보장하지 않음
- Feature Error는 Reconstruction Error 분해이며 인과 설명이 아님
- Sensor 교체, 운전 조건 변화, Data Drift 발생 시 Threshold 재보정 필요
- 이상 판정은 고장 진단이 아니라 점검이 필요한 운영 신호임
- 인증, Request Logging, Monitoring, Model Registry는 범위 밖

### Next Steps

1. 실제 또는 공개 설비 Sensor Dataset으로 재학습
2. Window 기반 LSTM AutoEncoder, 1D CNN과 비교
3. 운전 Mode별 Model 또는 Conditional Threshold 검토
4. 비용 기반 Threshold와 Alert Suppression 정책 설계
5. Data Drift와 Threshold 재보정 기준 추가
6. Model Registry, Request Log, Monitoring 추가
7. Batch Inference와 History Dashboard 구현

---

## What This Project Demonstrates

- 다변량 Sensor Data 생성과 전처리 경험
- Train, Validation, Test 역할 분리 경험
- 정상 데이터 기반 AutoEncoder 학습 경험
- Early Stopping과 Validation Percentile Threshold 적용 경험
- Model, Scaler, Threshold, Feature 순서를 Artifact로 연결한 경험
- Precision, Recall, F1, Confusion Matrix, PR Curve 기반 평가 경험
- 유형별 이상 탐지 결과와 실패 사례를 구분한 경험
- CLI와 FastAPI Model Serving 경험
- pytest와 GitHub Actions로 End-to-End Pipeline을 검증한 경험
- 다변량 Row Input과 Sequence 기반 Time Series 모델의 차이를 구분한 경험

---

## Contact

- Developer: 김수진
- GitHub: https://github.com/lightleaping
- Email: workingskyroad@gmail.com
