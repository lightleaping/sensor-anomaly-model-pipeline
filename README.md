# 다변량 센서 이상 탐지 파이프라인

> 온도·진동·압력·습도 데이터를 생성하고, 정상 데이터만으로 PyTorch Autoencoder를 학습한 뒤 평가·CLI·FastAPI 추론까지 한 번에 재현하는 프로젝트입니다.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Autoencoder-EE4C2C?logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![CI](https://github.com/lightleaping/sensor-anomaly-model-pipeline/actions/workflows/ci.yml/badge.svg)

![센서 이상 탐지 시스템 구조](docs/assets/sensor-anomaly-architecture.svg)

## 프로젝트 목표

실제 이상 데이터는 정상 데이터보다 적고, 사전에 정의하지 못한 고장 유형도 발생할 수 있습니다. 이 프로젝트는 정상 상태의 센서 관계를 Autoencoder가 복원하도록 학습하고, 입력의 복원 오차가 정상 Validation 분포에서 정한 Threshold를 넘으면 이상으로 판정합니다.

```text
Synthetic sensor data
  → validation / cleaning
  → train / validation / test split
  → train-only StandardScaler
  → normal-only Autoencoder training
  → validation 95th-percentile threshold
  → held-out test evaluation
  → CLI / FastAPI inference
```

핵심은 모델 파일만 만드는 데서 끝나지 않는 것입니다. Feature 순서, Scaler, Threshold, 모델 버전을 하나의 실행 흐름으로 묶고 테스트와 품질 게이트로 검증합니다.

## 구현 완료 범위

| 영역 | 구현 |
|---|---|
| 데이터 | 재현 가능한 정상 데이터와 열 과부하·기계 결함·압력 이벤트·누수·복합 결함 생성 |
| 전처리 | 필수 열·숫자·유한값·Binary Label 검증, Stratified Test Split |
| 누수 방지 | 정상 Train에서만 Scaler Fit, 정상 Validation에서만 Threshold 산출 |
| 학습 | PyTorch Autoencoder, Seed 고정, 조기 종료, Best Weight 복원 |
| Artifact | Model State, Feature 순서, Threshold, Model Version, 학습 설정 저장 |
| 평가 | Accuracy, Precision, Recall, F1, Specificity, ROC AUC, Average Precision |
| 시각화 | 학습 곡선, Confusion Matrix, Error 분포, Precision-Recall Curve |
| 추론 | 단일 샘플 CLI, Feature별 Error 기여도 |
| API | FastAPI `/health`, `/predict`, Swagger UI, 입력 범위 검증 |
| 자동화 | 한 명령 End-to-End Pipeline, Smoke Test, Recall/F1 품질 게이트 |
| 테스트·CI | 독립 임시 학습 기반 Pytest와 GitHub Actions |

최신 실측 평가는 [Model Card](reports/model_card.md)와 [JSON Summary](reports/evaluation_summary.json)에 저장됩니다. 이 파일들은 실제 `python -m src.pipeline` 실행 결과로 갱신하며 임의 수치를 기록하지 않습니다.

### 검증된 기본 실행 결과

Python 3.11.9, Seed 42, 정상 1,500건·이상 300건으로 전체 파이프라인을 실행한 결과입니다.

| Metric | Result |
|---|---:|
| Accuracy | 0.9333 |
| Precision | 0.7571 |
| Recall | 0.8833 |
| F1 | 0.8154 |
| Specificity | 0.9433 |
| ROC AUC | 0.9640 |
| Average Precision | 0.9351 |

Confusion Matrix는 TN 283, FP 17, FN 7, TP 53이며, Validation 95 Percentile Threshold는 `0.411495`입니다.

## 빠른 실행

### 1. 환경 구성

Python 3.11을 기준 런타임으로 사용합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

macOS 또는 Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 전체 파이프라인 실행

```powershell
python -m src.pipeline
```

Windows에서는 다음 Wrapper도 사용할 수 있습니다.

```powershell
.\scripts\run_pipeline.ps1
```

이 명령은 다음 작업을 순서대로 수행합니다.

1. 1,500개 정상 Sample과 300개 이상 Sample 생성
2. 데이터 검증·분할·Scaling
3. 정상 Sample만 사용해 최대 120 Epoch 학습
4. Validation 정상 Error의 95 Percentile Threshold 저장
5. Held-out Test 평가와 그래프 생성
6. 정상·극단 이상 입력 Smoke Test
7. 기본 Recall 0.85, F1 0.80 품질 게이트 확인

개발 중 빠른 실행 예시:

```powershell
python -m src.pipeline `
  --normal-count 800 `
  --anomaly-count 160 `
  --epochs 80 `
  --minimum-recall 0.75 `
  --minimum-f1 0.70
```

## 생성되는 Artifact

| 경로 | 내용 |
|---|---|
| `data/sensor_data.csv` | 생성한 센서 데이터와 이상 유형 |
| `outputs/preprocessed_data.npz` | Scaling된 Train·Validation·Test Split |
| `outputs/preprocessing_metadata.json` | 데이터 Hash, 사용 행 수, Split 정보 |
| `models/scaler.pkl` | Train 기준 StandardScaler |
| `models/autoencoder.pt` | Weight, Threshold, Feature 순서, Model Version |
| `models/model_metadata.json` | 사람이 확인할 수 있는 Checkpoint Metadata |
| `outputs/train_history.csv` | Epoch별 Train·Validation Loss |
| `outputs/training_curve.png` | 학습 곡선 |
| `outputs/evaluation_metrics.json` | 전체 평가 지표 |
| `outputs/test_predictions.csv` | Test 입력별 Label, Error, 판정 |
| `outputs/confusion_matrix.png` | Confusion Matrix |
| `outputs/error_distribution.png` | 정상·이상 Error 분포와 Threshold |
| `outputs/precision_recall_curve.png` | Precision-Recall Curve |
| `outputs/run_summary.json` | 전체 실행·품질 게이트·Smoke Test 결과 |
| `reports/model_card.md` | 공개 가능한 최신 평가 요약과 한계 |

원본 데이터·Checkpoint·대용량 실행 산출물은 Git에서 제외하고, 재현 가능한 코드와 경량 평가 보고서는 저장소에 유지합니다.

## CLI 추론

정상에 가까운 예:

```powershell
python -m src.predict `
  --temperature 30 `
  --vibration 0.35 `
  --pressure 100 `
  --humidity 45
```

이상 예:

```powershell
python -m src.predict `
  --temperature 55 `
  --vibration 1.6 `
  --pressure 135 `
  --humidity 80
```

CLI와 API는 Threshold를 상수로 복사하지 않습니다. 학습 Checkpoint에 저장된 Threshold와 Feature 순서를 직접 읽으므로 재학습 후에도 같은 판별 기준을 사용합니다.

## FastAPI

먼저 전체 파이프라인으로 모델을 생성한 뒤 서버를 시작합니다.

```powershell
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health Check: `http://127.0.0.1:8000/health`

PowerShell 요청 예:

```powershell
$body = @{
  temperature = 30
  vibration = 0.35
  pressure = 100
  humidity = 45
} | ConvertTo-Json

Invoke-RestMethod `
  -Method Post `
  -Uri http://127.0.0.1:8000/predict `
  -ContentType "application/json" `
  -Body $body
```

응답 구조:

```json
{
  "prediction": "normal",
  "reconstruction_error": 0.123456,
  "threshold": 1.234567,
  "error_margin": -1.111111,
  "feature_errors": {
    "temperature": 0.1,
    "vibration": 0.2,
    "pressure": 0.1,
    "humidity": 0.09
  },
  "model_version": "sensor-ae-...",
  "input": {
    "temperature": 30.0,
    "vibration": 0.35,
    "pressure": 100.0,
    "humidity": 45.0
  }
}
```

위 숫자는 응답 형식 예시입니다. 실제 값은 생성된 Artifact를 사용합니다.

## 단계별 실행

문제 확인이나 실험을 위해 각 단계를 따로 실행할 수 있습니다.

```powershell
python -m src.generate_data
python -m src.preprocess
python -m src.train
python -m src.evaluate
```

주요 학습 옵션:

```powershell
python -m src.train `
  --epochs 160 `
  --batch-size 64 `
  --lr 0.001 `
  --latent-dim 2 `
  --hidden-dim 8 `
  --threshold-percentile 95 `
  --patience 25
```

Threshold를 운영 기준으로 임시 재평가할 때만 다음 Override를 사용합니다. 기본 추론은 Checkpoint Threshold를 사용합니다.

```powershell
python -m src.evaluate --threshold 1.5
python -m src.predict <센서 인자> --threshold 1.5
```

## 모델과 평가 설계

### 데이터 역할 분리

- Train: 정상 Sample만 사용해 Weight와 Scaler 학습
- Validation: 정상 Sample만 사용해 조기 종료와 Threshold 산출
- Test: 정상·이상 Sample을 함께 사용해 최종 성능 측정

Test Label을 Threshold 선택에 사용하지 않으므로 평가 데이터에 맞춘 기준값 조정을 방지합니다.

### 모델

기본 구조는 `4 → 8 → 2 → 8 → 4` Dense Autoencoder입니다. 현재 입력은 시간 Window가 아닌 한 시점의 네 Feature이므로 작은 Dense Baseline을 사용합니다.

```text
reconstruction_error = mean((scaled_input - reconstruction)²)

error > checkpoint.threshold  → anomaly
error <= checkpoint.threshold → normal
```

### 평가

Class 불균형에서 Accuracy만으로 성능을 판단하지 않습니다.

- Recall: 실제 이상을 얼마나 놓치지 않았는지
- Precision: 이상 경보 중 실제 이상의 비율
- F1: Precision과 Recall의 균형
- Specificity / False Positive Rate: 정상 오탐 비용
- ROC AUC / Average Precision: 연속 Error Score의 분리력
- Confusion Matrix: TN·FP·FN·TP의 실제 개수

Pipeline은 Recall·F1이 설정한 하한보다 낮거나 대표 Smoke Sample을 잘못 분류하면 실패 상태를 기록하고 종료 코드 1을 반환합니다.

## 테스트

```powershell
python -m pytest -q
```

테스트는 저장소의 기존 모델 파일에 의존하지 않습니다. 임시 디렉터리에 데이터를 만들고 별도 모델을 학습하여 다음을 검증합니다.

- Model Tensor Shape와 Reconstruction Error
- 데이터 생성·전처리·학습·평가 Artifact
- Checkpoint Threshold와 추론 Threshold 일치
- 정상·이상 단일 입력 추론
- `/health`, `/predict`, 잘못된 API 입력의 `422` 응답

## 프로젝트 구조

```text
sensor-anomaly-model-pipeline/
├─ .github/workflows/ci.yml
├─ docs/assets/
├─ reports/
│  ├─ model_card.md
│  └─ evaluation_summary.json
├─ scripts/
│  └─ run_pipeline.ps1
├─ src/
│  ├─ artifacts.py
│  ├─ config.py
│  ├─ generate_data.py
│  ├─ preprocess.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  ├─ app.py
│  └─ pipeline.py
├─ tests/
├─ pyproject.toml
├─ requirements.txt
└─ README.md
```

## 운영 적용 전 확인할 점

현재 프로젝트는 실행 가능한 Row-level Baseline이지만 실제 설비 모델이라고 간주해서는 안 됩니다.

- 실제 센서의 허용 범위, 단위, Sampling 주기를 명시해야 합니다.
- 시간 의존 이상에는 Sliding Window와 LSTM/1D CNN/Transformer 비교가 필요합니다.
- 설비 상태·운전 모드별 정상 분포가 다르면 조건별 모델 또는 Feature가 필요합니다.
- Drift Monitoring과 정기 Threshold 재보정이 필요합니다.
- 경보 비용에 맞춰 Recall과 False Positive Rate의 목표를 합의해야 합니다.
- Checkpoint와 Pickle Artifact는 신뢰할 수 있는 저장소에서만 로드해야 합니다.

## 저장소

- GitHub: [lightleaping/sensor-anomaly-model-pipeline](https://github.com/lightleaping/sensor-anomaly-model-pipeline)
- Developer: 김수진
