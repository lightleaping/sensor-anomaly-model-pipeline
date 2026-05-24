# Sensor Anomaly Detection Model Pipeline

PyTorch AutoEncoder 기반 센서 데이터 이상탐지 모델 개발 및 평가 파이프라인입니다.

## 1. Project Overview

이 프로젝트는 센서 데이터를 전처리하고, 정상 패턴을 학습한 AutoEncoder 모델을 이용해 이상 데이터를 탐지하는 AI 모델 개발 프로젝트입니다.

데이터플로 AI 모델 개발자 직무의 주요 요구사항인 데이터 전처리, AI 모델 설계 및 학습, 성능 평가, 추론 API, 실험 결과 문서화를 보여주기 위해 구성했습니다.

## 2. Key Features

- 센서 데이터 샘플 생성
- 결측치 처리 및 StandardScaler 정규화
- 정상 데이터 기반 PyTorch AutoEncoder 학습
- Reconstruction Error 기반 이상탐지
- Validation Error 기반 Threshold 설정
- Precision, Recall, F1-score 평가
- Confusion Matrix 시각화
- 단일 샘플 추론 CLI
- FastAPI 기반 `/predict` 추론 API

## 3. Tech Stack

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- FastAPI
- Uvicorn

## 4. Project Structure

```text
sensor-anomaly-model-pipeline/
├─ data/
│  └─ sensor_data.csv
├─ models/
│  ├─ autoencoder.pt
│  └─ scaler.pkl
├─ outputs/
│  ├─ preprocessed_data.npz
│  ├─ train_history.csv
│  ├─ evaluation_metrics.csv
│  ├─ test_predictions.csv
│  └─ confusion_matrix.png
├─ reports/
├─ src/
│  ├─ generate_data.py
│  ├─ preprocess.py
│  ├─ model.py
│  ├─ train.py
│  ├─ evaluate.py
│  ├─ predict.py
│  └─ app.py
├─ README.md
├─ requirements.txt
└─ .gitignore
```

## 5. Dataset

샘플 센서 데이터는 프로젝트 내부에서 생성합니다.

사용한 feature는 다음과 같습니다.

| Feature | Description |
|---|---|
| temperature | 센서 온도 |
| vibration | 진동 수치 |
| pressure | 압력 수치 |
| humidity | 습도 수치 |
| label | 정상 0, 이상 1 |

생성 데이터 구성:

| Class | Count |
|---|---:|
| Normal | 1000 |
| Anomaly | 120 |
| Total | 1120 |

## 6. Modeling Approach

AutoEncoder는 입력 데이터를 압축한 뒤 다시 복원하는 모델입니다.

이 프로젝트에서는 정상 데이터만 사용해 AutoEncoder를 학습시킨 뒤, 테스트 데이터의 reconstruction error를 계산했습니다.

정상 패턴과 다른 데이터는 복원 오차가 커지는 경향이 있으므로, validation error의 95 percentile 값을 threshold로 사용해 이상 여부를 판단했습니다.

## 7. Evaluation Result

테스트 데이터 구성:

| Class | Count |
|---|---:|
| Normal | 200 |
| Anomaly | 24 |
| Total | 224 |

평가 결과:

| Metric | Score |
|---|---:|
| Precision | 0.8000 |
| Recall | 1.0000 |
| F1-score | 0.8889 |
| Accuracy | 0.9700 |
| Threshold | 1.569105 |

해석:

- 이상 데이터를 놓치지 않아 Recall이 1.0으로 나타났습니다.
- 일부 정상 데이터를 이상으로 판단하여 Precision은 0.8로 나타났습니다.
- 이상탐지 문제에서는 이상을 놓치지 않는 것이 중요하므로, 현재 모델은 recall 중심의 탐지 목적에 적합한 결과를 보였습니다.

## 8. How to Run

### 8-1. Install

```bash
python -m venv .venv
.venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

### 8-2. Generate Sample Data

```bash
python src/generate_data.py
```

### 8-3. Preprocess Data

```bash
python src/preprocess.py
```

### 8-4. Train Model

```bash
python src/train.py
```

### 8-5. Evaluate Model

```bash
python src/evaluate.py
```

### 8-6. Predict Single Sample

Normal sample:

```bash
python src/predict.py --temperature 30 --vibration 0.35 --pressure 100 --humidity 45
```

Anomaly sample:

```bash
python src/predict.py --temperature 45 --vibration 0.9 --pressure 118 --humidity 70
```

## 9. FastAPI Inference

Run API server:

```bash
uvicorn src.app:app --reload
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

Prediction endpoint:

```text
POST /predict
```

Request example:

```json
{
  "temperature": 45,
  "vibration": 0.9,
  "pressure": 118,
  "humidity": 70
}
```

Response example:

```json
{
  "prediction": "anomaly",
  "reconstruction_error": 5.62816,
  "threshold": 1.569105,
  "input": {
    "temperature": 45,
    "vibration": 0.9,
    "pressure": 118,
    "humidity": 70
  }
}
```

## 10. What I Learned

이 프로젝트를 통해 데이터 생성, 전처리, 모델 학습, 평가, 추론 API까지 AI 모델 개발의 기본 흐름을 하나의 파이프라인으로 구성했습니다.

특히 단순히 모델을 학습하는 데서 끝내지 않고, validation error 기반 threshold 설정, precision/recall/F1-score 평가, confusion matrix 저장, FastAPI 추론 endpoint까지 연결하며 모델 개발 결과를 서비스 적용 형태로 확장하는 과정을 경험했습니다.

## 11. Limitations and Future Work

현재 프로젝트는 샘플 데이터를 기반으로 하므로 실제 산업 센서 데이터의 복잡성을 모두 반영하지는 못합니다.

향후 개선 방향은 다음과 같습니다.

- 실제 센서 데이터셋 적용
- feature engineering 추가
- threshold 튜닝 실험
- 모델 구조 비교
- 학습 로그 시각화
- Docker 기반 실행 환경 구성
- API 요청 로그 저장
