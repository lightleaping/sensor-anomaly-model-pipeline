# Sensor Anomaly Model Pipeline

PyTorch AutoEncoder 기반 센서 데이터 이상탐지 모델 개발 및 FastAPI 추론 API 프로젝트입니다.

센서 데이터를 생성하고 전처리한 뒤, 정상 데이터 기반 AutoEncoder를 학습하여 reconstruction error로 이상 여부를 판단합니다.  
모델 학습에서 끝내지 않고, 평가 지표와 FastAPI `/predict` API까지 연결해 모델 개발 결과를 추론 서비스 형태로 확인할 수 있도록 구성했습니다.

---

## 1. Overview

이 프로젝트는 온도, 진동, 압력, 습도 센서 데이터를 기반으로 이상 상태를 탐지하는 AI 모델 개발 파이프라인입니다.

전체 흐름은 다음과 같습니다.

```text
센서 데이터 생성
→ 데이터 전처리
→ 정상 데이터 기반 PyTorch AutoEncoder 학습
→ reconstruction error 계산
→ validation error 기반 threshold 설정
→ precision / recall / F1-score 평가
→ 단일 샘플 CLI 추론
→ FastAPI /predict API 응답
```

---

## 2. Why I Built This

AI 모델 개발은 모델 구조를 작성하는 것만으로 완성되지 않는다고 생각했습니다.

실제로 활용 가능한 AI 기능이 되기 위해서는 다음 흐름이 함께 연결되어야 합니다.

- 데이터 생성 및 전처리
- 학습 / 검증 / 테스트 데이터 분리
- 모델 학습
- 성능 평가
- 단일 샘플 추론
- API 응답 구조
- 실험 결과 문서화

이 프로젝트에서는 센서 이상탐지 문제를 대상으로 위 과정을 하나의 파이프라인으로 구성했습니다.

---

## 3. Key Features

- 센서 샘플 데이터 생성
- 결측치 처리
- feature / label 분리
- train / validation / test split
- `StandardScaler` 기반 정규화
- 정상 데이터 기반 PyTorch AutoEncoder 학습
- reconstruction error 계산
- validation reconstruction error 기반 threshold 설정
- precision, recall, F1-score, accuracy 평가
- confusion matrix 저장
- 단일 샘플 CLI 추론
- FastAPI `/predict` endpoint 제공
- Swagger UI 기반 API 테스트

---

## 4. Project Structure

```text
sensor-anomaly-model-pipeline/
├─ src/
│  ├─ generate_data.py   # 센서 샘플 데이터 생성
│  ├─ preprocess.py      # 데이터 전처리, split, scaling
│  ├─ model.py           # PyTorch AutoEncoder 모델 정의
│  ├─ train.py           # 모델 학습
│  ├─ evaluate.py        # reconstruction error 기반 평가
│  ├─ predict.py         # 단일 샘플 CLI 추론
│  └─ app.py             # FastAPI /predict API
├─ data/                 # 생성 데이터 및 전처리 결과
├─ models/               # 학습된 모델, scaler, threshold 저장
├─ outputs/              # 평가 결과, confusion matrix 등 결과물
├─ requirements.txt
└─ README.md
```

---

## 5. Architecture

### Training & Evaluation Flow

```text
센서 데이터 생성
→ CSV 로드
→ 결측치 처리
→ feature / label 분리
→ train / validation / test 분리
→ 정상 데이터만 학습 데이터로 사용
→ StandardScaler 정규화
→ PyTorch AutoEncoder 학습
→ validation error 기반 threshold 설정
→ test reconstruction error 계산
→ precision / recall / F1-score 평가
→ 평가 결과 저장
```

### Inference API Flow

```text
센서 입력값 수신
→ scaler 로드
→ 입력값 정규화
→ AutoEncoder 모델 로드
→ reconstruction error 계산
→ threshold와 비교
→ normal / anomaly 예측
→ FastAPI /predict JSON 응답 반환
```

---

## 6. Tech Stack

- Python
- PyTorch
- pandas
- NumPy
- scikit-learn
- matplotlib
- FastAPI
- Uvicorn

---

## 7. Model Approach

이상탐지에서는 이상 데이터가 충분하지 않은 경우가 많기 때문에, 정상 데이터의 패턴을 먼저 학습하는 AutoEncoder 방식을 사용했습니다.

AutoEncoder는 입력 데이터를 압축한 뒤 다시 복원하는 구조입니다.  
정상 데이터와 유사한 입력은 복원 오차가 작고, 정상 패턴과 다른 입력은 복원 오차가 커질 수 있습니다.

이 프로젝트에서는 각 샘플의 reconstruction error를 계산한 뒤, validation reconstruction error의 95 percentile 값을 threshold로 사용했습니다.

```text
reconstruction_error > threshold → anomaly
reconstruction_error <= threshold → normal
```

---

## 8. Evaluation

모델 평가는 다음 지표를 기준으로 확인했습니다.

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

이상탐지에서는 이상 데이터를 놓치지 않는 것이 중요하기 때문에 recall을 특히 중요하게 확인했습니다.

포트폴리오 기준 실험 결과에서는 샘플 테스트 데이터에서 anomaly recall 1.0을 확인했으며, 일부 정상 데이터를 이상으로 판단한 경우가 있었지만 이상 데이터를 놓치지 않는 방향의 탐지 결과로 해석했습니다.

---

## 9. How to Run

### 1) Create and activate virtual environment

Windows PowerShell 기준:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

### 2) Install dependencies

```powershell
pip install -r requirements.txt
```

실행 확인 결과, 아래 주요 패키지가 정상 설치되어 있음을 확인했습니다.

- pandas
- numpy
- scikit-learn
- matplotlib
- torch
- fastapi
- uvicorn

### 3) Generate sample data

```powershell
python src/generate_data.py
```

### 4) Preprocess data

```powershell
python src/preprocess.py
```

### 5) Train model

```powershell
python src/train.py
```

### 6) Evaluate model

```powershell
python src/evaluate.py
```

### 7) Run CLI prediction

`predict.py`는 단일 센서 입력값을 인자로 받아 추론합니다.

```powershell
python src/predict.py --temperature 25 --vibration 0.3 --pressure 101 --humidity 45
```

이상치에 가까운 값으로도 테스트할 수 있습니다.

```powershell
python src/predict.py --temperature 80 --vibration 5.0 --pressure 180 --humidity 90
```

---

## 10. FastAPI Inference

### Run API server

```powershell
uvicorn src.app:app --reload
```

실행 후 아래 주소에서 Swagger UI를 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

### `/predict` Request Example

```json
{
  "temperature": 25,
  "vibration": 0.3,
  "pressure": 101,
  "humidity": 45
}
```

### `/predict` Response Example

응답에는 예측 결과와 reconstruction error, threshold가 포함됩니다.

```json
{
  "prediction": "normal",
  "reconstruction_error": 0.0,
  "threshold": 0.0
}
```

실제 값은 학습된 모델과 저장된 threshold에 따라 달라질 수 있습니다.

---

## 11. Verified Execution

로컬 Windows PowerShell 환경에서 다음 항목을 확인했습니다.

```text
1. GitHub 원격 저장소와 로컬 코드 동기화 확인
2. Python 가상환경 활성화
3. requirements.txt 패키지 설치 확인
4. src 파일 구조 확인
5. predict.py CLI 필수 인자 확인
6. FastAPI 서버 실행 확인
7. Swagger UI에서 /predict API 응답 확인
```

실행 확인 명령:

```powershell
git pull
git push
pip install -r requirements.txt
uvicorn src.app:app --reload
```

FastAPI 서버 실행 결과:

```text
Uvicorn running on http://127.0.0.1:8000
```

---

## 12. My Role

- 센서 데이터 feature 설계
- 샘플 데이터 생성 코드 구현
- 데이터 전처리 파이프라인 구성
- train / validation / test split 구성
- `StandardScaler` 기반 정규화 적용
- PyTorch AutoEncoder 모델 구현
- reconstruction error 기반 이상탐지 구현
- validation error 기반 threshold 설정
- precision, recall, F1-score 기반 성능 평가
- 단일 샘플 CLI 추론 구현
- FastAPI `/predict` endpoint 구현
- Swagger 기반 API 응답 확인
- README 실행 방법 및 결과 문서화

---

## 13. What I Learned

이 프로젝트를 통해 모델 개발은 학습 코드만으로 끝나지 않는다는 점을 배웠습니다.

모델이 실제 서비스 흐름에서 사용되기 위해서는 데이터 전처리, 모델 학습, 평가, 저장된 모델 로드, 단일 입력 추론, API 응답 구조까지 연결되어야 합니다.

또한 이상탐지에서는 accuracy만으로 성능을 판단하기 어렵고, 이상 데이터를 놓치지 않기 위한 recall 중심의 해석이 중요하다는 점을 확인했습니다.

---

## 14. Limitations & Improvements

현재 프로젝트는 샘플 센서 데이터를 기반으로 구성되어 있습니다.  
실제 서비스에 적용하기 위해서는 다음 보완이 필요합니다.

- 실제 센서 데이터셋 적용
- threshold 튜닝
- 다양한 모델 구조 비교
- feature engineering
- 모델 버전 관리
- API 요청 로그 저장
- Docker 기반 배포 환경 구성
- 모니터링 대시보드 구성
- 운영 환경에서의 성능 및 안정성 검증

---

## 15. Interview Summary

이 프로젝트는 센서 데이터를 기반으로 PyTorch AutoEncoder를 학습하고, reconstruction error와 threshold를 이용해 정상/이상 여부를 판단하는 AI 모델 개발 파이프라인입니다.

모델 학습과 평가에서 끝내지 않고, FastAPI `/predict` endpoint를 구성해 Swagger에서 단일 센서 입력값에 대한 `prediction`, `reconstruction_error`, `threshold` 응답까지 확인했습니다.

데이터 전처리, 모델 학습, 성능 평가, 추론 API, 실험 결과 문서화를 하나의 흐름으로 연결한 점이 핵심입니다.
