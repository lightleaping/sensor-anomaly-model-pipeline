import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = ["temperature", "vibration", "pressure", "humidity"]
LABEL_COLUMN = "label"


def preprocess_data(
    input_path: str = "data/sensor_data.csv",
    output_path: str = "outputs/preprocessed_data.npz",
    scaler_path: str = "models/scaler.pkl",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
) -> None:
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    df = pd.read_csv(input_file)

    missing_columns = [col for col in FEATURE_COLUMNS + [LABEL_COLUMN] if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns: {missing_columns}")

    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    df[LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")

    df = df.dropna(subset=FEATURE_COLUMNS + [LABEL_COLUMN])
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    X = df[FEATURE_COLUMNS].values.astype(np.float32)
    y = df[LABEL_COLUMN].values.astype(np.int64)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_seed,
        stratify=y,
    )

    # AutoEncoder는 정상 패턴을 학습하는 모델이므로 학습 데이터는 정상 데이터만 사용한다.
    normal_train_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_train_mask]

    X_train, X_val = train_test_split(
        X_train_normal,
        test_size=val_size,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    scaler_file = Path(scaler_path)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_file,
        X_train=X_train_scaled,
        X_val=X_val_scaled,
        X_test=X_test_scaled,
        y_test=y_test,
        feature_columns=np.array(FEATURE_COLUMNS),
    )

    with open(scaler_file, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Saved preprocessed data: {output_file}")
    print(f"Saved scaler: {scaler_file}")
    print(f"Train shape: {X_train_scaled.shape}")
    print(f"Validation shape: {X_val_scaled.shape}")
    print(f"Test shape: {X_test_scaled.shape}")
    print("Test label counts:")
    print(pd.Series(y_test).value_counts().rename(index={0: "normal", 1: "anomaly"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/sensor_data.csv")
    parser.add_argument("--output", default="outputs/preprocessed_data.npz")
    parser.add_argument("--scaler", default="models/scaler.pkl")
    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_path=args.output,
        scaler_path=args.scaler,
    )


if __name__ == "__main__":
    main()
