from __future__ import annotations

import argparse
import hashlib
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.artifacts import write_json
from src.config import (
    DEFAULT_DATA_PATH,
    DEFAULT_PREPROCESS_METADATA_PATH,
    DEFAULT_PREPROCESSED_PATH,
    DEFAULT_SCALER_PATH,
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    project_path,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess_data(
    input_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path = DEFAULT_PREPROCESSED_PATH,
    scaler_path: str | Path = DEFAULT_SCALER_PATH,
    metadata_path: str | Path = DEFAULT_PREPROCESS_METADATA_PATH,
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_seed: int = 42,
) -> dict[str, Any]:
    if not 0 < test_size < 0.5:
        raise ValueError("test_size must be between 0 and 0.5")
    if not 0 < val_size < 0.5:
        raise ValueError("val_size must be between 0 and 0.5")

    input_file = project_path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(
            f"Input data not found: {input_file}. "
            "Run `python -m src.generate_data` first."
        )

    df = pd.read_csv(input_file, encoding="utf-8-sig")
    required_columns = [*FEATURE_COLUMNS, LABEL_COLUMN]
    missing_columns = [column for column in required_columns if column not in df]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    original_rows = len(df)
    feature_list = list(FEATURE_COLUMNS)
    df.loc[:, feature_list] = df.loc[:, feature_list].apply(
        pd.to_numeric, errors="coerce"
    )
    df.loc[:, LABEL_COLUMN] = pd.to_numeric(df[LABEL_COLUMN], errors="coerce")
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=required_columns).copy()
    df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

    labels = set(df[LABEL_COLUMN].unique())
    if not labels.issubset({0, 1}) or labels != {0, 1}:
        raise ValueError(f"`label` must contain both binary values 0 and 1; got {labels}")
    if int((df[LABEL_COLUMN] == 0).sum()) < 20:
        raise ValueError("At least 20 normal rows are required")
    if int((df[LABEL_COLUMN] == 1).sum()) < 2:
        raise ValueError("At least 2 anomaly rows are required")

    X = df.loc[:, feature_list].to_numpy(dtype=np.float32)
    y = df[LABEL_COLUMN].to_numpy(dtype=np.int64)
    source_indices = df.index.to_numpy(dtype=np.int64)
    split_strata = y
    if "anomaly_type" in df:
        anomaly_type_counts = df["anomaly_type"].astype(str).value_counts()
        if int(anomaly_type_counts.min()) >= 2:
            split_strata = df["anomaly_type"].astype(str).to_numpy()

    (
        X_train_full,
        X_test_raw,
        y_train_full,
        y_test,
        train_indices_full,
        test_indices,
    ) = train_test_split(
        X,
        y,
        source_indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=split_strata,
    )

    test_sample_ids = (
        np.asarray(
            df.loc[test_indices, "sample_id"].astype(str).tolist(),
            dtype=np.str_,
        )
        if "sample_id" in df
        else np.asarray([str(index) for index in test_indices], dtype=np.str_)
    )
    test_anomaly_types = (
        np.asarray(
            df.loc[test_indices, "anomaly_type"].astype(str).tolist(),
            dtype=np.str_,
        )
        if "anomaly_type" in df
        else np.asarray(
            np.where(y_test == 1, "anomaly", "normal"),
            dtype=np.str_,
        )
    )

    normal_train_mask = y_train_full == 0
    X_train_normal = X_train_full[normal_train_mask]
    normal_indices = train_indices_full[normal_train_mask]

    X_train_raw, X_val_raw, train_indices, val_indices = train_test_split(
        X_train_normal,
        normal_indices,
        test_size=val_size,
        random_state=random_seed,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_val = scaler.transform(X_val_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    output_file = project_path(output_path)
    scaler_file = project_path(scaler_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    scaler_file.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_file,
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        X_test_raw=X_test_raw,
        y_test=y_test,
        train_indices=train_indices,
        val_indices=val_indices,
        test_indices=test_indices,
        test_sample_ids=test_sample_ids,
        test_anomaly_types=test_anomaly_types,
        feature_columns=np.asarray(FEATURE_COLUMNS),
    )
    with scaler_file.open("wb") as file:
        pickle.dump(scaler, file, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "input_path": str(input_file),
        "input_sha256": _sha256(input_file),
        "rows_read": int(original_rows),
        "rows_used": int(len(df)),
        "rows_dropped": int(original_rows - len(df)),
        "feature_columns": list(FEATURE_COLUMNS),
        "label_column": LABEL_COLUMN,
        "train_normal_count": int(len(X_train)),
        "validation_normal_count": int(len(X_val)),
        "test_count": int(len(X_test)),
        "test_normal_count": int((y_test == 0).sum()),
        "test_anomaly_count": int((y_test == 1).sum()),
        "test_anomaly_type_counts": {
            key: int(value)
            for key, value in pd.Series(test_anomaly_types)
            .value_counts()
            .sort_index()
            .items()
        },
        "test_size": test_size,
        "validation_size_within_normal_train": val_size,
        "random_seed": random_seed,
        "output_path": str(output_file),
        "scaler_path": str(scaler_file),
    }
    metadata_file = write_json(metadata_path, metadata)

    print(f"Preprocessed data: {output_file}")
    print(f"Scaler: {scaler_file}")
    print(f"Metadata: {metadata_file}")
    print(
        f"Shapes: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate, split, and scale sensor data.")
    parser.add_argument("--input", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--output", default=str(DEFAULT_PREPROCESSED_PATH))
    parser.add_argument("--scaler", default=str(DEFAULT_SCALER_PATH))
    parser.add_argument("--metadata", default=str(DEFAULT_PREPROCESS_METADATA_PATH))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    preprocess_data(
        input_path=args.input,
        output_path=args.output,
        scaler_path=args.scaler,
        metadata_path=args.metadata,
        test_size=args.test_size,
        val_size=args.val_size,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
