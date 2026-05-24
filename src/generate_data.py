import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_sensor_data(
    output_path: str = "data/sensor_data.csv",
    normal_count: int = 1000,
    anomaly_count: int = 120,
    random_seed: int = 42,
) -> None:
    np.random.seed(random_seed)

    # 정상 센서 데이터
    normal_data = pd.DataFrame({
        "temperature": np.random.normal(loc=30.0, scale=2.0, size=normal_count),
        "vibration": np.random.normal(loc=0.35, scale=0.08, size=normal_count),
        "pressure": np.random.normal(loc=100.0, scale=3.0, size=normal_count),
        "humidity": np.random.normal(loc=45.0, scale=5.0, size=normal_count),
        "label": 0,
    })

    # 이상 센서 데이터
    anomaly_data = pd.DataFrame({
        "temperature": np.random.normal(loc=42.0, scale=4.0, size=anomaly_count),
        "vibration": np.random.normal(loc=0.85, scale=0.15, size=anomaly_count),
        "pressure": np.random.normal(loc=116.0, scale=6.0, size=anomaly_count),
        "humidity": np.random.normal(loc=65.0, scale=8.0, size=anomaly_count),
        "label": 1,
    })

    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"Saved: {output_file}")
    print(f"Total rows: {len(data)}")
    print(data["label"].value_counts().rename(index={0: "normal", 1: "anomaly"}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/sensor_data.csv")
    parser.add_argument("--normal-count", type=int, default=1000)
    parser.add_argument("--anomaly-count", type=int, default=120)
    args = parser.parse_args()

    generate_sensor_data(
        output_path=args.output,
        normal_count=args.normal_count,
        anomaly_count=args.anomaly_count,
    )


if __name__ == "__main__":
    main()
