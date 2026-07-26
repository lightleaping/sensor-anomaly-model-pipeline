from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import DEFAULT_DATA_PATH, project_path


def _normal_sensor_frame(count: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create correlated, physically plausible row-level sensor observations."""
    load = rng.beta(2.2, 2.0, size=count)
    ambient = rng.normal(0.0, 1.0, size=count)

    temperature = 27.5 + 5.5 * load + 0.7 * ambient + rng.normal(0, 0.7, count)
    vibration = 0.18 + 0.28 * load + rng.normal(0, 0.035, count)
    pressure = (
        97.0
        + 7.0 * load
        + 0.25 * (temperature - 30.0)
        + rng.normal(0, 1.1, count)
    )
    humidity = 51.0 - 1.05 * (temperature - 30.0) + rng.normal(0, 2.8, count)

    return pd.DataFrame(
        {
            "temperature": np.clip(temperature, 20.0, 40.0),
            "vibration": np.clip(vibration, 0.05, 0.75),
            "pressure": np.clip(pressure, 88.0, 115.0),
            "humidity": np.clip(humidity, 20.0, 80.0),
        }
    )


def generate_sensor_data(
    output_path: str | Path = DEFAULT_DATA_PATH,
    normal_count: int = 1500,
    anomaly_count: int = 300,
    random_seed: int = 42,
) -> dict[str, Any]:
    if normal_count < 100:
        raise ValueError("normal_count must be at least 100")
    if anomaly_count < 20:
        raise ValueError("anomaly_count must be at least 20")

    rng = np.random.default_rng(random_seed)

    normal_data = _normal_sensor_frame(normal_count, rng)
    normal_data["label"] = 0
    normal_data["anomaly_type"] = "normal"

    anomaly_data = _normal_sensor_frame(anomaly_count, rng)
    anomaly_types = np.resize(
        np.array(
            [
                "thermal_overload",
                "mechanical_fault",
                "pressure_event",
                "leak",
                "combined_fault",
            ]
        ),
        anomaly_count,
    )
    rng.shuffle(anomaly_types)
    anomaly_data["anomaly_type"] = anomaly_types

    thermal = anomaly_types == "thermal_overload"
    anomaly_data.loc[thermal, "temperature"] += rng.uniform(9, 16, thermal.sum())

    mechanical = anomaly_types == "mechanical_fault"
    anomaly_data.loc[mechanical, "vibration"] += rng.uniform(
        0.45, 0.95, mechanical.sum()
    )

    pressure_event = anomaly_types == "pressure_event"
    pressure_direction = rng.choice([-1.0, 1.0], size=pressure_event.sum())
    anomaly_data.loc[pressure_event, "pressure"] += pressure_direction * rng.uniform(
        14, 24, pressure_event.sum()
    )

    leak = anomaly_types == "leak"
    anomaly_data.loc[leak, "pressure"] -= rng.uniform(10, 18, leak.sum())
    anomaly_data.loc[leak, "humidity"] += rng.uniform(14, 25, leak.sum())

    combined = anomaly_types == "combined_fault"
    anomaly_data.loc[combined, "temperature"] += rng.uniform(6, 11, combined.sum())
    anomaly_data.loc[combined, "vibration"] += rng.uniform(0.3, 0.65, combined.sum())
    anomaly_data.loc[combined, "pressure"] += rng.uniform(8, 16, combined.sum())

    anomaly_data["label"] = 1

    data = pd.concat([normal_data, anomaly_data], ignore_index=True)
    data.insert(0, "sample_id", [f"sensor_{index:06d}" for index in range(len(data))])
    data = data.sample(frac=1, random_state=random_seed).reset_index(drop=True)

    output_file = project_path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_file, index=False, encoding="utf-8-sig")

    summary = {
        "output_path": str(output_file),
        "total_rows": int(len(data)),
        "normal_count": int((data["label"] == 0).sum()),
        "anomaly_count": int((data["label"] == 1).sum()),
        "anomaly_types": {
            key: int(value)
            for key, value in data.loc[data["label"] == 1, "anomaly_type"]
            .value_counts()
            .sort_index()
            .items()
        },
        "random_seed": random_seed,
    }
    print(f"Generated sensor data: {output_file}")
    print(
        f"Rows: {summary['total_rows']} "
        f"(normal={summary['normal_count']}, anomaly={summary['anomaly_count']})"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate reproducible sensor data.")
    parser.add_argument("--output", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--normal-count", type=int, default=1500)
    parser.add_argument("--anomaly-count", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_sensor_data(
        output_path=args.output,
        normal_count=args.normal_count,
        anomaly_count=args.anomaly_count,
        random_seed=args.seed,
    )


if __name__ == "__main__":
    main()
