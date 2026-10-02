"""Per-sensor anomaly features for the synthetic-fault demonstration model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.conditioned_anomaly import score_sensor_rows


def sensor_feature_rows(
    cycles: pd.DataFrame,
    detector: dict,
    sensors: list[str],
    events: pd.DataFrame | None = None,
    *,
    scores: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Expand cycle residuals to labeled sensor-cycle examples."""
    if scores is None:
        scores = score_sensor_rows(cycles, detector)
    max_columns = [f"{sensor}_score" for sensor in sensors]
    frames = []
    event_map = {}
    if events is not None and not events.empty:
        for event in events.to_dict("records"):
            event_map.setdefault((event["engine_id"], event["sensor"]), []).append(
                (int(event["start_cycle"]), int(event["end_cycle"]))
            )

    for sensor_index, sensor in enumerate(sensors):
        residual = scores[f"{sensor}_residual"]
        abs_score = scores[f"{sensor}_score"]
        frame = cycles[["unit", "cycle"]].copy()
        frame["sensor"] = sensor
        frame["sensor_index"] = sensor_index
        frame["signed_score"] = np.sign(residual) * abs_score
        frame["abs_score"] = abs_score
        frame["delta_score"] = frame.groupby("unit", sort=False)["signed_score"].diff().fillna(0.0)
        frame["rolling_max_3"] = frame.groupby("unit", sort=False)["abs_score"].transform(
            lambda values: values.rolling(3, min_periods=1).max()
        )
        frame["rolling_mean_3"] = frame.groupby("unit", sort=False)["abs_score"].transform(
            lambda values: values.rolling(3, min_periods=1).mean()
        )
        other_columns = [name for name in max_columns if name != f"{sensor}_score"]
        frame["max_peer_score"] = scores[other_columns].max(axis=1).to_numpy()
        label = np.zeros(len(frame), dtype=np.int8)
        if events is not None:
            for row_index, (engine_id, cycle) in enumerate(
                zip(frame["unit"].to_numpy(), frame["cycle"].to_numpy())
            ):
                label[row_index] = any(
                    start <= int(cycle) <= end
                    for start, end in event_map.get((engine_id, sensor), ())
                )
        frame["label"] = label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


FEATURE_COLUMNS = [
    "sensor_index",
    "signed_score",
    "abs_score",
    "delta_score",
    "rolling_max_3",
    "rolling_mean_3",
    "max_peer_score",
]

