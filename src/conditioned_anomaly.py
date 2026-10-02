"""Score cycle sensor readings against healthy, flight-condition baselines."""

from __future__ import annotations

import pandas as pd


CONDITION_FEATURES = ["alt", "Mach", "TRA", "T2", "Fc"]


def score_sensor_rows(cycles: pd.DataFrame, detector: dict) -> pd.DataFrame:
    """Return predicted readings, signed residuals, and robust residual scores."""
    conditions = cycles[CONDITION_FEATURES].astype(float)
    scores = pd.DataFrame(index=cycles.index)
    for sensor, baseline in detector["sensors"].items():
        expected = baseline["model"].predict(conditions) + baseline["bias"]
        residual = cycles[sensor].astype(float).to_numpy() - expected
        scores[f"{sensor}_expected"] = expected
        scores[f"{sensor}_residual"] = residual
        scores[f"{sensor}_score"] = abs(residual) / baseline["scale"]
    return scores
