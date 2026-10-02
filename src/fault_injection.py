"""Create reproducible synthetic sensor-fault examples for offline evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def inject_labeled_faults(
    cycles: pd.DataFrame,
    detector: dict,
    sensors: list[str],
    *,
    seed: int = 41,
    events_per_engine: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inject step, ramp, and spike faults into a copy of healthy cycle rows."""
    rng = np.random.default_rng(seed)
    injected = cycles.sort_values(["data_set", "unit_number", "cycle"]).copy()
    for sensor in sensors:
        injected[sensor] = injected[sensor].astype(np.float64)
    events: list[dict] = []
    event_number = 0
    kinds = ("bias_step", "gradual_drift", "single_cycle_spike")

    for engine, indexes in injected.groupby("unit", sort=True).groups.items():
        engine_rows = injected.loc[indexes]
        healthy_indexes = engine_rows.index[engine_rows["hs"] == 1].to_numpy()
        if len(healthy_indexes) < 8:
            continue
        for event_index in range(events_per_engine):
            kind = kinds[event_index % len(kinds)]
            length = {"bias_step": 4, "gradual_drift": 5, "single_cycle_spike": 1}[kind]
            if len(healthy_indexes) <= length + 2:
                continue
            start_position = int(rng.integers(1, len(healthy_indexes) - length))
            event_indexes = healthy_indexes[start_position : start_position + length]
            # Keep an event within one continuous sequence of healthy cycles.
            event_cycles = injected.loc[event_indexes, "cycle"].to_numpy(dtype=int)
            if not np.all(np.diff(event_cycles) == 1):
                continue
            sensor = str(rng.choice(sensors))
            baseline = detector["sensors"][sensor]
            direction = float(rng.choice([-1.0, 1.0]))
            magnitude = float(rng.uniform(4.0, 7.0) * baseline["scale"])
            original = injected.loc[event_indexes, sensor].to_numpy(dtype=float).copy()
            if kind == "gradual_drift":
                offsets = direction * magnitude * np.linspace(0.35, 1.0, length)
            else:
                offsets = np.full(length, direction * magnitude)
            injected.loc[event_indexes, sensor] = original + offsets
            event_number += 1
            events.append({
                "event_id": f"SYN-{event_number:05d}",
                "source_type": "synthetic_injection",
                "data_set": str(engine_rows["data_set"].iloc[0]),
                "engine_id": str(engine),
                "start_cycle": int(event_cycles[0]),
                "end_cycle": int(event_cycles[-1]),
                "sensor": sensor,
                "label": "injected_sensor_anomaly",
                "injection_type": kind,
                "severity_robust_scale": round(magnitude / baseline["scale"], 3),
                "finding": "Synthetic signal alteration; not an observed fault.",
                "action": "Not applicable; synthetic benchmark only.",
                "post_action_result": "Not applicable.",
                "evidence_ref": "scripts/evaluate_fault_injection_detector.py",
                "label_confidence": "known_by_construction",
                "annotator": "FleetPulse synthetic generator",
                "notes": "Does not establish real aircraft fault-detection performance.",
            })

    return injected, pd.DataFrame(events)

