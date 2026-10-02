"""Train FleetPulse's experimental RUL model on available N-CMAPSS files.

The NASA development arrays are reduced to one row per engine cycle. The
official test arrays are held out for evaluation and are never used to fit the
model. Inputs intentionally match dashboard/preflight_monitor.py's 46 features.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import h5py
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
MODEL_PATH = ROOT / "models" / "rul_random_forest_expanded.pkl"
BASELINE_MODEL_PATH = ROOT / "models" / "rul_random_forest.pkl"
REPLAY_DATA_PATH = ROOT / "data" / "processed" / "cycle_data_expanded.csv"
REPLAY_ENGINE_COUNT = 20
REPORT_PATH = ROOT / "reports" / "rul_model_expansion.md"
ANOMALY_MODEL_PATH = ROOT / "models" / "conditioned_sensor_baselines.pkl"
ANOMALY_REPORT_PATH = ROOT / "reports" / "conditioned_anomaly_evaluation.md"
ANOMALY_THRESHOLD_QUANTILE = 0.99
SENSORS = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]
CONDITIONS = ["alt", "Mach", "TRA", "T2"]
BASE_FEATURES = SENSORS + CONDITIONS
MODEL_FEATURES = (
    BASE_FEATURES
    + [f"{sensor}_rate" for sensor in SENSORS]
    + [f"{sensor}_rolling_mean_10" for sensor in SENSORS]
)
CONDITION_FEATURES = CONDITIONS + ["Fc"]


def _names(dataset: h5py.Dataset) -> list[str]:
    return [value.decode("utf-8") for value in dataset[:]]


def aggregate_cycles(path: str, split: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Aggregate time-step arrays into cycle-level features without pooling engines."""
    with h5py.File(path, "r") as handle:
        a = handle[f"A_{split}"]
        x_sensor = handle[f"X_s_{split}"]
        w = handle[f"W_{split}"]
        y = handle[f"Y_{split}"]
        n_rows = a.shape[0]

        sensor_columns = _names(handle["X_s_var"])
        condition_columns = _names(handle["W_var"])
        auxiliary_columns = _names(handle["A_var"])
        if sensor_columns != SENSORS or condition_columns != CONDITIONS:
            raise ValueError(f"Unexpected N-CMAPSS columns in {os.path.basename(path)}")
        if auxiliary_columns != ["unit", "cycle", "Fc", "hs"]:
            raise ValueError(f"Unexpected N-CMAPSS auxiliary columns in {os.path.basename(path)}")

        # The arrays are ordered by engine and cycle. Reading only the first
        # two A columns keeps the grouping index small for these large files.
        keys = np.asarray(a[:, :2], dtype=np.int32)
        if not np.all(
            (np.diff(keys[:, 0]) > 0)
            | ((np.diff(keys[:, 0]) == 0) & (np.diff(keys[:, 1]) >= 0))
        ):
            raise ValueError(f"Engine/cycle rows are not sorted in {os.path.basename(path)}")

        cycle_starts = np.r_[
            0,
            np.flatnonzero(np.any(keys[1:] != keys[:-1], axis=1)) + 1,
        ]
        cycle_metadata = np.asarray(a[cycle_starts, 2:4], dtype=np.int8)
        counts = np.diff(np.r_[cycle_starts, n_rows]).astype(np.float32)

        sensor_readings = np.empty((n_rows, len(SENSORS)), dtype=np.float32)
        condition_readings = np.empty((n_rows, len(CONDITIONS)), dtype=np.float32)
        x_sensor.read_direct(sensor_readings)
        w.read_direct(condition_readings)
        sensor_means = np.add.reduceat(sensor_readings, cycle_starts, axis=0)
        sensor_means /= counts[:, None]
        condition_means = np.add.reduceat(condition_readings, cycle_starts, axis=0)
        condition_means /= counts[:, None]
        cycle_means = np.column_stack([sensor_means, condition_means])

        rul_samples = np.asarray(y[:, 0], dtype=np.float32)
        cycle_rul = np.add.reduceat(rul_samples, cycle_starts) / counts
        cycle_keys = keys[cycle_starts]

    frame = pd.DataFrame(cycle_means, columns=BASE_FEATURES)
    frame.insert(0, "cycle", cycle_keys[:, 1])
    frame.insert(0, "unit", cycle_keys[:, 0])
    frame["rul"] = cycle_rul
    frame["Fc"] = cycle_metadata[:, 0]
    frame["hs"] = cycle_metadata[:, 1]
    frame = frame.sort_values(["unit", "cycle"]).reset_index(drop=True)

    grouped = frame.groupby("unit", sort=False)
    for sensor in SENSORS:
        frame[f"{sensor}_rate"] = grouped[sensor].diff().fillna(0.0).astype(np.float32)
        frame[f"{sensor}_rolling_mean_10"] = grouped[sensor].transform(
            lambda values: values.rolling(window=10, min_periods=1).mean()
        ).astype(np.float32)

    frame["data_set"] = Path(path).stem
    frame["unit_number"] = frame["unit"].astype(int)
    frame["unit"] = (
        frame["data_set"] + " · Engine " + frame["unit_number"].astype(str)
    )
    summary = {
        "rows": n_rows,
        "engines": int(frame["unit"].nunique()),
        "cycles": int(len(frame)),
    }
    return frame, summary


def train_conditioned_anomaly_model(
    development: pd.DataFrame, evaluation: pd.DataFrame
) -> tuple[dict, list[str]]:
    """Fit healthy sensor response models and calibrate a fleet-wide threshold."""
    calibration_engines = set()
    for data_set, group in development.groupby("data_set", sort=True):
        units = np.sort(group["unit_number"].unique())
        calibration_engines.add((data_set, int(units[-1])))

    is_calibration = development.apply(
        lambda row: (row["data_set"], int(row["unit_number"])) in calibration_engines,
        axis=1,
    )
    fit_rows = development.loc[~is_calibration]
    calibration_rows = development.loc[is_calibration]
    # N-CMAPSS encodes healthy operation as hs=1 and unhealthy operation as hs=0.
    healthy_fit = fit_rows.loc[fit_rows["hs"] == 1]
    healthy_calibration = calibration_rows.loc[calibration_rows["hs"] == 1]
    if healthy_fit.empty or healthy_calibration.empty:
        raise ValueError("Healthy-state cycles are required for fit and calibration")

    detector = {
        "condition_features": CONDITION_FEATURES,
        "sensors": {},
        "calibration_quantile": ANOMALY_THRESHOLD_QUANTILE,
        "calibration_engines": len(calibration_engines),
    }
    calibration_scores = []
    for index, sensor in enumerate(SENSORS):
        model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=3,
            max_features=1.0,
            n_jobs=-1,
            random_state=42 + index,
        )
        model.fit(healthy_fit[CONDITION_FEATURES], healthy_fit[sensor])
        predicted = model.predict(healthy_calibration[CONDITION_FEATURES])
        residual = healthy_calibration[sensor].to_numpy(dtype=float) - predicted
        bias = float(np.median(residual))
        mad = float(np.median(np.abs(residual - bias)))
        scale = 1.4826 * mad
        if not np.isfinite(scale) or scale < 1e-8:
            scale = float(np.std(residual))
        scale = max(scale, 1e-8)
        sensor_scores = np.abs(residual - bias) / scale
        calibration_scores.append(sensor_scores)
        detector["sensors"][sensor] = {
            "model": model,
            "bias": bias,
            "scale": scale,
        }

    # Calibrate on the largest standardized channel residual per healthy cycle,
    # so the threshold covers all 14 channels as one screening family.
    calibration_max = np.max(np.column_stack(calibration_scores), axis=1)
    detector["threshold"] = float(
        np.quantile(calibration_max, ANOMALY_THRESHOLD_QUANTILE)
    )

    test_scores = []
    for sensor in SENSORS:
        baseline = detector["sensors"][sensor]
        predicted = baseline["model"].predict(evaluation[CONDITION_FEATURES])
        residual = evaluation[sensor].to_numpy(dtype=float) - predicted - baseline["bias"]
        test_scores.append(np.abs(residual) / baseline["scale"])
    max_scores = np.max(np.column_stack(test_scores), axis=1)
    detected = max_scores >= detector["threshold"]
    health_state = evaluation["hs"].to_numpy(dtype=int)
    healthy_mask = health_state == 1
    degraded_mask = health_state == 0
    detected_alerts = detected.astype(int)
    true_degraded = degraded_mask.astype(int)

    metrics = {
        "threshold": detector["threshold"],
        "healthy_cycles_fit": len(healthy_fit),
        "healthy_cycles_calibration": len(healthy_calibration),
        "calibration_engines": len(calibration_engines),
        "test_cycles": len(evaluation),
        "healthy_test_cycles": int(healthy_mask.sum()),
        "degraded_test_cycles": int(degraded_mask.sum()),
        "healthy_false_alert_rate": float(detected[healthy_mask].mean()) if healthy_mask.any() else float("nan"),
        "degraded_detection_rate": float(detected[degraded_mask].mean()) if degraded_mask.any() else float("nan"),
        "precision": float(precision_score(true_degraded, detected_alerts, zero_division=0)),
        "recall": float(recall_score(true_degraded, detected_alerts, zero_division=0)),
        "f1": float(f1_score(true_degraded, detected_alerts, zero_division=0)),
    }

    ANOMALY_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(detector, ANOMALY_MODEL_PATH, compress=3)

    report = [
        "# Condition-aware anomaly screening review",
        "",
        "## Method",
        "",
        "For each of the 14 measured sensors, a Random Forest baseline predicts the healthy-state sensor-cycle mean from altitude, Mach, TRA, inlet temperature, and flight class. N-CMAPSS encodes healthy operation as `hs=1` and unhealthy operation as `hs=0`; models fit only `hs=1` cycles from development engines. One engine from each dataset is held out from fitting to estimate residual bias and robust scale. The family-wide threshold is the 99th percentile of the largest standardized sensor residual on those held-out healthy cycles. A replay flags a sensor when its residual score crosses this calibrated threshold; the UI retains the existing current-crossing / 2-of-3 persistence rule.",
        "",
        f"- Fit healthy cycles: {metrics['healthy_cycles_fit']:,}.",
        f"- Healthy calibration cycles: {metrics['healthy_cycles_calibration']:,} across {metrics['calibration_engines']} held-out development engines.",
        f"- Calibrated family-wide threshold: {metrics['threshold']:.2f} robust residual scales.",
        f"- Held-out NASA test cycles: {metrics['test_cycles']:,} ({metrics['healthy_test_cycles']:,} healthy, {metrics['degraded_test_cycles']:,} degraded).",
        "",
        "## Held-out test results",
        "",
        f"- False-alert rate on healthy (`hs=1`) cycles: {metrics['healthy_false_alert_rate']:.1%}.",
        f"- Detection rate on unhealthy (`hs=0`) cycles: {metrics['degraded_detection_rate']:.1%}.",
        f"- Precision: {metrics['precision']:.1%}; recall: {metrics['recall']:.1%}; F1: {metrics['f1']:.3f}.",
        "",
        "## Limits",
        "",
        "`hs` is a simulated healthy/degraded state label, not a confirmed sensor fault, component diagnosis, maintenance outcome, or aircraft dispatch label. The false-alert and detection rates therefore describe only this held-out N-CMAPSS test split. Cycle means discard within-cycle transients, and the model is experimental. This screening score cannot identify a failure mechanism or validate a repair.",
        "",
        "The dashboard replay remains a balanced 20-engine development sample. Its RUL model and its independent RUL test evaluation are documented in `reports/rul_model_expansion.md`.",
    ]
    ANOMALY_REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(
        "Condition-aware anomaly holdout: "
        f"healthy false-alert rate={metrics['healthy_false_alert_rate']:.1%}, "
        f"degraded detection rate={metrics['degraded_detection_rate']:.1%}, "
        f"threshold={metrics['threshold']:.2f} robust scales"
    )
    return detector, report


def main() -> None:
    development = []
    evaluation = []
    included = []
    skipped = []

    for path in sorted(glob.glob(str(RAW_DIR / "*.h5"))):
        try:
            dev, dev_summary = aggregate_cycles(path, "dev")
            test, test_summary = aggregate_cycles(path, "test")
            if set(dev["unit_number"].unique()) & set(test["unit_number"].unique()):
                raise ValueError("development and test engine IDs overlap")
            development.append(dev)
            evaluation.append(test)
            included.append((Path(path).name, dev_summary, test_summary))
            print(
                f"Loaded {Path(path).name}: "
                f"{dev_summary['engines']} dev engines / {dev_summary['cycles']:,} cycles; "
                f"{test_summary['engines']} test engines / {test_summary['cycles']:,} cycles"
            )
        except Exception as exc:
            skipped.append((Path(path).name, str(exc).splitlines()[0]))
            print(f"Skipped {Path(path).name}: {str(exc).splitlines()[0]}")

    if not development or not evaluation:
        raise RuntimeError("No complete N-CMAPSS development/test file pairs were readable")

    train = pd.concat(development, ignore_index=True)
    test = pd.concat(evaluation, ignore_index=True)
    dataset_engines = {
        data_set: np.sort(group["unit_number"].unique())
        for data_set, group in train.groupby("data_set", sort=True)
    }
    replay_allocation = {data_set: min(2, len(units)) for data_set, units in dataset_engines.items()}
    remaining_slots = REPLAY_ENGINE_COUNT - sum(replay_allocation.values())
    for data_set, units in sorted(
        dataset_engines.items(), key=lambda item: (-len(item[1]), item[0])
    ):
        if remaining_slots <= 0:
            break
        if replay_allocation[data_set] < len(units):
            replay_allocation[data_set] += 1
            remaining_slots -= 1
    if remaining_slots:
        raise ValueError(f"Only {REPLAY_ENGINE_COUNT - remaining_slots} replay engines are available")

    replay_parts = []
    for data_set, units in dataset_engines.items():
        selected_positions = np.linspace(
            0, len(units) - 1, replay_allocation[data_set], dtype=int
        )
        selected_units = units[selected_positions]
        replay_parts.append(
            train.loc[
                (train["data_set"] == data_set)
                & train["unit_number"].isin(selected_units)
            ]
        )
    replay = pd.concat(replay_parts, ignore_index=True)
    REPLAY_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    replay.to_csv(REPLAY_DATA_PATH, index=False)
    x_train = train[MODEL_FEATURES].astype(np.float32)
    y_train = train["rul"].astype(np.float32)
    x_test = test[MODEL_FEATURES].astype(np.float32)
    y_test = test["rul"].astype(np.float32)

    model = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        max_features=0.8,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    r2 = r2_score(y_test, predictions)
    baseline_metrics = None
    if BASELINE_MODEL_PATH.exists():
        try:
            baseline_model = joblib.load(BASELINE_MODEL_PATH)
            baseline_predictions = baseline_model.predict(x_test)
            baseline_metrics = (
                mean_absolute_error(y_test, baseline_predictions),
                float(np.sqrt(mean_squared_error(y_test, baseline_predictions))),
                r2_score(y_test, baseline_predictions),
            )
        except Exception as exc:
            print(f"Could not score the previous model on this holdout: {exc}")
    anomaly_detector, _ = train_conditioned_anomaly_model(train, test)
    test["prediction"] = predictions
    per_dataset = test.groupby("data_set").apply(
        lambda group: mean_absolute_error(group["rul"], group["prediction"]),
        include_groups=False,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)

    report = [
        "# Expanded N-CMAPSS RUL model review",
        "",
        "## Data used",
        "",
        "The model is fitted on all readable `*_dev` arrays and evaluated only on their paired `*_test` arrays. Each time-step file is reduced to one row per engine-cycle by averaging the 14 measured sensor channels and four flight-condition channels. The feature set also uses per-cycle sensor rates and 10-cycle rolling means. Development and test engine IDs are checked for overlap within each file.",
        "",
        f"- Training: {len(train):,} engine-cycles from {train['data_set'].nunique()} dataset files and {train.groupby('data_set')['unit'].nunique().sum()} dataset-specific engines.",
        f"- Held-out evaluation: {len(test):,} engine-cycles from {test['data_set'].nunique()} dataset files and {test.groupby('data_set')['unit'].nunique().sum()} dataset-specific engines.",
        f"- Input features: {len(MODEL_FEATURES)} (14 measured sensors, 4 flight conditions, 14 sensor rates, and 14 rolling means).",
        "- The internal virtual-sensor arrays and latent degradation-parameter arrays are not model inputs because their availability as operational telemetry is not established.",
        f"- Dashboard replay: {replay['unit'].nunique()} dataset-qualified engines and {len(replay):,} cycles, balanced across the readable datasets. The RUL model still trains on all development engines.",
        f"- Replay data is saved to `{REPLAY_DATA_PATH.relative_to(ROOT).as_posix()}`.",
        "",
        "## Held-out test results",
        "",
        "Both models below are scored against the same 2,938 cycle-level rows from the paired NASA test arrays.",
        "",
        "| Model | MAE (cycles) | RMSE (cycles) | R² |",
        "|---|---:|---:|---:|",
    ]
    if baseline_metrics is not None:
        report.append(
            f"| Previous small-subset model | {baseline_metrics[0]:.2f} | "
            f"{baseline_metrics[1]:.2f} | {baseline_metrics[2]:.3f} |"
        )
    report.append(f"| Expanded model | {mae:.2f} | {rmse:.2f} | {r2:.3f} |")
    report += [
        "",
        f"- MAE: {mae:.2f} cycles",
        f"- RMSE: {rmse:.2f} cycles",
        f"- R²: {r2:.3f}",
        "",
        "| Test dataset | Engine-cycles | MAE (cycles) |",
        "|---|---:|---:|",
    ]
    for data_set, group in test.groupby("data_set"):
        report.append(
            f"| {data_set} | {len(group):,} | {per_dataset.loc[data_set]:.2f} |"
        )
    report += [
        "",
        "## Limitations",
        "",
        "These are simulated-engine, held-out-file test metrics, not evidence of performance on operational aircraft. The cycle-level means discard within-cycle transients. Test results must be interpreted alongside per-dataset scores, and the RUL output remains experimental. This model does not validate the dashboard's anomaly flags or its maintenance guidance.",
        "",
        "Files that could not be read:",
    ]
    if skipped:
        report.extend(f"- `{name}`: {reason}" for name, reason in skipped)
    else:
        report.append("- None.")

    REPORT_PATH.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"\nTrain cycles: {len(train):,}; test cycles: {len(test):,}")
    print(f"Held-out test MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.3f}")
    if baseline_metrics is not None:
        print(
            "Previous model on same holdout: "
            f"MAE={baseline_metrics[0]:.2f}, RMSE={baseline_metrics[1]:.2f}, "
            f"R2={baseline_metrics[2]:.3f}"
        )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved conditioned anomaly model: {ANOMALY_MODEL_PATH}")
    print(f"Saved expanded replay data: {REPLAY_DATA_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
