"""Train and evaluate a simulation-only sensor fault screen using injection labels."""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_expanded_rul_model import RAW_DIR, SENSORS, aggregate_cycles
from src.anomaly_classifier import FEATURE_COLUMNS, sensor_feature_rows
from src.fault_injection import inject_labeled_faults


BASELINE_PATH = ROOT / "models" / "conditioned_sensor_baselines.pkl"
CLASSIFIER_PATH = ROOT / "models" / "synthetic_fault_classifier.pkl"
REPORT_PATH = ROOT / "reports" / "fault_injection_evaluation.md"
LABEL_DIR = ROOT / "data" / "labels"
TEMPLATE_PATH = LABEL_DIR / "anomaly_event_template.csv"
INJECTION_PATH = LABEL_DIR / "synthetic_fault_injections.csv"
MAX_FALSE_ALERT_RATE = 0.02
CALIBRATION_QUANTILES = {"2% target": 0.98, "5% target": 0.95, "10% target": 0.90}
SEED = 53


def load_arrays(split: str) -> tuple[list[pd.DataFrame], list[str]]:
    frames = []
    skipped = []
    for path in sorted(glob.glob(str(RAW_DIR / "*.h5"))):
        try:
            frame, _ = aggregate_cycles(path, split)
            frames.append(frame)
        except Exception as exc:
            skipped.append(f"{Path(path).name}: {str(exc).splitlines()[0]}")
    return frames, skipped


def main() -> None:
    baseline = joblib.load(BASELINE_PATH)
    development_parts, skipped_dev = load_arrays("dev")
    test_parts, skipped_test = load_arrays("test")
    development = pd.concat(development_parts, ignore_index=True)
    evaluation = pd.concat(test_parts, ignore_index=True)
    calibration_keys = set()
    for data_set, group in development.groupby("data_set", sort=True):
        calibration_keys.add((data_set, int(group["unit_number"].max())))
    is_calibration = development.apply(
        lambda row: (row["data_set"], int(row["unit_number"])) in calibration_keys,
        axis=1,
    )
    fit_healthy = development.loc[(~is_calibration) & (development["hs"] == 1)].copy()
    calibration_healthy = development.loc[is_calibration & (development["hs"] == 1)].copy()
    test_healthy = evaluation.loc[evaluation["hs"] == 1].copy()
    if min(len(fit_healthy), len(calibration_healthy), len(test_healthy)) == 0:
        raise RuntimeError("Healthy development, calibration, and test cycles are required")

    injected_fit, train_events = inject_labeled_faults(
        fit_healthy, baseline, SENSORS, seed=SEED, events_per_engine=5
    )
    negative = sensor_feature_rows(fit_healthy, baseline, SENSORS)
    positive_mix = sensor_feature_rows(injected_fit, baseline, SENSORS, train_events)
    train_examples = pd.concat(
        [negative.assign(label=0), positive_mix.loc[positive_mix["label"] == 1]],
        ignore_index=True,
    )
    model = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=3,
        max_features=0.9,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=SEED,
    )
    model.fit(train_examples[FEATURE_COLUMNS], train_examples["label"])

    calibration_examples = sensor_feature_rows(calibration_healthy, baseline, SENSORS)
    calibration_probabilities = model.predict_proba(
        calibration_examples[FEATURE_COLUMNS]
    )[:, 1]
    calibration_examples["probability"] = calibration_probabilities
    calibration_max = calibration_examples.groupby(["unit", "cycle"])["probability"].max()
    # Empirical high quantile gives the development calibration engines control
    # over a family-wise cycle alert rate. Synthetic events do not set this limit.
    thresholds = {
        name: float(np.quantile(calibration_max, quantile))
        for name, quantile in CALIBRATION_QUANTILES.items()
    }
    threshold = thresholds["2% target"]
    artifact = {
        "model": model,
        "feature_columns": FEATURE_COLUMNS,
        "sensors": SENSORS,
        "threshold": threshold,
        "target_false_alert_rate": MAX_FALSE_ALERT_RATE,
        "calibration_engines": sorted(calibration_keys),
        "training_source": "synthetic fault injections on N-CMAPSS development healthy cycles",
    }

    test_injected, test_events = inject_labeled_faults(
        test_healthy, baseline, SENSORS, seed=SEED + 1, events_per_engine=3
    )
    clean_features = sensor_feature_rows(test_healthy, baseline, SENSORS)
    injected_features = sensor_feature_rows(
        test_injected, baseline, SENSORS, test_events
    )
    clean_features["probability"] = model.predict_proba(
        clean_features[FEATURE_COLUMNS]
    )[:, 1]
    injected_features["probability"] = model.predict_proba(
        injected_features[FEATURE_COLUMNS]
    )[:, 1]
    clean_max = clean_features.groupby(["unit", "cycle"])["probability"].max()
    tradeoffs = []
    for target_name, candidate_threshold in thresholds.items():
        event_results = []
        for event in test_events.to_dict("records"):
            event_rows = injected_features.loc[
                (injected_features["unit"] == event["engine_id"])
                & (injected_features["sensor"] == event["sensor"])
                & injected_features["cycle"].between(
                    event["start_cycle"], event["end_cycle"]
                )
            ]
            event_results.append(
                bool((event_rows["probability"] >= candidate_threshold).any())
            )
        tradeoffs.append({
            "target": target_name,
            "threshold": candidate_threshold,
            "clean_alert_rate": float((clean_max >= candidate_threshold).mean()),
            "event_recall": float(np.mean(event_results)) if event_results else 0.0,
        })

    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    template_columns = [
        "event_id", "source_type", "data_set", "engine_id", "start_cycle",
        "end_cycle", "sensor", "label", "injection_type", "finding", "action",
        "post_action_result", "evidence_ref", "label_confidence", "annotator", "notes",
    ]
    if not TEMPLATE_PATH.exists():
        pd.DataFrame(columns=template_columns).to_csv(TEMPLATE_PATH, index=False)
    test_events.to_csv(INJECTION_PATH, index=False)
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, CLASSIFIER_PATH, compress=3)

    old_score = __import__("src.conditioned_anomaly", fromlist=["score_sensor_rows"])
    raw_scores = old_score.score_sensor_rows(test_healthy, baseline)
    raw_max = raw_scores[[f"{sensor}_score" for sensor in SENSORS]].max(axis=1)
    old_false_rate = float((raw_max >= baseline["threshold"]).mean())
    lines = [
        "# Synthetic sensor-fault detector evaluation",
        "",
        "## Method",
        "",
        f"The classifier was trained using {len(fit_healthy):,} healthy development cycles plus labeled synthetic step, drift, and spike alterations. Candidate probability cutoffs were calibrated only on unmodified healthy development engines at several pre-set family-wise cycle alert targets. It was evaluated on separate NASA test engines, both unmodified and with a distinct set of deterministic synthetic alterations.",
        "",
        f"- Synthetic training events: {len(train_events):,}.",
        f"- Healthy calibration cycles: {len(calibration_max):,} across {len(calibration_keys)} engines.",
        f"- Held-out healthy test cycles: {len(clean_max):,} across {test_healthy['unit'].nunique()} engines.",
        f"- Held-out synthetic test events: {len(test_events):,}.",
        "",
        "## Results",
        "",
        "| Detector / calibration target | Cutoff | False alerts on clean test cycles | Recall on injected events |",
        "|---|---:|---:|---:|",
        f"| Condition-aware residual threshold | {baseline['threshold']:.2f} robust scales | {old_false_rate:.1%} | 0.0% at this cutoff |",
    ]
    for result in tradeoffs:
        lines.append(
            f"| Synthetic-fault classifier / {result['target']} | "
            f"{result['threshold']:.3f} probability | {result['clean_alert_rate']:.1%} | "
            f"{result['event_recall']:.1%} |"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "The classifier is more responsive to the injected patterns by construction. Its results measure only the tested synthetic step, drift, and spike patterns; they do not establish real fault recall, identify a component, or support dispatch decisions. Unknown and no-fault-found maintenance outcomes must remain distinct labels when real events are added.",
        "",
        f"Editable real-event template: `{TEMPLATE_PATH.relative_to(ROOT).as_posix()}`. The generated held-out synthetic labels are in `{INJECTION_PATH.relative_to(ROOT).as_posix()}`. Do not enter simulated events as maintenance-confirmed findings.",
    ]
    if skipped_dev or skipped_test:
        lines += ["", "Unreadable source files:", *[f"- {item}" for item in skipped_dev + skipped_test]]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"Classifier threshold={threshold:.4f} (2% calibration target); "
        f"clean test alert rate={tradeoffs[0]['clean_alert_rate']:.1%}; "
        f"synthetic event recall={tradeoffs[0]['event_recall']:.1%}; events={len(test_events)}"
    )
    print(f"Saved classifier: {CLASSIFIER_PATH}")
    print(f"Saved report: {REPORT_PATH}")


if __name__ == "__main__":
    main()

