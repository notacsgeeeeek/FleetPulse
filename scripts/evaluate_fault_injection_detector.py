"""Stress-check condition-aware sensor scores with labeled synthetic faults.

Healthy official NASA test engines are kept unmodified for false-alert
measurement. Faults are injected into copies of their healthy cycles only.
This is a reproducible simulation benchmark, not field validation.
"""

from __future__ import annotations

import glob
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.train_expanded_rul_model import RAW_DIR, SENSORS, aggregate_cycles
from src.conditioned_anomaly import score_sensor_rows
from src.fault_injection import inject_labeled_faults


MODEL_PATH = ROOT / "models" / "conditioned_sensor_baselines.pkl"
LABEL_DIR = ROOT / "data" / "labels"
TEMPLATE_PATH = LABEL_DIR / "anomaly_event_template.csv"
INJECTION_PATH = LABEL_DIR / "synthetic_fault_injections.csv"
REPORT_PATH = ROOT / "reports" / "fault_injection_evaluation.md"
THRESHOLDS = (2.5, 3.0, 4.0, 5.0, 8.0, 12.0, 20.0, 41.06)


def _score(cycles: pd.DataFrame, detector: dict) -> pd.DataFrame:
    scores = score_sensor_rows(cycles, detector)
    max_scores = scores[[f"{sensor}_score" for sensor in SENSORS]].max(axis=1)
    scores["max_score"] = max_scores
    return scores


def main() -> None:
    detector = joblib.load(MODEL_PATH)
    test_parts = []
    skipped = []
    for path in sorted(glob.glob(str(RAW_DIR / "*.h5"))):
        try:
            test, _ = aggregate_cycles(path, "test")
            test_parts.append(test)
        except Exception as exc:
            skipped.append(f"{Path(path).name}: {str(exc).splitlines()[0]}")
    if not test_parts:
        raise RuntimeError("No readable NASA test arrays found")
    test = pd.concat(test_parts, ignore_index=True)
    healthy = test.loc[test["hs"] == 1].copy()
    if healthy.empty:
        raise RuntimeError("Official test split has no healthy cycles to benchmark")

    clean_scores = _score(healthy, detector)
    injected, events = inject_labeled_faults(healthy, detector, SENSORS)
    injected_scores = _score(injected, detector)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    template_columns = [
        "event_id", "source_type", "data_set", "engine_id", "start_cycle",
        "end_cycle", "sensor", "label", "injection_type", "finding", "action",
        "post_action_result", "evidence_ref", "label_confidence", "annotator", "notes",
    ]
    if not TEMPLATE_PATH.exists():
        pd.DataFrame(columns=template_columns).to_csv(TEMPLATE_PATH, index=False)
    events.to_csv(INJECTION_PATH, index=False)

    event_detected_by_threshold = {threshold: [] for threshold in THRESHOLDS}
    for event in events.to_dict("records"):
        event_rows = injected.loc[
            (injected["unit"] == event["engine_id"])
            & injected["cycle"].between(event["start_cycle"], event["end_cycle"])
        ]
        event_scores = injected_scores.loc[event_rows.index, f"{event['sensor']}_score"]
        for threshold in THRESHOLDS:
            event_detected_by_threshold[threshold].append(bool((event_scores >= threshold).any()))

    results = []
    for threshold in THRESHOLDS:
        clean_alerts = clean_scores["max_score"] >= threshold
        event_detection = event_detected_by_threshold[threshold]
        results.append({
            "threshold": threshold,
            "clean_false_alert_rate": float(clean_alerts.mean()),
            "injected_event_recall": float(np.mean(event_detection)) if event_detection else 0.0,
        })

    lines = [
        "# Synthetic sensor-fault injection evaluation",
        "",
        "## Method",
        "",
        f"The saved condition-aware detector was evaluated on {len(healthy):,} healthy (`hs=1`) cycles from the paired NASA test arrays. Those rows were kept unchanged for false-alert measurement. A separate copy received deterministic labeled step, gradual-drift, and single-cycle-spike injections (4–7 robust residual scales) on individual channels. Thresholds below are a sensitivity sweep, not a selected operational cutoff.",
        "",
        f"- Clean healthy cycles: {len(healthy):,} across {healthy['unit'].nunique()} test engines.",
        f"- Synthetic events generated: {len(events):,}.",
        "- All reported results are for this synthetic benchmark; injected patterns are not confirmed aircraft faults.",
        "",
        "## Threshold trade-off",
        "",
        "| Score threshold | Clean-cycle alert rate | Injected-event detection |",
        "|---:|---:|---:|",
    ]
    for result in results:
        lines.append(
            f"| {result['threshold']:.2f} | {result['clean_false_alert_rate']:.1%} | "
            f"{result['injected_event_recall']:.1%} |"
        )
    lines += [
        "",
        "## Use",
        "",
        f"Reusable annotation template: `{TEMPLATE_PATH.relative_to(ROOT).as_posix()}`. Generated injection labels: `{INJECTION_PATH.relative_to(ROOT).as_posix()}`. The template distinguishes synthetic injections from maintenance-confirmed findings. Add real events only from authorized, de-identified telemetry and reviewed maintenance outcomes; use `unknown` when the result was not confirmed.",
        "",
        "This sweep does not tune a threshold using the official test outcomes. Any threshold choice must be selected using development-engine calibration and then re-evaluated on a separate engine holdout. Synthetic injection performance is not evidence of real fault recall or aircraft dispatch suitability.",
    ]
    if skipped:
        lines += ["", "Files skipped:", *[f"- {item}" for item in skipped]]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Healthy test cycles: {len(healthy):,}; synthetic events: {len(events):,}")
    for result in results:
        print(
            f"threshold={result['threshold']:.2f}: "
            f"clean alert rate={result['clean_false_alert_rate']:.1%}, "
            f"injected event recall={result['injected_event_recall']:.1%}"
        )
    print(f"Saved report: {REPORT_PATH}")
    print(f"Saved labels: {INJECTION_PATH}")


if __name__ == "__main__":
    main()
