"""Compact preflight fleet monitor for the N-CMAPSS replay dataset."""

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from src.anomaly_classifier import FEATURE_COLUMNS, sensor_feature_rows
from src.conditioned_anomaly import score_sensor_rows


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cycle_data_expanded.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "rul_random_forest_expanded.pkl"
ANOMALY_MODEL_PATH = PROJECT_ROOT / "models" / "conditioned_sensor_baselines.pkl"
FAULT_CLASSIFIER_PATH = PROJECT_ROOT / "models" / "synthetic_fault_classifier.pkl"
SENSOR_COLUMNS = [
    "T24", "T30", "T48", "T50", "P15", "P2", "P21", "P24",
    "Ps30", "P40", "P50", "Nf", "Nc", "Wf",
]
SENSOR_GUIDANCE = {
    "T24": {
        "description": "LPC outlet total temperature",
        "unit": "°R",
        "compare": "P24, T30, Nc, Wf",
        "maintenance": "verify the T24 probe/indication circuit per the engine manual.",
    },
    "T30": {
        "description": "HPC outlet total temperature",
        "unit": "°R",
        "compare": "T24, Ps30, Nc, Wf",
        "maintenance": "verify the T30 indication circuit; use compressor troubleshooting if peers agree.",
    },
    "T48": {
        "description": "HPT outlet total temperature",
        "unit": "°R",
        "compare": "T30, T50, Nc, Wf",
        "maintenance": "review temperature trend/limits and verify the T48 circuit per the engine manual.",
    },
    "T50": {
        "description": "LPT outlet total temperature",
        "unit": "°R",
        "compare": "T30, T48, Nc, Wf",
        "maintenance": "review temperature trend/limits and verify the T50 circuit per the engine manual.",
    },
    "P15": {
        "description": "bypass-duct total pressure",
        "unit": "psia",
        "compare": "P21, Nf",
        "maintenance": "check the bypass pressure sensing path per the engine manual.",
    },
    "P2": {
        "description": "fan-inlet total pressure",
        "unit": "psia",
        "compare": "altitude, Mach, P21, Nf",
        "maintenance": "check the inlet pressure channel per the aircraft/engine manual.",
    },
    "P21": {
        "description": "fan-outlet total pressure",
        "unit": "psia",
        "compare": "P2, P24, Nf",
        "maintenance": "check the fan-outlet pressure channel per the engine manual.",
    },
    "P24": {
        "description": "LPC outlet total pressure",
        "unit": "psia",
        "compare": "P21, T24, T30, Nc",
        "maintenance": "check the LPC pressure channel; use compressor troubleshooting if peers agree.",
    },
    "Ps30": {
        "description": "HPC outlet static pressure",
        "unit": "psia",
        "compare": "P24, T30, Nc, Wf",
        "maintenance": "check the HPC pressure channel; use compressor troubleshooting if peers agree.",
    },
    "P40": {
        "description": "burner-outlet total pressure",
        "unit": "psia",
        "compare": "Ps30, P50, T48, Nc",
        "maintenance": "check the burner pressure channel per the engine manual.",
    },
    "P50": {
        "description": "LPT outlet total pressure",
        "unit": "psia",
        "compare": "P40, T50, T48, Nc",
        "maintenance": "check the turbine-outlet pressure channel; use turbine troubleshooting if peers agree.",
    },
    "Nf": {
        "description": "physical fan speed",
        "unit": "rpm",
        "compare": "Nc, Wf, T30",
        "maintenance": "verify the fan-speed indication; investigate performance if peers agree.",
    },
    "Nc": {
        "description": "physical core speed",
        "unit": "rpm",
        "compare": "Nf, Wf, T30, Ps30",
        "maintenance": "verify the core-speed indication; investigate core performance if peers agree.",
    },
    "Wf": {
        "description": "fuel flow",
        "unit": "pps",
        "compare": "Nf, Nc, T30, T48, and pressure channels",
        "maintenance": "if high flow lacks a matching speed/thrust response, check the flow indication and fuel-metering system per the engine manual.",
    },
}
FLIGHT_FEATURES = ["alt", "Mach", "TRA", "T2"]
MODEL_FEATURES = (
    SENSOR_COLUMNS
    + FLIGHT_FEATURES
    + [f"{sensor}_rate" for sensor in SENSOR_COLUMNS]
    + [f"{sensor}_rolling_mean_10" for sensor in SENSOR_COLUMNS]
)
BASELINE_LENGTH = 5


@st.cache_data
def load_cycle_data():
    data = pd.read_csv(DATA_PATH)
    return data.sort_values(["unit", "cycle"]).reset_index(drop=True)


@st.cache_resource
def load_rul_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_anomaly_model():
    return joblib.load(ANOMALY_MODEL_PATH)


@st.cache_resource
def load_fault_classifier():
    return joblib.load(FAULT_CLASSIFIER_PATH)


def start_replay():
    st.session_state.replay_playing = True
    st.session_state.show_engine_review = False
    st.session_state.show_rul = False
    st.session_state.rul_results_cycle = None
    st.session_state.rul_results = None


def pause_replay():
    st.session_state.replay_playing = False


def advance_replay():
    st.session_state.replay_cycle += 1


def reset_replay():
    st.session_state.replay_playing = False
    st.session_state.replay_cycle = 1
    st.session_state.show_engine_review = False
    st.session_state.show_rul = False
    st.session_state.rul_results_cycle = None
    st.session_state.rul_results = None


def finish_replay():
    st.session_state.replay_playing = False
    st.session_state.replay_cycle = st.session_state.max_replay_cycle


def inspect_engine(engine_id, replay_cycle, cycle_data):
    engine_rows = cycle_data[cycle_data["unit"] == engine_id]
    available = engine_rows[engine_rows["cycle"] <= replay_cycle]
    if available.empty:
        return None, []

    current = available.iloc[-1]
    flags = []
    recent = available.tail(3)
    engine_baseline = engine_rows.head(BASELINE_LENGTH)
    detector = load_anomaly_model()
    classifier = load_fault_classifier()
    scored = score_sensor_rows(recent, detector)
    sensor_examples = sensor_feature_rows(
        recent, detector, SENSOR_COLUMNS, scores=scored
    )
    sensor_examples["Synthetic-pattern score"] = classifier["model"].predict_proba(
        sensor_examples[FEATURE_COLUMNS]
    )[:, 1]
    probability_matrix = sensor_examples.pivot(
        index="cycle", columns="sensor", values="Synthetic-pattern score"
    )
    threshold = float(classifier["threshold"])
    for sensor in SENSOR_COLUMNS:
        recent_scores = probability_matrix[sensor]
        probability = float(recent_scores.iloc[-1])
        deviation = float(scored[f"{sensor}_score"].iloc[-1])
        pattern_hits = int((recent_scores >= threshold).sum())
        baseline_mean = float(engine_baseline[sensor].mean())
        baseline_scale = float(engine_baseline[sensor].std(ddof=1))
        if not np.isfinite(baseline_scale) or baseline_scale < 1e-8:
            baseline_scale = float("nan")
        baseline_deviations = (
            (recent[sensor].astype(float) - baseline_mean).abs() / baseline_scale
            if np.isfinite(baseline_scale)
            else pd.Series(0.0, index=recent.index)
        )
        baseline_deviation = float(baseline_deviations.iloc[-1])
        baseline_hits = int((baseline_deviations >= 2.5).sum())
        cycles_are_consecutive = recent["cycle"].diff().dropna().eq(1).all()
        pattern_persistence = pattern_hits if cycles_are_consecutive else 0
        baseline_persistence = baseline_hits if cycles_are_consecutive else 0
        pattern_confirmed = (
            len(recent) == 3 and cycles_are_consecutive and pattern_persistence >= 2
        )
        baseline_confirmed = (
            len(recent) == 3 and cycles_are_consecutive and baseline_persistence >= 2
        )
        pattern_flag = probability >= threshold or pattern_confirmed
        baseline_flag = baseline_deviation >= 2.5 or baseline_confirmed
        if pattern_flag or baseline_flag:
            if pattern_flag:
                persistence = pattern_persistence
                detection_source = "Synthetic-pattern classifier"
            else:
                persistence = baseline_persistence
                detection_source = "First-five-cycle baseline screen"
            residual = float(scored[f"{sensor}_residual"].iloc[-1])
            flags.append({
                "Engine": engine_id,
                "Cycle": int(current["cycle"]),
                "Sensor": sensor,
                "Reading": float(current[sensor]),
                "Expected (conditions)": float(scored[f"{sensor}_expected"].iloc[-1]),
                "Change vs expected": residual,
                "Direction": "Above" if residual >= 0 else "Below",
                "Deviation (robust σ)": deviation,
                "Baseline deviation (σ)": baseline_deviation,
                "Synthetic-pattern score": probability,
                "Pattern cutoff": threshold,
                "Detection source": detection_source,
                "Persistence": (
                    f"{persistence}/{len(recent)} available cycles"
                    if len(recent) < 3
                    else f"{persistence}/3 recent cycles"
                ),
                "Flag": (
                    "Current alert" if probability >= threshold
                    else "Sustained signal" if persistence == 3
                    else "Intermittent signal"
                ),
            })
    return current, flags


@st.cache_data(show_spinner=False)
def inspect_fleet(cycle_data, replay_cycle):
    """Score the fleet once at replay completion and reuse it for review."""
    snapshot = []
    all_flags = []
    for engine_id in sorted(cycle_data["unit"].dropna().unique()):
        current, flags = inspect_engine(engine_id, replay_cycle, cycle_data)
        if current is None:
            continue
        all_flags.extend(flags)
        snapshot.append({
            "Engine": engine_id,
            "Cycle": int(current["cycle"]),
            "Flagged sensors": ", ".join(
                flag["Sensor"] for flag in sorted(
                    flags,
                    key=lambda item: item["Deviation (robust σ)"],
                    reverse=True,
                )
            ) or "—",
            "Highest pattern score": max(
                (flag["Synthetic-pattern score"] for flag in flags),
                default=float("nan"),
            ),
            "Status": (
                "Current alert" if any(flag["Flag"] == "Current alert" for flag in flags)
                else "Sustained signal" if any(
                    flag["Flag"] == "Sustained signal" for flag in flags
                )
                else "Intermittent signal" if flags
                else "No threshold crossing"
            ),
        })
    snapshot_df = pd.DataFrame(snapshot).sort_values(
        ["Highest pattern score", "Engine"], ascending=[False, True]
    )
    return snapshot_df, all_flags


def sensor_pattern_guidance(flags):
    by_sensor = {flag["Sensor"]: flag for flag in flags}
    sensors = set(by_sensor)

    if {"T48", "T50"}.issubset(sensors):
        temp_flags = [by_sensor["T48"], by_sensor["T50"]]
        direction = (
            "both above" if all(flag["Direction"] == "Above" for flag in temp_flags)
            else "both below" if all(flag["Direction"] == "Below" for flag in temp_flags)
            else "moving in opposite directions"
        )
        summary = ", ".join(
            f"{flag['Sensor']} <span class=\"fp-emphasis\">{flag['Change vs expected']:+.1f}°R</span> "
            f"(<span class=\"fp-emphasis\">{flag['Deviation (robust σ)']:.2f} robust σ</span>, "
            f"<span class=\"fp-emphasis\">{flag['Persistence']}</span>)"
            for flag in temp_flags
        )
        return (
            f"**Turbine-temperature pattern:** T48 and T50 are {direction} their condition-adjusted expectations ({summary}). "
            "<span class=\"fp-action\">Compare:</span> T30, Nc, Wf and matched conditions; "
            "<span class=\"fp-action\">Maintenance:</span> review temperature trends/limits "
            "and verify both indication circuits per the engine manual.",
            {"T48", "T50"},
        )
    if {"Nc", "T30"}.issubset(sensors):
        pair = [by_sensor["Nc"], by_sensor["T30"]]
        direction = (
            "both above" if all(flag["Direction"] == "Above" for flag in pair)
            else "both below" if all(flag["Direction"] == "Below" for flag in pair)
            else "moving in opposite directions"
        )
        summary = ", ".join(
            f"{flag['Sensor']} <span class=\"fp-emphasis\">{flag['Change vs expected']:+.1f}"
            f"{SENSOR_GUIDANCE[flag['Sensor']]['unit']}</span> "
            f"(<span class=\"fp-emphasis\">{flag['Deviation (robust σ)']:.2f} robust σ</span>, "
            f"<span class=\"fp-emphasis\">{flag['Persistence']}</span>)"
            for flag in pair
        )
        return (
            f"**Core/compressor pattern:** Nc and T30 are {direction} their condition-adjusted expectations ({summary}). "
            "<span class=\"fp-action\">Compare:</span> Nf, Ps30, Wf and matched conditions; "
            "<span class=\"fp-action\">Maintenance:</span> verify both channels, then investigate "
            "core/compressor performance if corroborated.",
            {"Nc", "T30"},
        )
    return None


def sensor_follow_up(flag):
    sensor = flag["Sensor"]
    guidance = SENSOR_GUIDANCE[sensor]
    change = flag["Change vs expected"]
    return (
        f"**{sensor} · {guidance['description']}:** "
        f"<span class=\"fp-emphasis\">{change:+.1f} {guidance['unit']}</span> vs expected "
        f"(<span class=\"fp-emphasis\">{flag['Deviation (robust σ)']:.2f} robust σ</span>; "
        f"<span class=\"fp-emphasis\">{flag['Persistence']}</span>). "
        f"<span class=\"fp-action\">Compare:</span> {guidance['compare']} at matched conditions. "
        f"<span class=\"fp-action\">Maintenance:</span> {guidance['maintenance']}"
    )


def predict_fleet_rul(engine_ids, replay_cycle, cycle_data, all_flags, model):
    """Predict all before/what-if rows in batches using the already-scored flags."""
    flags_by_engine = {}
    for flag in all_flags:
        flags_by_engine.setdefault(flag["Engine"], set()).add(flag["Sensor"])

    metadata = []
    before_rows = []
    after_rows = []
    for engine_id in engine_ids:
        engine_rows = cycle_data[cycle_data["unit"] == engine_id].sort_values("cycle")
        available = engine_rows[engine_rows["cycle"] <= replay_cycle]
        if available.empty:
            continue

        current = available.iloc[-1]
        before = current[MODEL_FEATURES].astype(float).copy()
        after = before.copy()
        flagged_sensors = sorted(flags_by_engine.get(engine_id, set()))
        if flagged_sensors:
            baseline = engine_rows.head(BASELINE_LENGTH)
            previous = available[available["cycle"] < current["cycle"]]
            for sensor in flagged_sensors:
                corrected_reading = float(baseline[sensor].mean())
                after[sensor] = corrected_reading

                if not previous.empty:
                    previous_reading = float(previous.iloc[-1][sensor])
                    after[f"{sensor}_rate"] = corrected_reading - previous_reading

                recent_values = previous[sensor].tail(9).tolist() + [corrected_reading]
                after[f"{sensor}_rolling_mean_10"] = float(np.mean(recent_values))

        metadata.append({
            "Engine": engine_id,
            "Cycle": int(current["cycle"]),
            "Flagged sensors": ", ".join(flagged_sensors) or "None",
        })
        before_rows.append(before)
        after_rows.append(after)

    if not metadata:
        return pd.DataFrame(columns=[
            "Engine", "Cycle", "Flagged sensors", "Before (cycles)",
            "After what-if (cycles)", "Change (cycles)",
        ])

    # Batch inference avoids repeatedly invoking model setup for each engine.
    before_predictions = np.clip(model.predict(pd.DataFrame(before_rows)), 0, 100)
    after_predictions = np.clip(model.predict(pd.DataFrame(after_rows)), 0, 100)
    results = pd.DataFrame(metadata)
    results["Before (cycles)"] = before_predictions
    results["After what-if (cycles)"] = after_predictions
    results["Change (cycles)"] = after_predictions - before_predictions
    return results


def render_monitor(cycle_data):
    max_cycle = int(cycle_data["cycle"].max())
    st.session_state.setdefault("replay_cycle", 1)
    st.session_state.setdefault("replay_playing", False)
    st.session_state.setdefault("show_engine_review", False)
    st.session_state.setdefault("show_rul", False)
    st.session_state.setdefault("rul_results_cycle", None)
    st.session_state.setdefault("rul_results", None)
    st.session_state.max_replay_cycle = max_cycle

    with st.container(horizontal=True):
        st.button(
            "Start replay",
            icon=":material/play_arrow:",
            on_click=start_replay,
            disabled=st.session_state.replay_playing,
        )
        st.button(
            "Pause",
            icon=":material/pause:",
            on_click=pause_replay,
            disabled=not st.session_state.replay_playing,
        )
        st.button(
            "Next cycle",
            icon=":material/skip_next:",
            on_click=advance_replay,
            disabled=st.session_state.replay_playing,
        )
        st.button(
            "Reset",
            icon=":material/restart_alt:",
            on_click=reset_replay,
        )
        st.button(
            "Finish replay",
            icon=":material/fast_forward:",
            on_click=finish_replay,
        )

    @st.fragment(run_every="1s" if st.session_state.replay_playing else None)
    def live_view():
        cycle = min(int(st.session_state.replay_cycle), max_cycle)
        replay_complete = cycle >= max_cycle and not st.session_state.replay_playing
        replay_count = "Complete" if replay_complete else f"Cycle {cycle} / {max_cycle}"
        st.markdown(
            f'<div class="fp-cycle-counter"><div class="fp-cycle-label">Replay</div>'
            f'<div class="fp-cycle-value">{replay_count}</div></div>',
            unsafe_allow_html=True,
        )

        if not replay_complete:
            st.progress(cycle / max_cycle, text=f"Replay progress · {cycle} of {max_cycle} cycles")
            if cycle <= BASELINE_LENGTH:
                st.info("Sensor expectations account for recorded flight conditions and were learned from healthy N-CMAPSS cycles.")
            else:
                st.caption("Fleet findings and engine actions will be available when the replay is complete.")
        else:
            try:
                with st.spinner("Scoring the fleet replay…"):
                    snapshot_df, all_flags = inspect_fleet(cycle_data, cycle)
            except Exception as exc:
                st.error("Fleet scoring stopped before the review could be prepared.")
                with st.expander("Technical detail"):
                    st.code(str(exc))
                return
            st.success(f"Replay complete · {len(snapshot_df)} engines evaluated")
            if st.button("Review flagged engines", icon=":material/assignment_late:"):
                st.session_state.show_engine_review = True
                st.session_state.show_rul = False

            if st.session_state.show_engine_review:
                flagged_engines = snapshot_df.loc[
                    snapshot_df["Status"].isin(
                        ["Intermittent signal", "Sustained signal", "Current alert"]
                    ), "Engine"
                ].tolist()
                if not flagged_engines:
                    st.success("No engine crossed the demo threshold during this replay.")
                else:
                    st.metric("Engines requiring review", len(flagged_engines))
                    for engine_id in flagged_engines:
                        engine_flags = [f for f in all_flags if f["Engine"] == engine_id]
                        st.subheader(f"{engine_id} · {len(engine_flags)} sensor flag(s)")
                        evidence = pd.DataFrame(engine_flags).sort_values(
                            "Synthetic-pattern score", ascending=False
                        )
                        st.dataframe(
                            evidence,
                            width="stretch",
                            hide_index=True,
                            column_config={
                                "Reading": st.column_config.NumberColumn(format="%.4f"),
                                "Expected (conditions)": st.column_config.NumberColumn(format="%.4f"),
                                "Change vs expected": st.column_config.NumberColumn(format="%+.4f"),
                                "Deviation (robust σ)": st.column_config.NumberColumn(format="%.2f"),
                                "Baseline deviation (σ)": st.column_config.NumberColumn(format="%.2f"),
                                "Synthetic-pattern score": st.column_config.NumberColumn(format="%.1%"),
                                "Pattern cutoff": st.column_config.NumberColumn(format="%.1%"),
                            },
                        )
                        pattern = sensor_pattern_guidance(engine_flags)
                        summarized_sensors = set()
                        if pattern:
                            pattern_text, summarized_sensors = pattern
                            st.markdown(pattern_text, unsafe_allow_html=True)
                        for flag in engine_flags:
                            if flag["Sensor"] not in summarized_sensors:
                                st.markdown(sensor_follow_up(flag), unsafe_allow_html=True)

                    with st.expander("Sources And Use Limits"):
                        st.markdown(
                            "Channel definitions: [NASA N-CMAPSS](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf) · "
                            "indication checks: [FAA engine guidance](https://www.faa.gov/sites/faa.gov/files/aircraft/air_cert/design_approvals/engine_prop/engine_malf_famil.pdf) · "
                            "actual maintenance: the aircraft/engine's approved manual."
                        )

                if st.button(
                    "Predict Remaining Useful Life",
                    icon=":material/query_stats:",
                    disabled=not flagged_engines,
                    key="predict-rul-button",
                ):
                    st.session_state.show_rul = True

                if st.session_state.show_rul and flagged_engines:
                    try:
                        st.warning(
                            "What-if only: flagged readings are reset to baseline. This is not "
                            "a repair result or a dispatch decision."
                        )
                        if st.session_state.rul_results_cycle != cycle or st.session_state.rul_results is None:
                            with st.spinner(f"Calculating RUL estimates for {len(flagged_engines)} flagged engines…"):
                                model = load_rul_model()
                                st.session_state.rul_results = predict_fleet_rul(
                                    flagged_engines, cycle, cycle_data, all_flags, model
                                )
                                st.session_state.rul_results_cycle = cycle
                        rul_df = st.session_state.rul_results
                        rul_column_config = {
                            column: st.column_config.Column(alignment="center")
                            for column in rul_df.columns
                        }
                        rul_column_config.update({
                            "Before (cycles)": st.column_config.NumberColumn(
                                "Model estimate now (cycles)",
                                format="%.1f",
                                alignment="center",
                            ),
                            "After what-if (cycles)": st.column_config.NumberColumn(
                                "Baseline-normalized what-if (cycles)",
                                format="%.1f",
                                alignment="center",
                            ),
                            "Change (cycles)": st.column_config.NumberColumn(
                                format="%+.1f",
                                alignment="center",
                            ),
                        })
                        st.dataframe(
                            rul_df,
                            width="stretch",
                            hide_index=True,
                            column_config=rul_column_config,
                        )
                    except Exception as exc:
                        st.error(
                            "RUL prediction is unavailable because the saved model or its "
                            "runtime dependencies could not be loaded."
                        )
                        with st.expander("Technical detail"):
                            st.code(str(exc))

        if st.session_state.replay_playing:
            if cycle >= max_cycle:
                st.session_state.replay_playing = False
                st.rerun()
            else:
                st.session_state.replay_cycle = cycle + 1

    live_view()


def run():
    st.set_page_config(
        page_title="FleetPulse | Preflight Fleet Check",
        page_icon=":material/flight:",
        layout="wide",
    )
    st.markdown(
        """
        <style>
        html, body,
        .stApp,
        header[data-testid="stHeader"],
        div[data-testid="stAppViewContainer"],
        div[data-testid="stMain"],
        div[data-testid="stMainBlockContainer"] {
            background: #000000 !important;
            font-family: "Aptos", "Segoe UI", Arial, sans-serif;
            font-size: 18px;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li,
        div[data-testid="stCaptionContainer"],
        div[data-testid="stAlert"] p,
        div[data-testid="stSelectbox"] label {
            font-size: 1.08rem;
            line-height: 1.5;
        }

        div[data-testid="stExpander"] details summary {
            font-size: 1.5rem !important;
            line-height: 1.4;
        }

        div[data-testid="stMetricLabel"] {
            font-size: 1.05rem;
        }

        div[data-testid="stMetricValue"] {
            font-size: 2.4rem;
        }

        .fp-cycle-counter {
            margin: 0.5rem 0 1rem;
        }

        .fp-cycle-label {
            font-size: 1.05rem;
            line-height: 1.4;
        }

        .fp-cycle-value {
            font-size: 3.6rem;
            line-height: 1.15;
            font-weight: 500;
        }

        div[data-testid="stDataFrame"] {
            font-size: 1rem;
            background: #000000 !important;
        }

        div[data-testid="stButton"] button {
            background: #000000;
            color: #ffffff;
            border-color: #303030;
        }

        div[data-testid="stButton"] button:hover {
            background: #111111;
            color: #ffffff;
            border-color: #ff7078;
        }

        .st-key-predict-rul-button button {
            background: #b7efc5 !important;
            color: #102417 !important;
            border-color: #b7efc5 !important;
            font-size: 1.9rem !important;
        }

        .st-key-predict-rul-button button:hover {
            background: #d0f5d9 !important;
            color: #102417 !important;
            border-color: #d0f5d9 !important;
        }

        .st-key-predict-rul-button button:disabled {
            background: #b7efc5 !important;
            color: #102417 !important;
            opacity: 0.55;
        }

        .st-key-predict-rul-button button p,
        .st-key-predict-rul-button button span {
            color: #102417 !important;
        }

        .fp-emphasis,
        .fp-action {
            color: #ff7078 !important;
            font-weight: 700;
        }

        span[data-testid="stIconMaterial"] {
            font-family: "Material Symbols Rounded" !important;
            font-feature-settings: "liga";
        }

        div[data-testid="stHeading"] h1,
        h1 {
            font-size: 3.3rem;
            line-height: 1.12;
            font-weight: 700;
            letter-spacing: -0.025em;
        }

        div[data-testid="stHeading"] h2,
        h2 {
            font-size: 2.5rem;
            line-height: 1.18;
            font-weight: 650;
            letter-spacing: -0.02em;
        }

        div[data-testid="stHeading"] h3,
        h3 {
            font-size: 1.9rem;
            line-height: 1.22;
            font-weight: 600;
        }

        div[data-testid="stButton"] button {
            min-height: 4.5rem;
            padding: 1.1rem 1.8rem;
            border-radius: 0.7rem;
            font-size: 1.35rem;
            font-weight: 600;
        }

        div[data-testid="stButton"] button p {
            font-family: "Aptos", "Segoe UI", Arial, sans-serif;
            font-size: inherit;
            font-weight: inherit;
        }

        div[data-testid="stButton"] button:focus-visible {
            outline: 3px solid #78a9ff;
            outline-offset: 2px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    cycle_data = load_cycle_data()
    st.title("FleetPulse · Preflight Fleet Check", icon=":material/flight:")
    st.warning(
        "For demonstration only. Use the aircraft's approved flight release and "
        "operator procedures for dispatch decisions; this screen cannot clear an aircraft for flight."
    )
    render_monitor(cycle_data)
