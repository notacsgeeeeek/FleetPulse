# FleetPulse — Codex Project Handoff

Last updated: 2026-10-03

## Start here

This document captures the known FleetPulse project context and the reported implementation state. Some modeling and data facts below come from the prior project conversation and have not been independently recomputed during this handoff. Before changing code, inspect the repository and verify those claims against the current files and artifacts.

## Project purpose and intended user

FleetPulse is intended to be a concise pre-flight fleet and engine monitoring system that a pilot can use to spot sensor anomalies, see which engine is flagged, understand the measurement behind the flag, and know which approved procedure or maintenance contact to consult. The main screen should present fleet status and actionable evidence at a glance, with optional engine and sensor drill-down. It must not diagnose a failure, issue repair instructions, or clear an aircraft for flight. Maintenance, reliability, and engineering staff remain important users for follow-up investigation.

## Keep the projects distinct

FleetPulse is the engine health investigation and maintenance decision-support project in this repository. AeroPredict AI is a separate project/concept. Keep their code, datasets, branding, assumptions, and documentation separate unless the user explicitly requests a shared component. Do not import AeroPredict material into FleetPulse by assumption.

## Dataset provenance and limitations

The raw data is NASA's N-CMAPSS (Commercial Modular Aero-Propulsion System Simulation) run-to-failure simulation data. The repository includes the NASA dataset files and the accompanying `Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf` and loading/exploration notebook under `data/raw/`.

This is simulated/synthetic engine run-to-failure data generated under realistic flight conditions. It is not Airbus operational data and is not Airbus proprietary engine telemetry. Any product claims, evaluation, and UI labels must preserve this distinction. Validate provenance, dataset configuration, and the exact interpretation of fields from the included NASA documentation before making claims beyond this summary.

## Reported inspection facts and processed data

The source HDF5 datasets use grouped arrays including `A_dev`, `T_dev`, `W_dev`, `X_s_dev`, `X_v_dev`, and `Y_dev`. The current `src/data_loader.py` defines metadata names for A, T, W, X_s, and X_v; `load_dataset(file_name)` reads the `*_dev` arrays and returns them in a dictionary. The A columns are `unit`, `cycle`, `Fc`, and `hs`; the sensor and operating-condition names are declared in that file.

The original processed file is `data/processed/cycle_data.csv` (553 rows × 50 columns) and remains available as the prior six-engine replay extract. The current dashboard uses `data/processed/cycle_data_expanded.csv`, generated from readable N-CMAPSS development arrays. It contains the 20-engine replay sample selected from 4,535 available engine-cycles across 60 dataset-qualified engines, sensor readings, target `rul`, flight conditions, sensor rates, and 10-cycle rolling means. `N-CMAPSS_DS08d-010.h5` was excluded because it is truncated.

**Leakage concern:** `lifecycle_progress` is present in the processed data but is intentionally not in the 46-feature list used by the current dashboard model. Recheck all training code and feature selection to confirm it was excluded from fitting. Because it can encode position in an engine's lifecycle, it may leak target information. Do not include it as a predictor without a defensible deployment-time rationale and leakage-safe validation.

## Current repository structure

The root currently contains `.github/`, `.venv/`, `assets/`, `dashboard/`, `data/`, `docs/`, `models/`, `notebooks/`, `reports/`, `scripts/`, `src/`, and `tests/`, plus `.gitignore`, `README.md`, and `requirements.txt`. The last three root files were empty when inspected; verify before editing. The `.venv/` directory is local environment state and should not be committed.

Key files currently present:

- `dashboard/app.py` — Streamlit entry point for the pre-flight fleet monitor.
- `dashboard/preflight_monitor.py` — compact replay, anomaly evidence, and engine detail UI.
- `src/data_loader.py` — N-CMAPSS HDF5 loader.
- `scripts/train_expanded_rul_model.py` — cycle-level training/evaluation pipeline for the readable N-CMAPSS development/test arrays.
- `notebooks/01_data_exploration.ipynb` — exploration and reported modeling work.
- `data/raw/` — N-CMAPSS HDF5 files, source PDF, and example loading notebook.
- `data/processed/cycle_data.csv` — original six-engine cycle-level replay extract.
- `data/processed/cycle_data_expanded.csv` — current dashboard replay data: a balanced 20-engine sample from the readable N-CMAPSS development arrays.
- `models/rul_random_forest.pkl` — preserved earlier Random Forest artifact.
- `models/rul_random_forest_expanded.pkl` — expanded model currently used by the dashboard.
- `reports/rul_model_expansion.md` — expanded model's held-out evaluation results.

Inspect actual directory contents and Git status before deciding whether folders are empty or whether files have changed since this snapshot.

## Environment and setup

The project has a Windows virtual environment at `.venv/`, configured for Python 3.13.7. The compact dashboard currently needs Streamlit and pandas; the `requirements.txt` file records these runtime dependencies. Other project workflows may use h5py, NumPy, scikit-learn, joblib, and notebook dependencies.

Typical launch from the repository root after confirming the environment and dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run dashboard/app.py
```

## Loader responsibility and restoration

`src/data_loader.py` is currently restored as a data-loading module. It imports `h5py` and pandas, declares the N-CMAPSS array column names, opens an HDF5 file, and returns the development arrays. It is not dashboard code. The prior conversation records an accidental paste of Streamlit code into the loader followed by restoring the loader. The current file inspected here contains the expected loader implementation. Preserve that separation and check the working tree before further edits.

## Experimental RUL research artifact

The dashboard currently loads `models/rul_random_forest_expanded.pkl`. `scripts/train_expanded_rul_model.py` aggregates each readable N-CMAPSS file's development and test arrays to one row per engine-cycle. The model uses 14 measured sensor means, four flight-condition means, 14 per-cycle sensor rates, and 14 rolling means. It fits the development arrays and holds out the paired test arrays; engine IDs are checked for overlap within each file. The current run used 4,535 development cycles across nine files and evaluated 2,938 test cycles across 39 dataset-specific engines. Overall held-out MAE is 11.63 cycles, RMSE 15.68, and R² 0.552. On those same test cycles, the earlier model in `models/rul_random_forest.pkl` scored MAE 17.00, RMSE 22.21, and R² 0.100. See `reports/rul_model_expansion.md` for per-file scores and exclusions.

`N-CMAPSS_DS08d-010.h5` was skipped because its HDF5 file is truncated. The internal virtual-sensor arrays and latent degradation-parameter arrays were not model inputs because their availability as operational telemetry is not established. The processed replay shown in the dashboard remains `data/processed/cycle_data.csv`; expanding the RUL training data does not expand the replay dataset or validate anomaly detection. These held-out metrics are experimental simulated-data results, not operational performance. The dashboard loads the model only when the user requests the RUL view; its baseline-reset estimate remains a counterfactual, not measured post-repair RUL.

These are experimental baseline metrics from a very small engine-level split, not evidence of production readiness or generalization across the fleet. Verify the notebook, saved model, target construction, and exact validation process before quoting or comparing results. The model prediction is an estimate, not an exact remaining-life measurement; a displayed value such as 1.4 cycles should be contextualized by the large validation error.

## Saved RUL model features (46; used for the on-demand RUL what-if)

`dashboard/preflight_monitor.py` assembles these model inputs in `MODEL_FEATURES`:

1. Raw sensors (14): `T24`, `T30`, `T48`, `T50`, `P15`, `P2`, `P21`, `P24`, `Ps30`, `P40`, `P50`, `Nf`, `Nc`, `Wf`.
2. Flight conditions (4): `alt`, `Mach`, `TRA`, `T2`.
3. Per-cycle rates (14): `T24_rate`, `T30_rate`, `T48_rate`, `T50_rate`, `P15_rate`, `P2_rate`, `P21_rate`, `P24_rate`, `Ps30_rate`, `P40_rate`, `P50_rate`, `Nf_rate`, `Nc_rate`, `Wf_rate`.
4. 10-cycle rolling means (14): `T24_rolling_mean_10`, `T30_rolling_mean_10`, `T48_rolling_mean_10`, `T50_rolling_mean_10`, `P15_rolling_mean_10`, `P2_rolling_mean_10`, `P21_rolling_mean_10`, `P24_rolling_mean_10`, `Ps30_rolling_mean_10`, `P40_rolling_mean_10`, `P50_rolling_mean_10`, `Nf_rolling_mean_10`, `Nc_rolling_mean_10`, `Wf_rolling_mean_10`.

The expanded training script and dashboard use the same feature order. For the hypothetical post-normalization row, the app resets flagged sensor values and recomputes their rate/rolling features; that is a counterfactual scenario, not evidence of a completed repair.

## Current Streamlit capabilities

The current app entry point `dashboard/app.py` calls `dashboard/preflight_monitor.py`. It replays a balanced 20-engine sample from the expanded development data and exposes a “Review flagged engines” action after replay completion. Replay controls are outside the one-second fragment so their session-state changes reconfigure the timer correctly. Fleet scoring is deferred until completion, cached, and reused by the review panel. The RUL model trains on all 60 development engines. Sensor evidence uses condition-aware Random Forest expectations from altitude, Mach, TRA, inlet temperature, and flight class. A sensor-level Random Forest classifier is trained on labeled synthetic step, drift, and spike injections into healthy N-CMAPSS development cycles; its family-wise score cutoff is calibrated on separate healthy development engines. For review, the dashboard combines current or persistent synthetic-pattern matches with a separate first-five-cycle baseline screen (absolute deviation at least 2.5 sample standard deviations, currently or in at least two of the latest three consecutive cycles). Baseline-only results are identified in the detection-source column; this screen is not condition-adjusted and is not a confirmed fault detector. The UI gives the expected reading, residual, both measures, persistence, detection source, related channels to compare, and a sensor-specific maintenance investigation lead. The classifier score is not a real-fault probability.

The source basis is the NASA N-CMAPSS measurement table plus FAA general guidance on interpreting engine indications and testing temperature indication systems. These sources do not provide type-specific repair tasks for the aircraft/engine represented by an anomaly. To show manufacturer-approved fault-isolation or repair instructions, FleetPulse must identify the exact aircraft/engine configuration and connect each action to its current OEM maintenance manual or operator-approved troubleshooting procedure. The UI states this limit once rather than repeating the same paragraph for each engine.

The dashboard's “Predict remaining useful life” action shows an experimental model estimate and a baseline-normalized counterfactual. In the what-if, flagged sensor readings and derived rate/rolling features are reset to the initial five-cycle baseline. It does not model a repair or measured post-maintenance RUL. This limitation is shown before the estimates. The action reuses the replay's existing flag results, batches before/what-if model inference, and retains results for the current replay cycle in session state so later reruns do not repeat the calculation.

The condition-only residual model had a 1.9% alert rate on healthy cycles and detected 1.7% of unhealthy-state cycles at its calibrated cutoff; see `reports/conditioned_anomaly_evaluation.md`. The current synthetic-fault classifier evaluation is in `reports/fault_injection_evaluation.md`: at the 2% calibration target, it produced a 1.6% alert rate on held-out healthy test cycles and detected 44.4% of held-out injected events. These are step/drift/spike simulation results only. The separate first-five-cycle baseline screen is exploratory and condition-unadjusted; it has not been calibrated or validated as a fault detector. `hs` is not a confirmed sensor-fault label, and the dataset contains no maintenance outcomes, so real false-alarm and missed-fault rates cannot be established. `data/labels/anomaly_event_template.csv` is the schema for future authorized, de-identified maintenance events; generated synthetic records must not be treated as field labels. The app is not an aircraft release or airworthiness tool.

## Desired architecture

Build toward this pilot-oriented flow, with clear module boundaries and explainable evidence passed between stages:

```text
Authorized telemetry and aircraft status
  -> Data quality and flight-phase checks
  -> Condition-aware anomaly detection
  -> Explainable sensor evidence and thresholds
  -> Approved-procedure references / maintenance-control escalation
  -> Concise pre-flight fleet UI

Optional validated RUL analytics should remain secondary.
```

The Streamlit layer should orchestrate and present these capabilities, not absorb the data-loading or modeling responsibilities into one file.

## Next priorities

Proceed incrementally toward the pilot's pre-flight workflow:

1. Replace or supplement synthetic injection labels with authorized, de-identified maintenance-confirmed events; evaluate by engine-held-out splits and track event recall, alert rate, and detection lead time.
2. Identify an authorized operational telemetry and aircraft-status data source; do not label dataset replay as live monitoring.
3. Map any displayed next steps to approved aircraft/operator procedures and maintenance-control workflows.
4. Validate the interface and alert behavior with pilots and maintenance specialists before operational use.
5. Keep experimental RUL analytics secondary unless they are validated and add clear pre-flight value.

## Development rules

- Preserve the separation between FleetPulse and AeroPredict AI.
- Do not overwrite `src/data_loader.py` with dashboard code.
- Inspect the existing repository and working tree before changing files.
- Avoid destructive rewrites; make small, reviewable changes and preserve working artifacts.
- Ask the user before destructive changes.
- Keep dataset provenance and model limitations explicit.
- Do not claim the synthetic N-CMAPSS simulations are Airbus proprietary telemetry.
- Do not assume old conversation metrics or structure facts remain true without checking the current repository.

## Goal

Build an interactive pre-flight fleet monitor that makes flags, reasons, and approved follow-up steps clear without overwhelming pilots. Any operational use requires live authorized data, validated thresholds, and integration with approved aircraft/operator procedures. Until then, present it as a simulation demonstration only.

## Instruction for the next Codex session

Read this handoff first, inspect the repository and its current working-tree state, verify the data, model, loader, and app claims above, and then continue from the next priority without unnecessarily redoing completed work.
