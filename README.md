# FleetPulse

**A preflight fleet-health replay demo using NASA N-CMAPSS engine data.** FleetPulse lets viewers replay 20 simulated engines, review sensor signals with their supporting measurements, and explore an experimental remaining-useful-life (RUL) what-if.

> **Simulation demo only.** FleetPulse is not connected to aircraft sensors, does not diagnose faults, and cannot determine whether an aircraft is safe or cleared to fly. Use the aircraft's approved release and maintenance procedures for operational decisions.

## What you can explore

- **Replay a fleet:** step through the available cycles for 20 engines.
- **Review flagged signals:** see the engine and sensor, reading, condition-adjusted expected value, deviation measures, persistence, and which screening method raised the signal.
- **Compare investigation leads:** see related sensor channels and concise maintenance investigation prompts.
- **Explore an RUL what-if:** compare the model estimate at the current readings with a hypothetical estimate after flagged sensor readings are returned to their initial five-cycle baseline.

## Run locally

The repository includes the processed replay data and saved model artifacts needed for the demo. Raw multi-gigabyte N-CMAPSS HDF5 files are not included.

From PowerShell, clone the repository and start the app:

```powershell
git clone https://github.com/notacsgeeeeek/FleetPulse.git
cd FleetPulse
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run dashboard/app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in a browser. To run on macOS or Linux, activate the virtual environment with `source .venv/bin/activate` instead.

## How screening works

FleetPulse combines two different screens. A sensor can be flagged by either one:

1. **Synthetic-pattern classifier:** condition-aware sensor residuals and their recent patterns are scored against patterns learned from simulated step, drift, and spike injections. The current cutoff is approximately 0.852; a current crossing or crossings in at least two of the latest three consecutive cycles can be shown.
2. **Initial-baseline screen:** a reading at least 2.5 sample standard deviations from that engine's first-five-cycle sensor baseline can be shown immediately or after crossings in at least two of the latest three consecutive cycles. This screen is not adjusted for flight conditions.

The review table identifies the source of each signal. The classifier score is **not** the probability of a real fault. The baseline screen is exploratory; it may flag changes caused by operating conditions or other factors.

## RUL what-if limits

The RUL view is an experimental model output, not measured remaining life. Its what-if resets flagged sensor readings and their derived features to the engine's initial five-cycle baseline, then reruns the model. It does **not** simulate a repair or predict measured post-maintenance life.

## Data and evaluation

FleetPulse uses processed NASA N-CMAPSS simulated run-to-failure data. The repository includes the processed 20-engine replay sample, saved models, evaluation reports, and scripts. The raw HDF5 source files are excluded because they are multi-gigabyte files; retraining workflows require obtaining those files separately from NASA.

- [NASA N-CMAPSS dataset paper](https://ntrs.nasa.gov/api/citations/20205001125/downloads/Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf)
- [Fault-injection detector evaluation](reports/fault_injection_evaluation.md)
- [RUL model evaluation](reports/rul_model_expansion.md)

The synthetic-fault classifier was evaluated on held-out simulated engines with additional synthetic fault injections. That evaluation does not establish performance on real maintenance-confirmed faults. N-CMAPSS does not provide confirmed sensor-fault or maintenance-outcome labels, so real-world false-alarm and missed-fault rates are unknown.

## Project layout

```text
dashboard/   Streamlit app and preflight replay interface
data/        Processed replay data and event-label templates
models/      Saved anomaly and RUL model artifacts
reports/     Evaluation summaries
scripts/     Model training and evaluation scripts
src/         Shared sensor-feature and anomaly utilities
```
