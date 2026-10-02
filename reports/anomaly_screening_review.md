# Anomaly screening review

## Scope

This is a descriptive review of `data/processed/cycle_data.csv` using the dashboard's current sensor list and first-five-cycle baseline. For each engine and sensor, the measure is:

`abs(reading - first-five-cycle mean) / sample standard deviation`

The exploratory threshold is 2.5σ. The file contains six simulated engines with 553 engine-cycles total; cycle counts range from 75 to 100 per engine. The 14 sensor channels produce 298 sensor-cycle readings at or above 2.5σ after each engine's five-cycle baseline.

| Engine | 2.5σ readings | Trailing 2-of-3 windows | Trailing 3-of-3 confirmation readings |
| --- | ---: | ---: | ---: |
| 1 | 34 | 21 | 3 |
| 2 | 17 | 8 | 0 |
| 3 | 120 | 61 | 0 |
| 4 | 66 | 39 | 23 |
| 5 | 39 | 19 | 6 |
| 6 | 22 | 1 | 0 |
| **Total** | **298** | **149** | **32** |

The trailing-window counts overlap: a sustained run can contribute multiple consecutive windows. They are screening counts, not unique fault events. T50 and T48 account for 22 and 10 of the 32 three-in-a-row confirmation readings, respectively. Engine 3 has many isolated or interrupted threshold crossings, while engines 1, 4, and 5 have at least one three-in-a-row confirmation reading.

## Interpretation and limitation

This comparison shows that persistence changes how many readings reach a sustained-signal state. It does **not** establish false-alarm or missed-anomaly rates. The processed file has no confirmed sensor-fault labels or maintenance outcomes, so a threshold crossing cannot be called a false alarm, and an unflagged reading cannot be called a missed fault. Synthetic anomaly injection can check whether a rule reacts to a deliberately added signal, but it cannot validate real engine faults.

The five-cycle baseline is small, and flight conditions are not normalized. A future detector needs a condition-aware baseline with enough representative data and labeled maintenance or engineering outcomes. Thresholds and persistence settings need review with engine specialists before operational use.

## Dashboard behavior

The dashboard keeps single-cycle crossings visible as intermittent signals and shows how many of the latest three consecutive cycles crossed the threshold. A current crossing, or crossings in at least two of those three cycles, is shown for review. Three consecutive crossings are labeled sustained; a deviation magnitude at or above 4σ is labeled large deviation, with above/below direction shown separately. These are experimental screening labels, not diagnoses or dispatch decisions.

The RUL comparison is a counterfactual model scenario: flagged sensor readings and their derived rate/rolling features are reset to the first-five-cycle baseline before predicting again. It is not a repaired-engine prediction or measured post-maintenance result.
