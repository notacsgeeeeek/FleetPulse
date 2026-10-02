# Condition-aware anomaly screening review

## Method

For each of the 14 measured sensors, a Random Forest baseline predicts the healthy-state sensor-cycle mean from altitude, Mach, TRA, inlet temperature, and flight class. N-CMAPSS encodes healthy operation as `hs=1` and unhealthy operation as `hs=0`; models fit only `hs=1` cycles from development engines. One engine from each dataset is held out from fitting to estimate residual bias and robust scale. The family-wide threshold is the 99th percentile of the largest standardized sensor residual on those held-out healthy cycles. A replay flags a sensor when its residual score crosses this calibrated threshold; the UI retains the existing current-crossing / 2-of-3 persistence rule.

- Fit healthy cycles: 1,147.
- Healthy calibration cycles: 193 across 9 held-out development engines.
- Calibrated family-wide threshold: 41.06 robust residual scales.
- Held-out NASA test cycles: 2,938 (947 healthy, 1,991 degraded).

## Held-out test results

- False-alert rate on healthy (`hs=1`) cycles: 1.9%.
- Detection rate on unhealthy (`hs=0`) cycles: 1.7%.
- Precision: 64.7%; recall: 1.7%; F1: 0.032.

## Limits

`hs` is a simulated healthy/degraded state label, not a confirmed sensor fault, component diagnosis, maintenance outcome, or aircraft dispatch label. The false-alert and detection rates therefore describe only this held-out N-CMAPSS test split. Cycle means discard within-cycle transients, and the model is experimental. This screening score cannot identify a failure mechanism or validate a repair.

The dashboard replay remains a balanced 20-engine development sample. Its RUL model and its independent RUL test evaluation are documented in `reports/rul_model_expansion.md`.
