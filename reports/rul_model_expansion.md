# Expanded N-CMAPSS RUL model review

## Data used

The model is fitted on all readable `*_dev` arrays and evaluated only on their paired `*_test` arrays. Each time-step file is reduced to one row per engine-cycle by averaging the 14 measured sensor channels and four flight-condition channels. The feature set also uses per-cycle sensor rates and 10-cycle rolling means. Development and test engine IDs are checked for overlap within each file.

- Training: 4,535 engine-cycles from 9 dataset files and 60 dataset-specific engines.
- Held-out evaluation: 2,938 engine-cycles from 9 dataset files and 39 dataset-specific engines.
- Input features: 46 (14 measured sensors, 4 flight conditions, 14 sensor rates, and 14 rolling means).
- The internal virtual-sensor arrays and latent degradation-parameter arrays are not model inputs because their availability as operational telemetry is not established.
- Dashboard replay: 20 dataset-qualified engines and 1,524 cycles, balanced across the readable datasets. The RUL model still trains on all development engines.
- Replay data is saved to `data/processed/cycle_data_expanded.csv`.

## Held-out test results

Both models below are scored against the same 2,938 cycle-level rows from the paired NASA test arrays.

| Model | MAE (cycles) | RMSE (cycles) | R² |
|---|---:|---:|---:|
| Previous small-subset model | 17.00 | 22.21 | 0.100 |
| Expanded model | 11.63 | 15.68 | 0.552 |

- MAE: 11.63 cycles
- RMSE: 15.68 cycles
- R²: 0.552

| Test dataset | Engine-cycles | MAE (cycles) |
|---|---:|---:|
| N-CMAPSS_DS01-005 | 341 | 9.06 |
| N-CMAPSS_DS02-006 | 202 | 12.53 |
| N-CMAPSS_DS03-012 | 438 | 7.84 |
| N-CMAPSS_DS04 | 344 | 19.20 |
| N-CMAPSS_DS05 | 327 | 12.37 |
| N-CMAPSS_DS06 | 322 | 11.80 |
| N-CMAPSS_DS07 | 344 | 12.58 |
| N-CMAPSS_DS08a-009 | 383 | 11.01 |
| N-CMAPSS_DS08c-008 | 237 | 8.92 |

## Limitations

These are simulated-engine, held-out-file test metrics, not evidence of performance on operational aircraft. The cycle-level means discard within-cycle transients. Test results must be interpreted alongside per-dataset scores, and the RUL output remains experimental. This model does not validate the dashboard's anomaly flags or its maintenance guidance.

Files that could not be read:
- `N-CMAPSS_DS08d-010.h5`: Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)
