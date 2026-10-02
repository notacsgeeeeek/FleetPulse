# Synthetic sensor-fault detector evaluation

## Method

The classifier was trained using 1,147 healthy development cycles plus labeled synthetic step, drift, and spike alterations. Candidate probability cutoffs were calibrated only on unmodified healthy development engines at several pre-set family-wise cycle alert targets. It was evaluated on separate NASA test engines, both unmodified and with a distinct set of deterministic synthetic alterations.

- Synthetic training events: 255.
- Healthy calibration cycles: 193 across 9 engines.
- Held-out healthy test cycles: 947 across 39 engines.
- Held-out synthetic test events: 117.

## Results

| Detector / calibration target | Cutoff | False alerts on clean test cycles | Recall on injected events |
|---|---:|---:|---:|
| Condition-aware residual threshold | 41.06 robust scales | 1.9% | 0.0% at this cutoff |
| Synthetic-fault classifier / 2% target | 0.852 probability | 1.6% | 44.4% |
| Synthetic-fault classifier / 5% target | 0.660 probability | 7.5% | 62.4% |
| Synthetic-fault classifier / 10% target | 0.560 probability | 11.7% | 68.4% |

## Interpretation

The classifier is more responsive to the injected patterns by construction. Its results measure only the tested synthetic step, drift, and spike patterns; they do not establish real fault recall, identify a component, or support dispatch decisions. Unknown and no-fault-found maintenance outcomes must remain distinct labels when real events are added.

Editable real-event template: `data/labels/anomaly_event_template.csv`. The generated held-out synthetic labels are in `data/labels/synthetic_fault_injections.csv`. Do not enter simulated events as maintenance-confirmed findings.

Unreadable source files:
- N-CMAPSS_DS08d-010.h5: Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)
- N-CMAPSS_DS08d-010.h5: Unable to synchronously open file (truncated file: eof = 2885034848, sblock->base_addr = 0, stored_eof = 2885034880)
