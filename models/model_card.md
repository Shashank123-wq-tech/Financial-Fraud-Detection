# Model Card — Fraud Detection (XGBoost)

**Generated:** 2026-07-27 12:15 UTC
**Model artifact:** `models/xgboost_fraud_model.pkl`
**Feature schema:** `models/feature_columns.json`

## Intended Use
Binary classifier scoring **TRANSFER** and **CASH_OUT** transactions for
fraud, at authorization time — before funds leave the system. Not intended
for PAYMENT, CASH_IN, or DEBIT transactions; the model was never trained
on them because fraud does not occur in those types in the source data.

## Training Data
- Source: PaySim synthetic mobile money simulator dataset
- Scope: rows filtered to `type in [TRANSFER, CASH_OUT]` only
- Split: **time-based** (by `step`), not random — train on earlier steps,
  evaluate on later steps, to reflect real deployment (scoring transactions
  the model hasn't seen, not just held-out random rows)
- Class balance handled via `scale_pos_weight` in XGBoost, not resampling

## Features (13 total, in training order)
- `newbalanceOrig`
- `amount_sender_ratio`
- `dest_balance_missing`
- `emptied_account`
- `sender_balance_change`
- `type_CASH_OUT`
- `type_TRANSFER`
- `sender_error`
- `oldbalanceOrg`
- `log_oldbalanceOrg`
- `newbalanceDest`
- `receiver_balance_change`
- `receiver_error`

Full feature engineering logic: `src/features.py::engineer_features()`
(shared verbatim between training and the serving API — no duplicated logic).

## Evaluation Metrics (held-out time-based test set)
| Metric | Value |
|---|---|
| PR-AUC (primary) | 1.0000 |
| ROC-AUC | 1.0000 |
| Precision | 0.9984 |
| Recall | 0.9998 |
| F1 | 0.9991 |
| Accuracy | 1.0000 |

Confusion matrix: `assets/confusion_matrix_xgboost.png`
Correlation analysis: `assets/correlation_before.png`, `assets/correlation_after.png`

## Serving Thresholds
Decision thresholds applied in `api/risk_engine.py` (independent of the
model's internal 0.5 default; tunable via env vars without redeploying code):

| Probability range | Action |
|---|---|
| < 0.3 | approve |
| 0.3 – 0.6 | otp (step-up auth) |
| 0.6 – 0.85 | review (manual/analyst hold) |
| ≥ 0.85 | block |

"is_fraud" in API responses is true at or above the **review** threshold
(0.6), not the model's raw 0.5 default — recall was prioritized
over precision per the project objective, so this is deliberately more
sensitive than a default classifier cutoff.

## Known Limitations
- Trained on **synthetic** data (PaySim); real transaction patterns,
  fraud rings, and adversarial behavior may differ from the simulation.
- No fairness/bias audit performed across account segments.
- No adversarial robustness testing (e.g. structured amounts just under
  detection thresholds).
- Feature set assumes clean, non-null balance fields; malformed upstream
  data (e.g. missing `oldbalanceDest`) will fail feature engineering and
  return a 422 from the API rather than a silent prediction.
- Not evaluated for concept drift; no automated retraining trigger yet —
  recommend periodic re-evaluation against fresh production data and
  monitoring PR-AUC over time, not just error rate.

## Retraining
Run `src/train.py` then `src/evaluate.py`. Re-running this script
(`generate_model_card.py`) after evaluation keeps this card in sync with
whatever model is currently in `models/xgboost_fraud_model.pkl`.
