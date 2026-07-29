from datetime import datetime, timezone

from api.risk_engine import THRESHOLDS
from src.features import SELECTED_FEATURES

MODEL_CARD_PATH = "models/model_card.md"

TEMPLATE = """# Model Card — Fraud Detection (XGBoost)

**Generated:** {generated_at}
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

## Features ({n_features} total, in training order)
{feature_list}

Full feature engineering logic: `src/features.py::engineer_features()`
(shared verbatim between training and the serving API — no duplicated logic).

## Evaluation Metrics (held-out time-based test set)
| Metric | Value |
|---|---|
| PR-AUC (primary) | {pr_auc} |
| ROC-AUC | {roc_auc} |
| Precision | {precision} |
| Recall | {recall} |
| F1 | {f1} |
| Accuracy | {accuracy} |

Confusion matrix: `assets/confusion_matrix_xgboost.png`
Correlation analysis: `assets/correlation_before.png`, `assets/correlation_after.png`

## Serving Thresholds
Decision thresholds applied in `api/risk_engine.py` (independent of the
model's internal 0.5 default; tunable via env vars without redeploying code):

| Probability range | Action |
|---|---|
| < {t_otp} | approve |
| {t_otp} – {t_review} | otp (step-up auth) |
| {t_review} – {t_block} | review (manual/analyst hold) |
| ≥ {t_block} | block |

"is_fraud" in API responses is true at or above the **review** threshold
({t_review}), not the model's raw 0.5 default — recall was prioritized
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
"""


def generate_model_card(metrics: dict, out_path: str = MODEL_CARD_PATH) -> None:
    """Writes model_card.md from evaluate.py's metrics dict. Call this
    at the end of evaluate.py's __main__ block so the card can never
    drift from the model actually being shipped.
    """
    feature_list = "\n".join(f"- `{f}`" for f in SELECTED_FEATURES)

    content = TEMPLATE.format(
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        n_features=len(SELECTED_FEATURES),
        feature_list=feature_list,
        pr_auc=f"{metrics['pr_auc']:.4f}",
        roc_auc=f"{metrics['roc_auc']:.4f}",
        precision=f"{metrics['precision']:.4f}",
        recall=f"{metrics['recall']:.4f}",
        f1=f"{metrics['f1']:.4f}",
        accuracy=f"{metrics['accuracy']:.4f}",
        t_otp=THRESHOLDS.otp,
        t_review=THRESHOLDS.review,
        t_block=THRESHOLDS.block,
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Model card written to {out_path}")


if __name__ == "__main__":
    from src.evaluate import evaluate

    metrics = evaluate()
    generate_model_card(metrics)