import os
from dataclasses import dataclass


@dataclass(frozen=True)
class RiskThresholds:
    """Serving-time decision thresholds, separate from the model's own
    0.5 default. These are tuned against the PR curve (see
    04_model_evaluation.ipynb) to trade precision for recall in the
    zones that matter most for a real payments flow — cheap friction
    (OTP) is applied liberally, hard blocks are reserved for the
    highest-confidence predictions.

    Override via env vars at deploy time without a code change, e.g. to
    retune after a production drift alert without waiting on a full
    CI/CD cycle.
    """

    otp: float = float(os.getenv("THRESHOLD_OTP", "0.30"))
    review: float = float(os.getenv("THRESHOLD_REVIEW", "0.60"))
    block: float = float(os.getenv("THRESHOLD_BLOCK", "0.85"))

    def __post_init__(self):
        if not (0 < self.otp < self.review < self.block < 1):
            raise ValueError(
                "Thresholds must satisfy 0 < otp < review < block < 1, "
                f"got otp={self.otp}, review={self.review}, block={self.block}"
            )


THRESHOLDS = RiskThresholds()


def recommend_action(probability: float, thresholds: RiskThresholds = THRESHOLDS) -> str:
    """Maps a fraud probability to an action a transaction pipeline can
    act on directly, before funds leave the system.

    approve : below THRESHOLD_OTP        -> let the transaction proceed
    otp     : [THRESHOLD_OTP, REVIEW)    -> step-up auth (one-time password)
    review  : [THRESHOLD_REVIEW, BLOCK)  -> hold for manual/analyst review
    block   : >= THRESHOLD_BLOCK         -> reject outright
    """
    if probability >= thresholds.block:
        return "block"
    if probability >= thresholds.review:
        return "review"
    if probability >= thresholds.otp:
        return "otp"
    return "approve"


def is_fraud_decision(probability: float, thresholds: RiskThresholds = THRESHOLDS) -> bool:
    """Binary label used for metrics/logging — anything at 'review'
    severity or above counts as a positive prediction."""
    return probability >= thresholds.review