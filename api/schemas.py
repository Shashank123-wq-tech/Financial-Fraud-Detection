from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class Transaction(BaseModel):
    """Raw transaction payload, matching the columns engineer_features()
    expects. Only TRANSFER and CASH_OUT are accepted — the model was
    trained exclusively on these two types, per the PaySim finding that
    fraud never occurs in PAYMENT, CASH_IN, or DEBIT transactions.
    """

    type: Literal["TRANSFER", "CASH_OUT"] = Field(
        ..., description="Transaction type. Only TRANSFER and CASH_OUT are scored."
    )
    amount: float = Field(..., gt=0, description="Transaction amount")
    oldbalanceOrg: float = Field(..., ge=0, description="Sender balance before the transaction")
    newbalanceOrig: float = Field(..., ge=0, description="Sender balance after the transaction")
    oldbalanceDest: float = Field(..., ge=0, description="Receiver balance before the transaction")
    newbalanceDest: float = Field(..., ge=0, description="Receiver balance after the transaction")

    # Optional metadata, not used by the model, but useful for logging,
    # tracing, and joining predictions back to source systems.
    transaction_id: Optional[str] = Field(None, description="Upstream transaction/reference ID")
    nameOrig: Optional[str] = Field(None, description="Sender account ID")
    nameDest: Optional[str] = Field(None, description="Receiver account ID")

    @field_validator("newbalanceOrig")
    @classmethod
    def _sanity_check_amount(cls, v, info):
        # Not a hard rejection — real-world data has messy balances —
        # but this stays a no-op validator hook if stricter checks are
        # ever needed (e.g. flagging negative computed deltas upstream).
        return v


class PredictionResponse(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float = Field(..., description="Model's predicted probability of fraud, 0-1")
    is_fraud: bool = Field(..., description="Binary decision at the serving threshold")
    recommended_action: Literal["approve", "otp", "review", "block"]
    threshold_used: float
    model_version: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str