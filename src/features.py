import numpy as np
import pandas as pd

# Exact feature set and order the model was trained on, after
# correlation-based pruning (features with pairwise correlation > 0.90
# were removed) and Random Forest importance review. Note: step, amount,
# oldbalanceDest, and log_amount were all dropped at that stage — they
# are NOT part of the model input, even though some are used below as
# intermediate values to compute features that DID survive selection.
SELECTED_FEATURES = [
    "newbalanceOrig",
    "amount_sender_ratio",
    "dest_balance_missing",
    "emptied_account",
    "sender_balance_change",
    "type_CASH_OUT",
    "type_TRANSFER",
    "sender_error",
    "oldbalanceOrg",
    "log_oldbalanceOrg",
    "newbalanceDest",
    "receiver_balance_change",
    "receiver_error",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Single source of truth for feature engineering — mirrors the
    training notebook exactly, including which engineered columns were
    dropped by feature selection.

    Import this in BOTH the training pipeline and the serving API so a
    change here updates everywhere at once — never duplicate this logic.

    Expects raw columns: type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest. Returns exactly SELECTED_FEATURES,
    in the same order used for training.
    """
    df = df.copy()

    df["log_oldbalanceOrg"] = np.log1p(df["oldbalanceOrg"])

    df["sender_balance_change"] = df["oldbalanceOrg"] - df["newbalanceOrig"]
    df["receiver_balance_change"] = df["newbalanceDest"] - df["oldbalanceDest"]
    df["amount_sender_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)

    df["sender_error"] = df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"]
    df["receiver_error"] = (
        df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
    )

    df["emptied_account"] = (df["newbalanceOrig"] == 0).astype(int)
    df["dest_balance_missing"] = (
        (df["oldbalanceDest"] == 0) & (df["newbalanceDest"] == 0)
    ).astype(int)

    df["type_CASH_OUT"] = (df["type"] == "CASH_OUT").astype(int)
    df["type_TRANSFER"] = (df["type"] == "TRANSFER").astype(int)

    return df[SELECTED_FEATURES]