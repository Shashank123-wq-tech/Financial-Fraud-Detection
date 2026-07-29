import pandas as pd
import pytest

from src.features import SELECTED_FEATURES, engineer_features


def make_row(**overrides):
    row = {
        "type": "TRANSFER",
        "amount": 1000.0,
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 4000.0,
        "oldbalanceDest": 0.0,
        "newbalanceDest": 1000.0,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_engineer_features_returns_exact_selected_columns():
    df = make_row()
    result = engineer_features(df)
    assert list(result.columns) == SELECTED_FEATURES


def test_emptied_account_flag():
    df = make_row(newbalanceOrig=0.0)
    result = engineer_features(df)
    assert result["emptied_account"].iloc[0] == 1


def test_type_one_hot_encoding():
    df = make_row(type="CASH_OUT")
    result = engineer_features(df)
    assert result["type_CASH_OUT"].iloc[0] == 1
    assert result["type_TRANSFER"].iloc[0] == 0


def test_dest_balance_missing_flag():
    df = make_row(oldbalanceDest=0.0, newbalanceDest=0.0)
    result = engineer_features(df)
    assert result["dest_balance_missing"].iloc[0] == 1


def test_sender_error_computation():
    df = make_row(oldbalanceOrg=5000.0, amount=1000.0, newbalanceOrig=4000.0)
    result = engineer_features(df)
    # 5000 - 1000 - 4000 = 0, a clean reconciliation
    assert result["sender_error"].iloc[0] == 0


def test_missing_column_raises_keyerror():
    df = make_row().drop(columns=["oldbalanceDest"])
    with pytest.raises(KeyError):
        engineer_features(df)