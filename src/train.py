import json

import joblib
import pandas as pd
from xgboost import XGBClassifier

from src.features import SELECTED_FEATURES, engineer_features

DATA_PATH = "data/Fraud"
MODEL_PATH = "models/xgboost_fraud_model.pkl"
FEATURE_COLUMNS_PATH = "models/feature_columns.json"


def load_data(csv_path: str = DATA_PATH) -> pd.DataFrame:
    """Loads the raw dataset and scopes it to TRANSFER/CASH_OUT only —
    the only transaction types where fraud occurs, verified empirically
    (see the Problem Statement in the README)."""
    df = pd.read_csv(csv_path)
    return df[df["type"].isin(["TRANSFER", "CASH_OUT"])].copy()


def time_based_split(df_scoped: pd.DataFrame, test_size: float = 0.2):
    """Splits by `step` (time), not randomly, so evaluation reflects how
    the model would perform on transactions it hasn't seen yet — a random
    split would leak future information into training."""
    cutoff = df_scoped["step"].quantile(1 - test_size)
    train_df = df_scoped[df_scoped["step"] <= cutoff]
    test_df = df_scoped[df_scoped["step"] > cutoff]
    return train_df, test_df


def build_xy(df: pd.DataFrame):
    X = engineer_features(df)
    y = df["isFraud"].values
    return X, y


def train_model(X_train, y_train) -> XGBClassifier:
    negative = (y_train == 0).sum()
    positive = (y_train == 1).sum()
    scale_pos_weight = negative / positive
    print(f"scale_pos_weight: {scale_pos_weight:.2f}")

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    df_scoped = load_data()
    train_df, test_df = time_based_split(df_scoped)

    X_train, y_train = build_xy(train_df)

    model = train_model(X_train, y_train)

    joblib.dump(model, MODEL_PATH)
    with open(FEATURE_COLUMNS_PATH, "w") as f:
        json.dump(SELECTED_FEATURES, f, indent=2)

    print(f"Model saved to {MODEL_PATH}")
    print(f"Feature columns saved to {FEATURE_COLUMNS_PATH}")


if __name__ == "__main__":
    main()