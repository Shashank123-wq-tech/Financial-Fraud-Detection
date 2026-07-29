import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features import engineer_features
from src.train import MODEL_PATH, build_xy, load_data, time_based_split

RAW_BALANCE_COLUMNS = [
    "amount", "oldbalanceOrg", "newbalanceOrig",
    "oldbalanceDest", "newbalanceDest", "isFraud",
]


def compute_metrics(y_test, y_pred, y_prob) -> dict:
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def plot_confusion_matrix(y_test, y_pred, out_path="assets/confusion_matrix_xgboost.png"):
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Blues")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_correlation_before_after(df):
    # Before: raw balance columns only, as they existed prior to any
    # engineered features being added.
    plt.figure(figsize=(6, 5))
    sns.heatmap(df[RAW_BALANCE_COLUMNS].corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.tight_layout()
    plt.savefig("assets/correlation_before.png", dpi=150)
    plt.close()

    # After: the final selected feature set plus the target, so isFraud's
    # correlation with each engineered feature is visible directly.
    engineered = engineer_features(df).copy()
    engineered["isFraud"] = df["isFraud"].values
    plt.figure(figsize=(7, 6))
    sns.heatmap(engineered.corr(), cmap="coolwarm")
    plt.tight_layout()
    plt.savefig("assets/correlation_after.png", dpi=150)
    plt.close()


def evaluate():
    df_scoped = load_data()
    _, test_df = time_based_split(df_scoped)
    X_test, y_test = build_xy(test_df)

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_prob)
    for name, value in metrics.items():
        print(f"{name:10s}: {value:.4f}")

    plot_confusion_matrix(y_test, y_pred)
    plot_correlation_before_after(test_df)

    return metrics


if __name__ == "__main__":
    evaluate()