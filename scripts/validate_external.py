"""
External validation of the LT (Liver Transplant) and SR (Surgical Resection) models
on a held-out external cohort.

Requires trained models and scalers produced by train_lt.py and train_surgery.py.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import (
    CATEGORICAL_COLS,
    EXT_DATA_PATH,
    MODELS_DIR,
    TARGET_COL,
)


def _load_external_data():
    xls = pd.ExcelFile(EXT_DATA_PATH)
    return pd.read_excel(xls, sheet_name="Sheet1")


def _compute_metrics(y_true, y_scores, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return pd.DataFrame({
        "Metric": ["AUROC", "Accuracy", "Precision", "Recall (Sensitivity)", "Specificity", "F1 Score"],
        "Value": [
            roc_auc_score(y_true, y_scores),
            accuracy_score(y_true, y_pred),
            report["1"]["precision"],
            report["1"]["recall"],
            tn / (tn + fp),
            f1_score(y_true, y_pred),
        ],
    })


def validate_lt(df):
    """Validate the LT (SVM) model on the external cohort."""
    numerical_cols = [c for c in df.columns if c not in CATEGORICAL_COLS + [TARGET_COL]]

    scaler = joblib.load(MODELS_DIR / "scaler_LT.pkl")
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])

    model = joblib.load(MODELS_DIR / "LT_SVM.pkl")
    with open(MODELS_DIR / "LT_optimal_thresholds.json") as f:
        thresholds = json.load(f)

    threshold = thresholds["SVM"]
    y_scores = model.predict_proba(df_scaled)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    df["survival_probability_LT"] = y_scores
    df["risk_group_LT"] = np.where(y_scores >= threshold, "High Risk", "Low Risk")

    y_true = df[TARGET_COL]
    metrics = _compute_metrics(y_true, y_scores, y_pred)
    print("\n=== LT Model (SVM) — External Validation ===")
    print(metrics.to_string(index=False))
    return df, metrics


def validate_surgery(df):
    """Validate the SR (CatBoost) model on the external cohort."""
    numerical_cols = [c for c in df.columns if c not in CATEGORICAL_COLS + [TARGET_COL]]

    scaler = joblib.load(MODELS_DIR / "scaler_SR.pkl")
    df_scaled = df.copy()
    df_scaled[numerical_cols] = scaler.transform(df[numerical_cols])

    model = joblib.load(MODELS_DIR / "SR_CatBoost.pkl")
    with open(MODELS_DIR / "SR_optimal_thresholds.json") as f:
        thresholds = json.load(f)

    threshold = thresholds["CatBoost"]
    y_scores = model.predict_proba(df_scaled)[:, 1]
    y_pred = (y_scores >= threshold).astype(int)

    df["survival_probability_surgery"] = y_scores
    df["risk_group_Surgery"] = np.where(y_scores >= threshold, "High Risk", "Low Risk")

    y_true = df[TARGET_COL]
    metrics = _compute_metrics(y_true, y_scores, y_pred)
    print("\n=== Surgery Model (CatBoost) — External Validation ===")
    print(metrics.to_string(index=False))
    return df, metrics


if __name__ == "__main__":
    df = _load_external_data()
    df, lt_metrics = validate_lt(df)
    df, sr_metrics = validate_surgery(df)
