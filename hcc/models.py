from collections import OrderedDict

import catboost as cb
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.svm import SVC


def pos_weight(y_train):
    return (y_train == 0).sum() / (y_train == 1).sum()


def build_models(scale_pos_weight, xgb_params, rf_n_estimators, cat_params=None):
    """
    Return an OrderedDict of unfitted estimators for the 6-model ensemble.

    scale_pos_weight: computed from y_train via pos_weight().
    xgb_params: cohort-specific XGBoost hyperparameter dict from config.
    rf_n_estimators: cohort-specific Random Forest tree count from config.
    cat_params: cohort-specific CatBoost params dict; defaults to a balanced config.
    """
    if cat_params is None:
        cat_params = dict(random_state=42, verbose=0, auto_class_weights="Balanced")

    lgbm_params = dict(
        n_estimators=200, learning_rate=0.03, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight, random_state=42,
    )

    return OrderedDict([
        ("Logistic Regression", LogisticRegression(
            class_weight="balanced", solver="liblinear"
        )),
        ("Random Forest", RandomForestClassifier(
            n_estimators=rf_n_estimators, class_weight="balanced", random_state=42
        )),
        ("XGBoost", xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, **xgb_params
        )),
        ("SVM", SVC(
            kernel="rbf", probability=True, class_weight="balanced", random_state=42
        )),
        ("LightGBM", lgb.LGBMClassifier(**lgbm_params)),
        ("CatBoost", cb.CatBoostClassifier(**cat_params)),
    ])


def train_models(models_dict, X_train, y_train):
    """Fit all models in-place and return {name: fitted_model}."""
    for model in models_dict.values():
        model.fit(X_train, y_train)
    return models_dict


def predict_probabilities(models_dict, X_test):
    """Return {model_name: predicted_probabilities_array} for the positive class."""
    return {name: m.predict_proba(X_test)[:, 1] for name, m in models_dict.items()}


def compute_optimal_thresholds(pred_probs, y_test):
    """F1-maximizing threshold from precision_recall_curve for each model."""
    thresholds = {}
    for name, y_prob in pred_probs.items():
        precision, recall, thresh_vals = precision_recall_curve(y_test, y_prob)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        thresholds[name] = float(thresh_vals[np.argmax(f1)])
    return thresholds
