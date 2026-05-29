import numpy as np
import pytest
from sklearn.metrics import accuracy_score

from hcc.evaluation import bootstrap_ci, build_results_table, specificity_score


def test_specificity_score_exact():
    y_true = [0, 0, 0, 1, 1]
    y_pred = [0, 0, 1, 1, 0]
    # TN=2, FP=1, FN=1, TP=1 → specificity = 2/(2+1) = 0.6667
    assert abs(specificity_score(y_true, y_pred) - 2 / 3) < 1e-9


def test_specificity_score_perfect():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    assert specificity_score(y_true, y_pred) == 1.0


def test_bootstrap_ci_auroc_bounds():
    rng = np.random.default_rng(0)
    y_true = rng.choice([0, 1], size=100)
    y_prob = rng.uniform(0, 1, size=100)

    from sklearn.metrics import roc_auc_score
    lo, hi = bootstrap_ci(y_true, y_prob, roc_auc_score, n_bootstrap=200, seed=42)

    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_with_threshold():
    rng = np.random.default_rng(1)
    y_true = rng.choice([0, 1], size=80)
    y_prob = rng.uniform(0, 1, size=80)

    lo, hi = bootstrap_ci(y_true, y_prob, accuracy_score, threshold=0.5, n_bootstrap=200, seed=42)

    assert 0.0 <= lo <= hi <= 1.0


def test_build_results_table_columns():
    rng = np.random.default_rng(2)
    y_test = rng.choice([0, 1], size=60)
    pred_probs = {
        "ModelA": rng.uniform(0, 1, 60),
        "ModelB": rng.uniform(0, 1, 60),
    }
    thresholds = {"ModelA": 0.5, "ModelB": 0.4}

    df = build_results_table(pred_probs, thresholds, y_test)

    assert list(df["Model"]) == ["ModelA", "ModelB"]
    expected_cols = {"AUROC", "Accuracy", "F1 Score", "Precision", "Recall", "Specificity"}
    assert expected_cols.issubset(df.columns)
