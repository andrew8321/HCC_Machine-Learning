import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def specificity_score(y_true, y_pred):
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return tn / (tn + fp)


def bootstrap_ci(y_true, y_pred_prob, metric_fn, threshold=None, n_bootstrap=1000, seed=42):
    """
    Bootstrap 95% confidence interval for a metric.

    If threshold is None, metric_fn receives raw probabilities (e.g. roc_auc_score).
    If threshold is set, metric_fn receives binary predictions.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred_prob = np.asarray(y_pred_prob)
    n = len(y_true)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        yp = y_pred_prob[idx]
        if threshold is not None:
            score = metric_fn(yt, (yp >= threshold).astype(int))
        else:
            score = metric_fn(yt, yp)
        scores.append(score)
    return round(float(np.percentile(scores, 2.5)), 2), round(float(np.percentile(scores, 97.5)), 2)


def build_results_table(pred_probs, optimal_thresholds, y_test):
    """
    Compute point estimates + 95% bootstrap CI for all models.
    Returns a DataFrame with one row per model.
    """
    y_test = np.asarray(y_test)
    rows = []
    for model_name, y_prob in pred_probs.items():
        y_prob = np.asarray(y_prob)
        thresh = optimal_thresholds[model_name]
        y_pred = (y_prob >= thresh).astype(int)

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        spec = specificity_score(y_test, y_pred)

        acc_ci  = bootstrap_ci(y_test, y_prob, accuracy_score,  threshold=thresh)
        f1_ci   = bootstrap_ci(y_test, y_prob, f1_score,        threshold=thresh)
        auc_ci  = bootstrap_ci(y_test, y_prob, roc_auc_score)
        prec_ci = bootstrap_ci(y_test, y_prob, precision_score, threshold=thresh)
        rec_ci  = bootstrap_ci(y_test, y_prob, recall_score,    threshold=thresh)
        spec_ci = bootstrap_ci(y_test, y_prob, specificity_score, threshold=thresh)

        rows.append({
            "Model":           model_name,
            "Accuracy":        round(acc,  2),
            "F1 Score":        round(f1,   2),
            "AUROC":           round(auc,  2),
            "Precision":       round(prec, 2),
            "Recall":          round(rec,  2),
            "Specificity":     round(spec, 2),
            "Accuracy 95% CI":    f"[{acc_ci[0]}, {acc_ci[1]}]",
            "F1 Score 95% CI":    f"[{f1_ci[0]}, {f1_ci[1]}]",
            "AUROC 95% CI":       f"[{auc_ci[0]}, {auc_ci[1]}]",
            "Precision 95% CI":   f"[{prec_ci[0]}, {prec_ci[1]}]",
            "Recall 95% CI":      f"[{rec_ci[0]}, {rec_ci[1]}]",
            "Specificity 95% CI": f"[{spec_ci[0]}, {spec_ci[1]}]",
        })

    return pd.DataFrame(rows)
