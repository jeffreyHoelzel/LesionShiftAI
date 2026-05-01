"""metrics.py

Computes all binary metrics required for binary classifier models.
"""
from typing import Dict
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Computes binary classification metrics from labels and predicted probabilities.

    Parameters
    ------------
        y_true : np.ndarray
            Ground truth binary labels.
        y_prob : np.ndarray
            Predicted positive class probabilities.
        threshold : float
            Probability cutoff used to convert predicted probabilities into binary labels.

    Returns
    --------
        metrics : Dict[str, float]
            Dictionary containing accuracy, precision, recall, F1, ROC AUC, PR AUC, and confusion matrix counts.

    Raises
    -------
        ValueError
            Raised when y_true and y_prob have incompatible shapes or invalid values for metric computation.
    """
    y_pred = (y_prob >= threshold).astype(int)

    def _safe_auc(fn, yt, yp):
        try:
            return float(fn(yt, yp))
        except ValueError:
            return float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": _safe_auc(roc_auc_score, y_true, y_prob),
        "pr_auc": _safe_auc(average_precision_score, y_true, y_prob),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1])
    }
