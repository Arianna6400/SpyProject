"""
Standard binary classification metrics.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None) -> dict:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
    if y_proba is not None and len(set(np.asarray(y_true))) > 1:
        metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
    else:
        metrics["auc_roc"] = None
    return metrics
