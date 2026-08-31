"""
Per-family threat analysis: computes detection metrics separately for
each family present in the test set, to check whether the DNS layer's
contribution depends on the threat's architectural characteristics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def per_family_recall(df_test: pd.DataFrame, y_pred: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    """For each family present in ``df_test`` (the ``family`` column),
    computes recall (for malicious families) or the false-positive rate
    (for benign traffic)."""
    out = df_test.copy()
    out["y_pred"] = y_pred
    out["y_proba"] = y_proba

    rows = []
    for family, group in out.groupby("family"):
        is_malicious_family = (group["label"] == "malicious").all()
        detection_rate = float((group["y_pred"] == 1).mean())
        rows.append({
            "family": family,
            "n_sessions": len(group),
            "recall": detection_rate if is_malicious_family else None,
            "false_positive_rate": detection_rate if not is_malicious_family else None,
            "mean_proba": float(group["y_proba"].mean()),
        })
    return pd.DataFrame(rows).sort_values("family").reset_index(drop=True)
