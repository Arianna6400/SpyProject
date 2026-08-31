"""
Detection-latency metric: the minimum traffic-observation time needed for
the system to reach a given recall threshold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import DETECTION_WINDOWS_MIN, RECALL_THRESHOLD_DETECTION
from src.pipeline.build_dataset import extract_session_features


def _predict_recall_at_window(session_metas: list, y_true_by_id: dict, feature_columns: list, model, t_max_s: float) -> float:
    rows, ids = [], []
    for meta in session_metas:
        feats = extract_session_features(meta, t_max_s=t_max_s)
        rows.append({c: feats.get(c, np.nan) for c in feature_columns})
        ids.append(meta["session_id"])
    X = pd.DataFrame(rows, columns=feature_columns)
    y_true = np.array([y_true_by_id[i] for i in ids])
    if y_true.sum() == 0:
        return float("nan")
    y_pred = model.predict(X)
    true_positives = int(((y_pred == 1) & (y_true == 1)).sum())
    return true_positives / int(y_true.sum())


def compute_detection_latency(
    malicious_session_metas: list,
    y_true_by_id: dict,
    feature_columns: list,
    model,
    windows_min: list = DETECTION_WINDOWS_MIN,
    rho_star: float = RECALL_THRESHOLD_DETECTION,
) -> dict:
    """Re-evaluates an already-trained pipeline over growing traffic
    observation windows (t in {5, 10, 15, 20, 30} minutes by default),
    computing recall for each window and the minimum time T_det(rho*)
    needed to exceed the ``rho_star`` threshold."""
    recall_by_window = {
        t_min: _predict_recall_at_window(malicious_session_metas, y_true_by_id, feature_columns, model, t_min * 60)
        for t_min in windows_min
    }
    t_det = next(
        (t for t in sorted(windows_min) if not np.isnan(recall_by_window[t]) and recall_by_window[t] >= rho_star),
        None,
    )
    return {"recall_by_window_min": recall_by_window, "t_det_min": t_det, "rho_star": rho_star}
