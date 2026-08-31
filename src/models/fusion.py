"""
Score fusion between the L1 classifier (flow + TLS) and the DNS-only
classifier (L2 layer).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_score, recall_score


def fuse_scores(proba_l1: np.ndarray, proba_dns: np.ndarray, w_dns: float) -> np.ndarray:
    """P_fused = (1 - w_dns) * P_L1 + w_dns * P_DNS."""
    return (1.0 - w_dns) * proba_l1 + w_dns * proba_dns


def optimize_fusion_weight(
    proba_l1_val: np.ndarray,
    proba_dns_val: np.ndarray,
    y_val: np.ndarray,
    min_precision: float = 0.80,
    step: float = 0.1,
    threshold: float = 0.5,
) -> dict:
    """Optimizes w_dns in [0, 1] with step ``step`` on the validation set,
    maximizing recall subject to precision staying >= ``min_precision``.
    If no weight satisfies the constraint, falls back to the one with the
    highest precision."""
    best, fallback = None, None
    w = 0.0
    while w <= 1.0 + 1e-9:
        w_r = round(w, 4)
        proba = fuse_scores(proba_l1_val, proba_dns_val, w_r)
        y_pred = (proba >= threshold).astype(int)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        candidate = {"w_dns": w_r, "precision": float(prec), "recall": float(rec)}
        if fallback is None or prec > fallback["precision"]:
            fallback = candidate
        if prec >= min_precision and (best is None or rec > best["recall"]):
            best = candidate
        w += step
    return best if best is not None else fallback
