"""
Nested k-fold cross-validation: for each outer fold, hyperparameter
selection (GridSearchCV) happens only on that fold's training data, and
evaluation happens on the held-out outer test fold. This avoids the
optimism/fragility of a single fixed split on small datasets, where one
"lucky" or "unlucky" test sample can noticeably swing the reported metric.

Returns the mean and standard deviation of each metric across the outer
folds, plus out-of-fold predictions (every sample scored by a model that
never saw it during training) usable for a per-family breakdown that
isn't distorted by a single split.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from src.config import RANDOM_SEED
from src.evaluation.metrics import compute_metrics
from src.models.classifiers import train_and_select


def nested_cv_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> dict:
    X, y = df[feature_cols], df["y"]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_metrics = []
    oof_pred = np.full(len(df), -1, dtype=int)
    oof_proba = np.full(len(df), np.nan)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        search = train_and_select(X_train, y_train, model_name)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        fold_metrics.append(compute_metrics(y_test, y_pred, y_proba))
        oof_pred[test_idx] = y_pred
        oof_proba[test_idx] = y_proba

    agg = {}
    for key in ("accuracy", "precision", "recall", "f1"):
        vals = [m[key] for m in fold_metrics]
        agg[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    auc_vals = [m["auc_roc"] for m in fold_metrics if m["auc_roc"] is not None]
    agg["auc_roc"] = (
        {"mean": float(np.mean(auc_vals)), "std": float(np.std(auc_vals))}
        if auc_vals else {"mean": None, "std": None}
    )
    agg["n_folds"] = n_splits
    agg["fold_metrics"] = fold_metrics

    return {"aggregate": agg, "oof_pred": oof_pred, "oof_proba": oof_proba}
