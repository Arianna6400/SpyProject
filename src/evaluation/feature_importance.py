"""
Feature importance for tree-based classifiers (Random Forest, XGBoost),
computed by training a model on the entire available dataset (not on a
single CV fold: the goal here is interpretive — understanding which
features drive the separation — not performance estimation, which is
already covered by nested_cv_evaluate and leave_one_family_out_evaluate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import RANDOM_SEED
from src.models.classifiers import train_and_select


def compute_feature_importance(
    df: pd.DataFrame,
    feature_cols: list,
    model_name: str = "random_forest",
    top_n: int = 20,
    seed: int = RANDOM_SEED,
) -> list:
    df = df.copy()
    if "y" not in df.columns:
        df["y"] = (df["family"] != "benign").astype(int)

    X, y = df[feature_cols], df["y"]
    search = train_and_select(X, y, model_name, cv=5)
    clf = search.best_estimator_.named_steps["clf"]

    if not hasattr(clf, "feature_importances_"):
        raise ValueError(f"Model '{model_name}' doesn't expose feature_importances_")

    importances = clf.feature_importances_
    order = np.argsort(importances)[::-1][:top_n]
    return [{"feature": feature_cols[i], "importance": float(importances[i])} for i in order]
