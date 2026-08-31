"""
L1 pipeline: trains and evaluates the classifiers on network-flow + TLS
features only, with no DNS analysis. This is the baseline against which
the DNS layer's incremental value is measured.
"""
from __future__ import annotations

import pandas as pd

from src.config import RANDOM_SEED
from src.evaluation.metrics import compute_metrics
from src.models.classifiers import MODEL_NAMES, train_and_select
from src.pipeline.common import feature_columns
from src.pipeline.split import split_dataset


def run_l1_pipeline(df_l1: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
    df = df_l1.copy()
    df["y"] = (df["label"] == "malicious").astype(int)
    cols = feature_columns(df_l1)

    df_train, df_val, df_test = split_dataset(df, seed=seed)
    X_train, y_train = df_train[cols], df_train["y"]
    X_test, y_test = df_test[cols], df_test["y"]

    results = {
        "feature_columns": cols,
        "models": {},
        "df_train": df_train, "df_val": df_val, "df_test": df_test,
        "y_test": y_test.values,
    }

    for model_name in MODEL_NAMES:
        search = train_and_select(X_train, y_train, model_name)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]
        results["models"][model_name] = {
            "best_params": search.best_params_,
            "cv_score": float(search.best_score_),
            "test_metrics": compute_metrics(y_test, y_pred, y_proba),
            "estimator": best_model,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    best_name = max(results["models"], key=lambda m: results["models"][m]["test_metrics"]["f1"])
    results["best_model_name"] = best_name
    return results
