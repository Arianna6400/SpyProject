"""
L2 pipeline: network-flow analysis + DNS analysis. Implements both the
feature-fusion and score-fusion modes, used to measure the DNS layer's
incremental contribution over the L1-only baseline.
"""
from __future__ import annotations

import pandas as pd

from src.config import RANDOM_SEED
from src.evaluation.metrics import compute_metrics
from src.models.classifiers import MODEL_NAMES, train_and_select
from src.models.fusion import fuse_scores, optimize_fusion_weight
from src.pipeline.common import feature_columns
from src.pipeline.split import split_dataset


def run_l2_feature_fusion(df_full: pd.DataFrame, seed: int = RANDOM_SEED) -> dict:
    """Feature fusion: L1 and DNS features are concatenated into a single
    vector and fed to the classifiers as one combined input."""
    df = df_full.copy()
    df["y"] = (df["label"] == "malicious").astype(int)
    cols = feature_columns(df_full)

    df_train, df_val, df_test = split_dataset(df, seed=seed)
    X_train, y_train = df_train[cols], df_train["y"]
    X_test, y_test = df_test[cols], df_test["y"]

    results = {"feature_columns": cols, "models": {}, "df_test": df_test, "y_test": y_test.values}
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
    results["best_model_name"] = max(results["models"], key=lambda m: results["models"][m]["test_metrics"]["f1"])
    return results


def run_l2_score_fusion(
    df_l1: pd.DataFrame,
    df_dns: pd.DataFrame,
    model_name: str = "random_forest",
    seed: int = RANDOM_SEED,
) -> dict:
    """Score fusion: trains an L1 classifier (flow + TLS) and a DNS-only
    classifier separately, then combines their probabilities via a
    weighted average optimized on the validation set. This isolates and
    directly measures the DNS layer's incremental contribution."""
    df_l1 = df_l1.copy()
    df_l1["y"] = (df_l1["label"] == "malicious").astype(int)

    l1_cols = feature_columns(df_l1)
    dns_cols = feature_columns(df_dns)

    df_train, df_val, df_test = split_dataset(df_l1, seed=seed)
    train_ids = sorted(df_train["session_id"].tolist())
    val_ids = sorted(df_val["session_id"].tolist())
    test_ids = sorted(df_test["session_id"].tolist())

    l1_indexed = df_l1.set_index("session_id")
    dns_indexed = df_dns.set_index("session_id")

    X_train_l1, y_train = l1_indexed.loc[train_ids, l1_cols], l1_indexed.loc[train_ids, "y"]
    X_val_l1, y_val = l1_indexed.loc[val_ids, l1_cols], l1_indexed.loc[val_ids, "y"]
    X_test_l1, y_test = l1_indexed.loc[test_ids, l1_cols], l1_indexed.loc[test_ids, "y"]

    X_train_dns = dns_indexed.loc[train_ids, dns_cols]
    X_val_dns = dns_indexed.loc[val_ids, dns_cols]
    X_test_dns = dns_indexed.loc[test_ids, dns_cols]

    search_l1 = train_and_select(X_train_l1, y_train, model_name)
    search_dns = train_and_select(X_train_dns, y_train, model_name)
    model_l1, model_dns = search_l1.best_estimator_, search_dns.best_estimator_

    proba_l1_val = model_l1.predict_proba(X_val_l1)[:, 1]
    proba_dns_val = model_dns.predict_proba(X_val_dns)[:, 1]
    fusion_choice = optimize_fusion_weight(proba_l1_val, proba_dns_val, y_val.values)
    w_dns = fusion_choice["w_dns"]

    proba_l1_test = model_l1.predict_proba(X_test_l1)[:, 1]
    proba_dns_test = model_dns.predict_proba(X_test_dns)[:, 1]
    proba_fused_test = fuse_scores(proba_l1_test, proba_dns_test, w_dns)
    y_pred_fused = (proba_fused_test >= 0.5).astype(int)

    return {
        "model_name": model_name,
        "w_dns": w_dns,
        "fusion_validation_choice": fusion_choice,
        "metrics_l1_only": compute_metrics(y_test, (proba_l1_test >= 0.5).astype(int), proba_l1_test),
        "metrics_dns_only": compute_metrics(y_test, (proba_dns_test >= 0.5).astype(int), proba_dns_test),
        "metrics_fused": compute_metrics(y_test, y_pred_fused, proba_fused_test),
        "model_l1": model_l1,
        "model_dns": model_dns,
        "l1_feature_columns": l1_cols,
        "dns_feature_columns": dns_cols,
        "df_test": l1_indexed.loc[test_ids].reset_index(),
        "y_test": y_test.values,
        "proba_fused_test": proba_fused_test,
        "y_pred_fused": y_pred_fused,
    }
