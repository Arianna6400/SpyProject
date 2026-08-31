"""
Leave-One-Family-Out (LOFO) evaluation: for each malware family, the
model is trained WITHOUT any sample from that family and evaluated on its
ability to detect it as "never seen before" (training on all other
families + benign traffic, testing on the held-out family + a fixed
portion of benign samples).

Unlike standard k-fold CV (where samples from the same family can appear
in both training and test — see nested_cv_evaluate in
cross_validation.py), this protocol measures the model's real ability to
generalize to unseen malware families, and is the correct test for
checking whether a very high accuracy reflects genuine behavioral
separation or the memorization of patterns specific to families already
seen in training.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_SEED
from src.evaluation.metrics import compute_metrics
from src.models.classifiers import train_and_select


def leave_one_family_out_evaluate(
    df: pd.DataFrame,
    feature_cols: list,
    model_name: str,
    seed: int = RANDOM_SEED,
    benign_test_frac: float = 0.3,
) -> dict:
    df = df.copy()
    if "y" not in df.columns:
        df["y"] = (df["family"] != "benign").astype(int)

    benign_df = df[df["family"] == "benign"]
    malicious_df = df[df["family"] != "benign"]
    families = sorted(malicious_df["family"].unique())

    benign_train, benign_test = train_test_split(benign_df, test_size=benign_test_frac, random_state=seed)

    per_family = []
    for family in families:
        train_mal = malicious_df[malicious_df["family"] != family]
        test_mal = malicious_df[malicious_df["family"] == family]

        train_df = pd.concat([train_mal, benign_train], ignore_index=True)
        test_df = pd.concat([test_mal, benign_test], ignore_index=True)

        X_train, y_train = train_df[feature_cols], train_df["y"]
        X_test, y_test = test_df[feature_cols], test_df["y"]

        search = train_and_select(X_train, y_train, model_name)
        best_model = search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        recall_family = float(y_pred[: len(test_mal)].mean()) if len(test_mal) else float("nan")

        per_family.append({
            "held_out_family": family,
            "n_family_samples": int(len(test_mal)),
            "recall_on_unseen_family": recall_family,
            "overall_metrics": metrics,
        })

    mean_recall = float(np.mean([r["recall_on_unseen_family"] for r in per_family]))
    return {"model_name": model_name, "per_family": per_family, "mean_recall_unseen_family": mean_recall}
