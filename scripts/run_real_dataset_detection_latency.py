"""
Detection latency on the REAL CIC-AndMal2017 dataset: re-evaluates recall
over growing observation windows (5/10/15/20/30 minutes), using
CICFlowMeter's per-flow Timestamp field to truncate each test capture to
its first N minutes.

The model (Random Forest, the primary classifier) is trained once on a
fixed training set (full-duration features) and then evaluated — without
retraining — on truncated versions of the TEST captures only, to avoid
measuring latency on data already seen in training.

Usage:
    python scripts/run_real_dataset_detection_latency.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from src.config import REPORT_DIR, RANDOM_SEED, DETECTION_WINDOWS_MIN, RECALL_THRESHOLD_DETECTION  # noqa: E402
from src.pipeline.real_dataset import (  # noqa: E402
    load_cic_andmal2017, load_cic_andmal2017_windowed, clean_feature_columns, CIC_ANDMAL2017_DIR,
)
from src.pipeline.common import feature_columns  # noqa: E402
from src.models.classifiers import train_and_select  # noqa: E402

MODEL_NAME = "random_forest"


def main() -> None:
    print(f"Loading CIC-AndMal2017 from {CIC_ANDMAL2017_DIR} ...")
    df = load_cic_andmal2017().rename(columns={"sample_id": "session_id"}).reset_index(drop=True)
    df["y"] = (df["family"] != "benign").astype(int)
    cols = clean_feature_columns(feature_columns(df))
    print(f"  {len(df)} samples, {len(cols)} features (min_packet_length_* family excluded as a suspected artifact)")

    df_train, df_test = train_test_split(df, test_size=0.3, stratify=df["family"], random_state=RANDOM_SEED)
    print(f"  split: {len(df_train)} train / {len(df_test)} test")

    print(f"\nTraining {MODEL_NAME} on the training set (full-duration features)...")
    search = train_and_select(df_train[cols], df_train["y"], MODEL_NAME)
    model = search.best_estimator_

    test_ids = df_test["session_id"].tolist()
    y_true_by_id = dict(zip(df_test["session_id"], df_test["y"]))
    malicious_test_ids = [i for i in test_ids if y_true_by_id[i] == 1]
    print(f"  {len(malicious_test_ids)} malicious samples in the test set to measure recall against, per window")

    recall_by_window = {}
    for t_min in DETECTION_WINDOWS_MIN:
        print(f"  window {t_min} min: reloading and re-truncating the test captures...")
        df_win = load_cic_andmal2017_windowed(test_ids, t_max_min=t_min)
        df_win = df_win.rename(columns={"sample_id": "session_id"})
        df_win = df_win.set_index("session_id").reindex(test_ids)  # align columns/order, NaN if missing
        X_win = df_win.reindex(columns=cols)  # any missing columns -> NaN, handled by the imputer

        y_pred = model.predict(X_win)
        y_true = np.array([y_true_by_id[i] for i in test_ids])
        mal_mask = y_true == 1
        recall = float((y_pred[mal_mask] == 1).mean()) if mal_mask.sum() else float("nan")
        recall_by_window[t_min] = recall
        print(f"    recall = {recall:.3f}")

    t_det = next((t for t in sorted(DETECTION_WINDOWS_MIN) if recall_by_window[t] >= RECALL_THRESHOLD_DETECTION), None)
    print(f"\nT_det({RECALL_THRESHOLD_DETECTION:.0%}) = {t_det} minutes")
    print(f"Recall by window: {recall_by_window}")

    report = {
        "model": MODEL_NAME,
        "n_train": len(df_train), "n_test": len(df_test), "n_malicious_test": len(malicious_test_ids),
        "recall_by_window_min": recall_by_window,
        "t_det_min": t_det,
        "rho_star": RECALL_THRESHOLD_DETECTION,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "real_dataset_detection_latency.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
