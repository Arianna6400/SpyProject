"""
Diagnostics for the real CIC-AndMal2017 dataset: checks whether the
near-perfect accuracy obtained with nested CV
(scripts/run_real_dataset_pipeline.py) reflects genuine behavioral
separation or the memorization of patterns specific to malware families
already seen in training.

Runs:
  1. Leave-One-Family-Out CV (src/evaluation/leave_one_family_out.py):
     the correct test for the ability to generalize to unseen malware.
  2. Random Forest and XGBoost feature importance, to identify whether
     the most discriminative features are behavioral or "trivial" (e.g.
     capture-environment artifacts).

Usage:
    python scripts/run_real_dataset_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_DIR, RANDOM_SEED  # noqa: E402
from src.pipeline.real_dataset import load_cic_andmal2017, CIC_ANDMAL2017_DIR  # noqa: E402
from src.pipeline.common import feature_columns  # noqa: E402
from src.models.classifiers import MODEL_NAMES  # noqa: E402
from src.evaluation.leave_one_family_out import leave_one_family_out_evaluate  # noqa: E402
from src.evaluation.feature_importance import compute_feature_importance  # noqa: E402


def main() -> None:
    print(f"Loading CIC-AndMal2017 from {CIC_ANDMAL2017_DIR} ...")
    df = load_cic_andmal2017()
    df = df.rename(columns={"sample_id": "session_id"}).reset_index(drop=True)
    df["y"] = (df["family"] != "benign").astype(int)
    cols = feature_columns(df)
    print(f"  {len(df)} samples, {len(cols)} aggregated features")

    report = {"n_samples": len(df), "lofo": {}, "feature_importance": {}}

    print("\n=== Leave-One-Family-Out CV (generalization to an unseen family) ===")
    for model_name in MODEL_NAMES:
        result = leave_one_family_out_evaluate(df, cols, model_name, seed=RANDOM_SEED)
        report["lofo"][model_name] = result
        print(f"\n{model_name} — mean recall on held-out family: {result['mean_recall_unseen_family']:.3f}")
        for fam_res in result["per_family"]:
            m = fam_res["overall_metrics"]
            print(
                f"  held_out={fam_res['held_out_family']:10s} n={fam_res['n_family_samples']:2d}  "
                f"family_recall={fam_res['recall_on_unseen_family']:.3f}  "
                f"overall_acc={m['accuracy']:.3f}  overall_prec={m['precision']:.3f}"
            )

    print("\n=== Feature importance (model trained on the entire dataset) ===")
    for model_name in ("random_forest", "xgboost"):
        try:
            fi = compute_feature_importance(df, cols, model_name=model_name, top_n=15)
        except ValueError as exc:
            print(f"{model_name}: {exc}")
            continue
        report["feature_importance"][model_name] = fi
        print(f"\nTop 15 features — {model_name}:")
        for item in fi:
            print(f"  {item['feature']:45s} {item['importance']:.4f}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "real_dataset_diagnostics.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
