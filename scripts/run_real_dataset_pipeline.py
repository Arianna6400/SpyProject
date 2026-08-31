"""
Evaluates the classifiers on the REAL CIC-AndMal2017 dataset (Adware +
Benign) via nested k-fold cross-validation (5 outer folds, inner
GridSearchCV per fold), complementing the main synthetic-data pipeline
(scripts/run_full_pipeline.py).

See src/pipeline/real_dataset.py for the dataset's provenance and this
mirror's limitations (Adware only, no DNS logs -> L1-style evaluation
only), and src/evaluation/cross_validation.py for why nested CV is used
instead of a single fixed split (fragile on a dataset this size,
especially for families with few samples).

Usage:
    python scripts/run_real_dataset_pipeline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import REPORT_DIR, RANDOM_SEED  # noqa: E402
from src.pipeline.real_dataset import (  # noqa: E402
    load_cic_andmal2017, CIC_ANDMAL2017_DIR, clean_feature_columns, KNOWN_ARTIFACT_FEATURE_PREFIXES,
)
from src.pipeline.common import feature_columns  # noqa: E402
from src.models.classifiers import MODEL_NAMES  # noqa: E402
from src.evaluation.cross_validation import nested_cv_evaluate  # noqa: E402
from src.evaluation.threat_family import per_family_recall  # noqa: E402

N_FOLDS = 5


def _fmt_agg(name: str, agg: dict) -> str:
    def mstd(k):
        d = agg[k]
        return f"{d['mean']:.3f}±{d['std']:.3f}" if d["mean"] is not None else "n/a"
    return (
        f"  {name:22s} acc={mstd('accuracy')}  prec={mstd('precision')}  "
        f"rec={mstd('recall')}  f1={mstd('f1')}  auc={mstd('auc_roc')}"
    )


def main() -> None:
    print(f"Loading CIC-AndMal2017 from {CIC_ANDMAL2017_DIR} ...")
    df = load_cic_andmal2017()
    n_benign = (df["label"] == "benign").sum()
    n_malicious = (df["label"] == "malicious").sum()
    families = sorted(df.loc[df["label"] == "malicious", "family"].unique())
    print(f"  {len(df)} samples loaded ({n_benign} benign, {n_malicious} malicious, families: {families})")

    df = df.rename(columns={"sample_id": "session_id"}).reset_index(drop=True)
    df["y"] = (df["label"] == "malicious").astype(int)
    cols_all = feature_columns(df)
    cols = clean_feature_columns(cols_all)
    n_removed = len(cols_all) - len(cols)
    print(f"  {n_removed} features excluded as suspected capture artifacts (prefixes: {KNOWN_ARTIFACT_FEATURE_PREFIXES}) -> {len(cols)} features used")

    print(f"\nNested {N_FOLDS}-fold cross-validation (hyperparameter search confined to each fold's training data):")
    report = {
        "n_samples": len(df), "n_folds": N_FOLDS,
        "n_features_used": len(cols), "n_features_excluded_as_artifact": n_removed,
        "excluded_feature_prefixes": list(KNOWN_ARTIFACT_FEATURE_PREFIXES),
        "models": {},
    }
    best_name, best_f1 = None, -1.0
    best_oof_pred, best_oof_proba = None, None

    for model_name in MODEL_NAMES:
        result = nested_cv_evaluate(df, cols, model_name, n_splits=N_FOLDS, seed=RANDOM_SEED)
        agg = result["aggregate"]
        print(_fmt_agg(model_name, agg))
        report["models"][model_name] = agg
        if agg["f1"]["mean"] > best_f1:
            best_f1, best_name = agg["f1"]["mean"], model_name
            best_oof_pred, best_oof_proba = result["oof_pred"], result["oof_proba"]

    print(f"\nBest model (mean F1 over {N_FOLDS} folds): {best_name} (F1={best_f1:.3f})")

    fam = per_family_recall(df, best_oof_pred, best_oof_proba)
    print("\nRecall/FPR by Adware family (out-of-fold predictions, every sample scored by a model that never trained on it):")
    print(fam.to_string(index=False))

    report["best_model"] = best_name
    report["per_family_recall_oof"] = fam.to_dict(orient="records")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORT_DIR / "real_dataset_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nReport saved to {out_path}")


if __name__ == "__main__":
    main()
