"""
End-to-end orchestration script: generates the synthetic dataset (no
physical capture testbed available), runs the L1 and L2 pipelines
(feature fusion and score fusion), computes detection latency and the
per-family breakdown, and saves a summary report to
results/report/report.json.

Usage:
    python scripts/run_full_pipeline.py --n-benign 40 --n-per-family 15
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

import numpy as np  # noqa: E402

from src.config import REPORT_DIR, RANDOM_SEED  # noqa: E402
from src.pipeline.build_dataset import build_dataset  # noqa: E402
from src.pipeline.l1_pipeline import run_l1_pipeline  # noqa: E402
from src.pipeline.l2_pipeline import run_l2_feature_fusion, run_l2_score_fusion  # noqa: E402
from src.evaluation.detection_latency import compute_detection_latency  # noqa: E402
from src.evaluation.threat_family import per_family_recall  # noqa: E402
from src.evaluation.metrics import compute_metrics  # noqa: E402
from src.baseline.ioc_engine import analyze_sessions  # noqa: E402


def _fmt_metrics(name: str, metrics: dict) -> str:
    auc = metrics["auc_roc"] if metrics["auc_roc"] is not None else float("nan")
    return (
        f"  {name:22s} acc={metrics['accuracy']:.3f}  prec={metrics['precision']:.3f}  "
        f"rec={metrics['recall']:.3f}  f1={metrics['f1']:.3f}  auc={auc:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="End-to-end pipeline (synthetic data) — mobile spyware detection")
    parser.add_argument("--n-benign", type=int, default=40, help="number of benign sessions to generate")
    parser.add_argument("--n-per-family", type=int, default=15, help="malicious sessions per each of the 3 families")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--score-fusion-model", type=str, default="random_forest",
        choices=["random_forest", "xgboost", "logistic_regression"],
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Generating {args.n_benign} benign sessions + {args.n_per_family} for each of the 3 malicious families...")
    dataset = build_dataset(n_benign=args.n_benign, n_per_family=args.n_per_family, seed=args.seed)
    df_l1 = pd.read_csv(dataset["l1_path"])
    df_dns = pd.read_csv(dataset["dns_path"])
    df_full = pd.read_csv(dataset["full_path"])
    print(f"      {dataset['n_sessions']} total sessions generated in results/sessions/")

    print("[2/6] IoC/signature-based baseline (SpyGuard-style)...")
    ioc_verdicts = analyze_sessions(dataset["session_metas"], out_path=REPORT_DIR / "ioc_baseline_report.json")
    y_true_all = np.array([0 if m["label"] == "benign" else 1 for m in dataset["session_metas"]])
    y_pred_ioc_all = np.array([1 if v.is_flagged else 0 for v in ioc_verdicts])
    ioc_metrics_all = compute_metrics(y_true_all, y_pred_ioc_all)
    print(_fmt_metrics("IoC baseline (all)", ioc_metrics_all))

    print("[3/6] L1 pipeline (flow + TLS)...")
    l1_results = run_l1_pipeline(df_l1, seed=args.seed)
    for name, res in l1_results["models"].items():
        print(_fmt_metrics(name, res["test_metrics"]))
    print(f"      Best L1 model: {l1_results['best_model_name']}")

    print("[4/6] L2 pipeline - feature fusion (flow + TLS + DNS)...")
    l2_ff_results = run_l2_feature_fusion(df_full, seed=args.seed)
    for name, res in l2_ff_results["models"].items():
        print(_fmt_metrics(name, res["test_metrics"]))
    print(f"      Best L2 model (feature fusion): {l2_ff_results['best_model_name']}")

    print(f"[5/6] L2 pipeline - score fusion (base model: {args.score_fusion_model})...")
    l2_sf_results = run_l2_score_fusion(df_l1, df_dns, model_name=args.score_fusion_model, seed=args.seed)
    print(f"      Optimal w_dns on validation set (precision >= 0.80): {l2_sf_results['w_dns']}")
    print(_fmt_metrics("L1-only", l2_sf_results["metrics_l1_only"]))
    print(_fmt_metrics("DNS-only", l2_sf_results["metrics_dns_only"]))
    print(_fmt_metrics("Fused (L1+DNS)", l2_sf_results["metrics_fused"]))

    print("[6/6] Detection latency, IoC baseline on the test set, and per-family breakdown...")
    y_true_by_id = {m["session_id"]: 0 if m["label"] == "benign" else 1 for m in dataset["session_metas"]}
    test_ids = set(l1_results["df_test"]["session_id"])
    malicious_test_metas = [
        m for m in dataset["session_metas"] if m["session_id"] in test_ids and m["label"] == "malicious"
    ]

    # IoC baseline restricted to the same test set used by L1/L2, for a fair comparison
    id_to_verdict = {v.session_id: v for v in ioc_verdicts}
    test_metas_ordered = [m for m in dataset["session_metas"] if m["session_id"] in test_ids]
    y_true_test = np.array([y_true_by_id[m["session_id"]] for m in test_metas_ordered])
    y_pred_ioc_test = np.array([1 if id_to_verdict[m["session_id"]].is_flagged else 0 for m in test_metas_ordered])
    ioc_metrics_test = compute_metrics(y_true_test, y_pred_ioc_test)
    print(_fmt_metrics("IoC baseline (test set)", ioc_metrics_test))
    ioc_test_df = pd.DataFrame({
        "session_id": [m["session_id"] for m in test_metas_ordered],
        "label": [m["label"] for m in test_metas_ordered],
        "family": [m["family"] or "benign" for m in test_metas_ordered],
    })
    fam_ioc = per_family_recall(ioc_test_df, y_pred_ioc_test, y_pred_ioc_test.astype(float))

    best_l1 = l1_results["models"][l1_results["best_model_name"]]
    dl_l1 = compute_detection_latency(malicious_test_metas, y_true_by_id, l1_results["feature_columns"], best_l1["estimator"])

    best_l2 = l2_ff_results["models"][l2_ff_results["best_model_name"]]
    dl_l2 = compute_detection_latency(malicious_test_metas, y_true_by_id, l2_ff_results["feature_columns"], best_l2["estimator"])

    print(f"      Detection latency L1: T_det(80%) = {dl_l1['t_det_min']} min | recall by window: {dl_l1['recall_by_window_min']}")
    print(f"      Detection latency L2: T_det(80%) = {dl_l2['t_det_min']} min | recall by window: {dl_l2['recall_by_window_min']}")

    fam_l1 = per_family_recall(l1_results["df_test"], best_l1["y_pred"], best_l1["y_proba"])
    fam_l2 = per_family_recall(l2_ff_results["df_test"], best_l2["y_pred"], best_l2["y_proba"])

    print("\n      Recall/FPR by family (IoC/signature-based baseline):")
    print(fam_ioc.to_string(index=False))
    print("\n      Recall/FPR by family (L1):")
    print(fam_l1.to_string(index=False))
    print("\n      Recall/FPR by family (L2 - feature fusion):")
    print(fam_l2.to_string(index=False))

    report = {
        "n_sessions": dataset["n_sessions"],
        "n_benign": args.n_benign,
        "n_per_family": args.n_per_family,
        "ioc_baseline_all_sessions": ioc_metrics_all,
        "ioc_baseline_test_set": ioc_metrics_test,
        "per_family_recall_ioc_baseline": fam_ioc.to_dict(orient="records"),
        "l1": {name: res["test_metrics"] for name, res in l1_results["models"].items()},
        "l1_best_model": l1_results["best_model_name"],
        "l2_feature_fusion": {name: res["test_metrics"] for name, res in l2_ff_results["models"].items()},
        "l2_feature_fusion_best_model": l2_ff_results["best_model_name"],
        "l2_score_fusion": {
            "model_name": l2_sf_results["model_name"],
            "w_dns": l2_sf_results["w_dns"],
            "metrics_l1_only": l2_sf_results["metrics_l1_only"],
            "metrics_dns_only": l2_sf_results["metrics_dns_only"],
            "metrics_fused": l2_sf_results["metrics_fused"],
        },
        "detection_latency_l1": dl_l1,
        "detection_latency_l2": dl_l2,
        "per_family_recall_l1": fam_l1.to_dict(orient="records"),
        "per_family_recall_l2": fam_l2.to_dict(orient="records"),
    }
    report_path = REPORT_DIR / "report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {report_path}")


if __name__ == "__main__":
    main()
