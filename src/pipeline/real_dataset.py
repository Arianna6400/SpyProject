"""
Loader for the real CIC-AndMal2017 dataset (Canadian Institute for
Cybersecurity). Unlike the rest of the pipeline (built around synthetic
sessions from src/simulate), this module works on traffic that was
actually captured and processed with CICFlowMeter.

Source: https://www.kaggle.com/datasets/tinlmnguyn/cicandmal2017
        (a partial mirror of the original UNB/CIC dataset: includes the
        Adware — 10 families, 104 samples — and Benign — 202 samples —
        categories, but not the Ransomware/Scareware/SMS-malware
        categories present in the full release). Fetched 2026-08-31.

Each CSV file corresponds to one sample (a pcap capture processed by
CICFlowMeter, one row per flow). This loader aggregates each file's rows
into a single session-level feature vector (same logic as
src/features/flow_features.aggregate_session_flow_features), so the
generic classification pipeline (src/models/classifiers.py) can be reused
unchanged on this real dataset.

Known limitation: the dataset has no DNS logs, so it only supports an
L1-style evaluation, not the L1/L2 comparison — that would require DNS
traffic captured in parallel with the flow data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT

CIC_ANDMAL2017_DIR = PROJECT_ROOT / "data" / "cic_andmal2017" / "CSVs"

_NON_FEATURE_RAW_COLS = {"Flow ID", "Source IP", "Destination IP", "Timestamp", "Label"}

# Features empirically identified as a likely capture-environment artifact
# rather than genuine malicious behavior: min_packet_length_max separates
# the two classes with ZERO overlap (benign <= 300, malicious >= 315),
# both clustered on a handful of discrete values repeated many times — a
# pattern typical of a systematic configuration difference between the
# benign capture sessions and the sandboxed ones (e.g. MTU, device,
# CICFlowMeter version), not of real malicious network behavior. Verified
# with scripts/run_real_dataset_diagnostics.py: excluding these features
# moderately lowers performance but it stays strong (LOFO recall RF
# 0.982->0.961, XGBoost 1.000->0.982), confirming the rest of the signal
# is genuine rather than a single isolated artifact.
KNOWN_ARTIFACT_FEATURE_PREFIXES = ("min_packet_length",)


def clean_feature_columns(feature_cols: list) -> list:
    """Removes features identified as a likely capture artifact (see
    KNOWN_ARTIFACT_FEATURE_PREFIXES) from a list of feature names."""
    return [c for c in feature_cols if not c.startswith(KNOWN_ARTIFACT_FEATURE_PREFIXES)]


def _family_from_label(label: str) -> str:
    label = label.strip().upper()
    if label == "BENIGN":
        return "benign"
    # e.g. "ADWARE_DOWGIN" -> "dowgin"
    parts = label.split("_", 1)
    return parts[1].lower() if len(parts) > 1 else label.lower()


def _load_one_csv(csv_path: Path, t_max_min: float | None = None) -> dict:
    df = pd.read_csv(csv_path, low_memory=False)
    df.columns = [c.strip() for c in df.columns]

    labels = df["Label"].astype(str).str.strip().str.upper()
    is_benign = (labels == "BENIGN").all()
    label = "benign" if is_benign else "malicious"
    family = "benign" if is_benign else _family_from_label(labels.iloc[0])

    if t_max_min is not None:
        ts = pd.to_datetime(df["Timestamp"], format="%d/%m/%Y %H:%M:%S", errors="coerce")
        valid = ts.notna()
        if valid.any():
            cutoff = ts[valid].min() + pd.Timedelta(minutes=t_max_min)
            df = df[valid & (ts <= cutoff)]

    feature_cols = [c for c in df.columns if c not in _NON_FEATURE_RAW_COLS and c not in ("Source Port", "Destination Port", "Protocol")]
    numeric = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.replace([np.inf, -np.inf], np.nan)

    if numeric.empty:
        return {}

    row = {"sample_id": csv_path.stem, "label": label, "family": family}
    for col in numeric.columns:
        safe_col = col.strip().lower().replace("/", "_per_").replace(" ", "_")
        row[f"{safe_col}_mean"] = float(numeric[col].mean(skipna=True))
        row[f"{safe_col}_std"] = float(numeric[col].std(skipna=True, ddof=0))
        row[f"{safe_col}_max"] = float(numeric[col].max(skipna=True))
    row["n_flows"] = int(len(df))
    return row


def _require_dir(root_dir: Path) -> Path:
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(
            f"CIC-AndMal2017 dataset not found in {root_dir}. "
            "Download it from https://www.kaggle.com/datasets/tinlmnguyn/cicandmal2017 "
            "(kaggle datasets download -d tinlmnguyn/cicandmal2017 -p data/cic_andmal2017 --unzip)."
        )
    return root_dir


def load_cic_andmal2017(root_dir: Path = CIC_ANDMAL2017_DIR) -> pd.DataFrame:
    """Loads every sample from the CIC-AndMal2017 mirror available in
    ``root_dir``, returning a DataFrame with one row per sample
    (sample_id, label, family, aggregated features)."""
    root_dir = _require_dir(root_dir)
    csv_paths = sorted(root_dir.rglob("*.csv"))
    rows = [r for r in (_load_one_csv(p) for p in csv_paths) if r]
    return pd.DataFrame(rows)


def load_cic_andmal2017_windowed(sample_ids: list, t_max_min: float, root_dir: Path = CIC_ANDMAL2017_DIR) -> pd.DataFrame:
    """Reloads only the samples listed in ``sample_ids`` (filenames
    without extension), limiting each one to the first ``t_max_min``
    minutes of capture (based on CICFlowMeter's Timestamp field). Used by
    the detection-latency metric to re-evaluate recall over growing
    observation windows without reloading the whole dataset for every
    window."""
    root_dir = _require_dir(root_dir)
    wanted = set(sample_ids)
    csv_paths = [p for p in root_dir.rglob("*.csv") if p.stem in wanted]
    rows = [r for r in (_load_one_csv(p, t_max_min=t_max_min) for p in csv_paths) if r]
    return pd.DataFrame(rows)
