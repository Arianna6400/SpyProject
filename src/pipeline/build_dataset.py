"""
Orchestration: generates N synthetic sessions (benign and one per threat
family), extracts L1 (flow + TLS) and L2 (DNS) features, and saves the
resulting datasets to disk.
"""
from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import pandas as pd

from src.config import DATASET_DIR, SESSIONS_DIR, MALICIOUS_FAMILIES, RANDOM_SEED, SESSION_DURATION_S
from src.simulate.session_simulator import simulate_session
from src.features.flow_features import extract_flows_from_pcap, aggregate_session_flow_features
from src.features.tls_features import extract_tls_features
from src.features.dns_features import extract_dns_features


def extract_session_features(meta: dict, t_max_s: Optional[float] = None) -> dict:
    """Combines flow features (L1), TLS features (L1), and DNS features
    (L2) into a single session-level row. If given, ``t_max_s`` limits
    extraction to the first ``t_max_s`` seconds of the session (used for
    the detection-latency metric)."""
    session_start = meta["session_start"]
    df_flows = extract_flows_from_pcap(meta["pcap_path"], t_max_s=t_max_s, session_start=session_start)
    flow_feats = aggregate_session_flow_features(df_flows)
    tls_feats = extract_tls_features(meta["ssl_path"], t_max_s=t_max_s, session_start=session_start)
    dns_feats = extract_dns_features(meta["dns_path"], t_max_s=t_max_s, session_start=session_start)

    row = {
        "session_id": meta["session_id"],
        "label": meta["label"],
        "family": meta["family"] or "benign",
    }
    row.update(flow_feats)
    row.update(tls_feats)
    row.update(dns_feats)
    return row


def generate_sessions(n_benign: int, n_per_family: int, seed: int = RANDOM_SEED) -> list:
    """Generates the synthetic sessions (pcap + DNS/TLS logs + ground
    truth) and returns their metadata. Seeds are derived deterministically
    from the session index for reproducibility."""
    SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
    plan = [("benign", None)] * n_benign
    for fam in MALICIOUS_FAMILIES:
        plan += [("malicious", fam)] * n_per_family

    metas = []
    for i, (label, family) in enumerate(plan):
        session_id = f"{label}_{family or 'na'}_{i:04d}_{uuid.uuid4().hex[:8]}"
        meta = simulate_session(
            session_id=session_id, label=label, family=family,
            out_root=SESSIONS_DIR, duration_s=SESSION_DURATION_S, seed=seed * 100_000 + i,
        )
        metas.append(meta)
    return metas


def build_dataset(
    n_benign: int = 40,
    n_per_family: int = 15,
    seed: int = RANDOM_SEED,
    out_dir: Path = DATASET_DIR,
) -> dict:
    """Generates the full dataset and saves three CSVs:
      - dataset_l1.csv   : flow + TLS features (L1 pipeline)
      - dataset_dns.csv  : DNS features (L2 layer)
      - dataset_full.csv : L1 + DNS concatenated (feature fusion)
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = generate_sessions(n_benign, n_per_family, seed=seed)

    rows = [extract_session_features(m) for m in metas]
    df_full = pd.DataFrame(rows)

    l1_prefixes = (
        "n_flows", "duration", "fwd_", "bwd_", "flow_", "syn_", "ack_", "fin_",
        "rst_", "psh_", "urg_", "active_", "idle_", "total_", "tls_",
    )
    l1_cols = ["session_id", "label", "family"] + [c for c in df_full.columns if c.startswith(l1_prefixes)]
    dns_cols = ["session_id", "label", "family"] + [c for c in df_full.columns if c.startswith("dns_")]

    df_l1 = df_full[l1_cols]
    df_dns = df_full[dns_cols]

    df_l1.to_csv(out_dir / "dataset_l1.csv", index=False)
    df_dns.to_csv(out_dir / "dataset_dns.csv", index=False)
    df_full.to_csv(out_dir / "dataset_full.csv", index=False)

    return {
        "l1_path": out_dir / "dataset_l1.csv",
        "dns_path": out_dir / "dataset_dns.csv",
        "full_path": out_dir / "dataset_full.csv",
        "n_sessions": len(df_full),
        "session_metas": metas,
    }
