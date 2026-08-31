"""
TLS/certificate feature extraction from a ssl.log-style log (Zeek
equivalent), used as part of the L1 layer: the use of self-signed
certificates or ones issued by free CAs is an infrastructural indicator
documented in public Pegasus analyses.
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np

EMPTY_TLS_FEATURES = {
    "tls_n_connections": 0, "tls_sni_present_ratio": 0.0,
    "tls_self_signed_ratio": 0.0, "tls_free_ca_ratio": 0.0,
    "tls_commercial_ca_ratio": 0.0, "tls_v13_ratio": 0.0,
}


def load_ssl_records(ssl_jsonl_path) -> list:
    records = []
    with open(ssl_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_tls_features(
    ssl_jsonl_path,
    t_max_s: Optional[float] = None,
    session_start: Optional[float] = None,
) -> dict:
    records = load_ssl_records(ssl_jsonl_path)
    if session_start is not None and t_max_s is not None:
        records = [r for r in records if r["ts"] <= session_start + t_max_s]

    n = len(records)
    if n == 0:
        return dict(EMPTY_TLS_FEATURES)

    sni_present = [1.0 if r.get("sni") else 0.0 for r in records]
    ca_types = [r.get("cert_ca_type") for r in records]
    versions = [r.get("version") for r in records]

    return {
        "tls_n_connections": n,
        "tls_sni_present_ratio": float(np.mean(sni_present)),
        "tls_self_signed_ratio": ca_types.count("self_signed") / n,
        "tls_free_ca_ratio": ca_types.count("free_ca") / n,
        "tls_commercial_ca_ratio": ca_types.count("commercial") / n,
        "tls_v13_ratio": versions.count("TLS1.3") / n,
    }
