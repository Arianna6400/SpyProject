"""
DNS feature extraction (L2 layer): lexical, behavioral, and
infrastructural features, aggregated at the session level.

The input file is a JSONL log with one line per observed DNS query,
equivalent to a log produced by Unbound (log-queries/log-replies) merged
and enriched offline with domain age (WHOIS) and reputation-list
membership (e.g. Tranco). In a real deployment those last two pieces of
information would come from live WHOIS/Tranco lookups; here they're
provided directly in the log by the simulator (src/simulate) as known
ground truth, so this extraction function is identical to the one that
would run on real data enriched the same way.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from typing import Optional

import numpy as np

_HEX_B64_RE = re.compile(r"^[a-fA-F0-9]{12,}$|^[A-Za-z0-9+/]{16,}={0,2}$")

EMPTY_DNS_FEATURES = {
    "dns_n_queries": 0, "dns_query_rate": 0.0,
    "dns_fqdn_len_mean": 0.0, "dns_fqdn_len_max": 0.0,
    "dns_entropy_mean": 0.0, "dns_entropy_max": 0.0,
    "dns_digit_ratio_mean": 0.0, "dns_label_depth_mean": 0.0,
    "dns_hex_pattern_ratio": 0.0,
    "dns_iat_mean": 0.0, "dns_iat_std": 0.0, "dns_iat_min": 0.0,
    "dns_beacon_score": 0.0,
    "dns_txt_null_ratio": 0.0,
    "dns_n_unique_roots": 0,
    "dns_response_bytes_mean": 0.0, "dns_response_bytes_max": 0.0,
    "dns_ttl_mean": 0.0,
    "dns_domain_age_mean": np.nan, "dns_domain_age_min": np.nan,
    "dns_tranco_ratio": np.nan,
}


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


def _digit_ratio(s: str) -> float:
    return sum(ch.isdigit() for ch in s) / len(s) if s else 0.0


def _label_depth(fqdn: str) -> int:
    return fqdn.count(".") + 1


def _looks_encoded(label: str) -> bool:
    return bool(_HEX_B64_RE.match(label))


def load_dns_records(dns_jsonl_path) -> list:
    records = []
    with open(dns_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def extract_dns_features(
    dns_jsonl_path,
    t_max_s: Optional[float] = None,
    session_start: Optional[float] = None,
) -> dict:
    """Computes lexical, behavioral, and infrastructural DNS features
    aggregated over the whole session. If ``t_max_s`` is given, only
    queries within the first ``t_max_s`` seconds of the session are
    considered (used by the detection-latency metric)."""
    records = load_dns_records(dns_jsonl_path)
    if session_start is not None and t_max_s is not None:
        records = [r for r in records if r["ts"] <= session_start + t_max_s]

    n = len(records)
    if n == 0:
        return dict(EMPTY_DNS_FEATURES)

    fqdns = [r["qname"] for r in records]
    labels = [f.split(".")[0] for f in fqdns]
    entropies = [_shannon_entropy(f) for f in fqdns]
    lengths = [len(f) for f in fqdns]
    digit_ratios = [_digit_ratio(f) for f in fqdns]
    depths = [_label_depth(f) for f in fqdns]
    hex_flags = [_looks_encoded(lbl) for lbl in labels]

    timestamps = sorted(r["ts"] for r in records)
    duration_min = max((timestamps[-1] - timestamps[0]) / 60.0, 1e-6)
    query_rate = n / duration_min

    by_root = defaultdict(list)
    for r in records:
        by_root[r["root_domain"]].append(r["ts"])
    all_iats = []
    for ts_list in by_root.values():
        all_iats.extend(np.diff(sorted(ts_list)))

    if all_iats:
        iat_mean, iat_std, iat_min = float(np.mean(all_iats)), float(np.std(all_iats)), float(np.min(all_iats))
        beacon_score = float(np.mean([1.0 if x < 60.0 else 0.0 for x in all_iats]))
    else:
        iat_mean = iat_std = iat_min = 0.0
        beacon_score = 0.0

    qtypes = [r.get("qtype", "A") for r in records]
    txt_null_ratio = sum(1 for q in qtypes if q in ("TXT", "NULL")) / n

    response_bytes = [r.get("response_bytes", 0) for r in records]
    ttls = [r.get("ttl", 0) for r in records]
    ages = [r["domain_age_days"] for r in records if r.get("domain_age_days") is not None]
    tranco_flags = [r["in_tranco"] for r in records if r.get("in_tranco") is not None]

    return {
        "dns_n_queries": n,
        "dns_query_rate": query_rate,
        "dns_fqdn_len_mean": float(np.mean(lengths)), "dns_fqdn_len_max": float(np.max(lengths)),
        "dns_entropy_mean": float(np.mean(entropies)), "dns_entropy_max": float(np.max(entropies)),
        "dns_digit_ratio_mean": float(np.mean(digit_ratios)),
        "dns_label_depth_mean": float(np.mean(depths)),
        "dns_hex_pattern_ratio": float(np.mean(hex_flags)),
        "dns_iat_mean": iat_mean, "dns_iat_std": iat_std, "dns_iat_min": iat_min,
        "dns_beacon_score": beacon_score,
        "dns_txt_null_ratio": txt_null_ratio,
        "dns_n_unique_roots": len(by_root),
        "dns_response_bytes_mean": float(np.mean(response_bytes)),
        "dns_response_bytes_max": float(np.max(response_bytes)),
        "dns_ttl_mean": float(np.mean(ttls)),
        "dns_domain_age_mean": float(np.mean(ages)) if ages else np.nan,
        "dns_domain_age_min": float(np.min(ages)) if ages else np.nan,
        "dns_tranco_ratio": float(np.mean(tranco_flags)) if tranco_flags else np.nan,
    }
