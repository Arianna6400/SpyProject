"""
IoC-based and signature-based baseline, analogous to the analysis engine
used by tools like SpyGuard: matching against known indicators of
compromise (IoC), Suricata-style signature rules, and anomaly detection
on beaconing patterns. Used as a non-ML baseline to compare the ML
classifiers against — "we beat the baseline" only means something if the
baseline is a real, independent detector.

Unlike the ML classifiers, this engine doesn't learn from a training set:
it applies known IoCs and fixed heuristic rules, exactly as a production
tool would. By construction, it's expected to do well against threats
with known, stable infrastructure (commercial stalkerware catalogued by
ECHAP/Coalition Against Stalkerware) and to struggle against unknown or
rotating C2 infrastructure (Pegasus, whose infrastructure rotates
constantly, or Graphite, which avoids DNS entirely) — that's exactly the
documented limitation that motivates using ML classifiers trained on
behavioral patterns instead of static indicators.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.features.dns_features import load_dns_records
from src.features.tls_features import load_ssl_records
from src.simulate.real_iocs import load_stalkerware_indicators

# --------------------------------------------------------------------------- #
# IoC knowledge base: indicators REALLY published by ECHAP / Coalition
# Against Stalkerware for commercial stalkerware (mSpy, FlexiSpy,
# Hoverwatch — see data/real_iocs/). Nation-state C2 infrastructure (the
# "pegasus_like" pattern) is deliberately absent here: real Pegasus
# infrastructure rotates constantly and is never fully known in advance.
# The same goes for "graphite_like", which by definition produces no DNS
# query to match against an IoC list.
# --------------------------------------------------------------------------- #
_stalkerware_iocs = load_stalkerware_indicators()
KNOWN_MALICIOUS_ROOT_DOMAINS = _stalkerware_iocs["known_roots"]
KNOWN_MALICIOUS_IPS = set(_stalkerware_iocs["ips"])

RULE_NEW_DOMAIN_AGE_DAYS = 30
RULE_ENTROPY_THRESHOLD = 3.5
RULE_TXT_NULL_RATIO_THRESHOLD = 0.10
RULE_BEACON_SCORE_THRESHOLD = 0.85
RULE_MIN_QUERIES_FOR_BEACON = 3


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    counts = Counter(s)
    n = len(s)
    return -sum((c / n) * math.log2(c / n) for c in counts.values())


@dataclass
class IocVerdict:
    session_id: str
    is_flagged: bool
    matched_iocs: list = field(default_factory=list)
    matched_rules: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "is_flagged": self.is_flagged,
            "matched_iocs": self.matched_iocs,
            "matched_rules": self.matched_rules,
        }


def _check_ioc_matching(dns_records: list, ssl_records: list) -> list:
    """Direct matching against known IoCs (domains/IPs)."""
    matches = set()
    for r in dns_records:
        if r.get("root_domain") in KNOWN_MALICIOUS_ROOT_DOMAINS:
            matches.add(f"ioc:domain:{r['root_domain']}")
        if r.get("answer") in KNOWN_MALICIOUS_IPS:
            matches.add(f"ioc:ip:{r['answer']}")
    for r in ssl_records:
        if r.get("id.resp_h") in KNOWN_MALICIOUS_IPS:
            matches.add(f"ioc:ip:{r['id.resp_h']}")
    return sorted(matches)


def _check_signature_rules(dns_records: list, ssl_records: list) -> list:
    """Suricata-style heuristic rules: recently registered domain,
    self-signed/free-CA certificate, high domain-name entropy, unusual DNS
    record types (possible tunneling)."""
    fired = set()

    if any((r.get("domain_age_days") is not None and r["domain_age_days"] < RULE_NEW_DOMAIN_AGE_DAYS) for r in dns_records):
        fired.add("rule:newly_registered_domain")

    if any(r.get("cert_ca_type") in ("self_signed", "free_ca") for r in ssl_records):
        fired.add("rule:low_trust_certificate")

    # Entropy must be computed on the most specific label (the
    # subdomain), not on the whole FQDN: well-known second-level domains
    # (e.g. "googleapis.com") are low-entropy by definition, but a long
    # benign FQDN can still approach a threshold tuned on the whole name,
    # producing systematic false positives. Verified empirically: benign
    # labels stay under ~3.3 bits, algorithmically generated ones
    # (Pegasus-style pattern) exceed ~3.7 bits.
    if any(_entropy(r["qname"].split(".")[0]) > RULE_ENTROPY_THRESHOLD for r in dns_records):
        fired.add("rule:high_entropy_domain")

    if dns_records:
        txt_null_ratio = sum(1 for r in dns_records if r.get("qtype") in ("TXT", "NULL")) / len(dns_records)
        if txt_null_ratio > RULE_TXT_NULL_RATIO_THRESHOLD:
            fired.add("rule:dns_tunneling_pattern")

    return sorted(fired)


def _check_beaconing_anomaly(dns_records: list) -> list:
    """Behavioral anomaly detection: very regular beaconing toward the
    same root domain, independently of whether any known IoC matched
    (i.e. network activity with no corresponding user interaction)."""
    by_root = defaultdict(list)
    for r in dns_records:
        by_root[r["root_domain"]].append(r["ts"])

    for timestamps in by_root.values():
        if len(timestamps) < RULE_MIN_QUERIES_FOR_BEACON:
            continue
        iats = np.diff(sorted(timestamps))
        if len(iats) == 0:
            continue
        beacon_score = float(np.mean(iats < 60.0))
        if beacon_score > RULE_BEACON_SCORE_THRESHOLD:
            return ["rule:periodic_beaconing"]
    return []


def analyze_session(session_id: str, dns_path, ssl_path) -> IocVerdict:
    dns_records = load_dns_records(dns_path)
    ssl_records = load_ssl_records(ssl_path)

    ioc_matches = _check_ioc_matching(dns_records, ssl_records)
    rule_matches = sorted(set(_check_signature_rules(dns_records, ssl_records) + _check_beaconing_anomaly(dns_records)))

    is_flagged = bool(ioc_matches) or bool(rule_matches)
    return IocVerdict(session_id=session_id, is_flagged=is_flagged, matched_iocs=ioc_matches, matched_rules=rule_matches)


def analyze_sessions(session_metas: list, out_path: Optional[Path] = None) -> list:
    """Runs the IoC/signature-based engine on each session and, optionally,
    saves a JSON report (analogous to a SpyGuard-style analysis report)."""
    verdicts = [analyze_session(m["session_id"], m["dns_path"], m["ssl_path"]) for m in session_metas]
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump([v.to_dict() for v in verdicts], f, indent=2)
    return verdicts
