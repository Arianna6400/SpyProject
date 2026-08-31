"""
Loads real, publicly published indicators of compromise, used by the
simulator (session_simulator.py) and the IoC-based baseline
(src/baseline/ioc_engine.py) instead of made-up placeholders.

Sources (see data/real_iocs/*.txt|csv for provenance and licensing
details):
  - Amnesty International, "investigations" repo (Pegasus Project, 2021)
    https://github.com/AmnestyTech/investigations
  - ECHAP / Coalition Against Stalkerware, "stalkerware-indicators" repo
    https://github.com/AssoEchap/stalkerware-indicators (CC-BY 4.0)
"""
from __future__ import annotations

import csv
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "real_iocs"
PEGASUS_DOMAINS_FILE = DATA_DIR / "pegasus_amnesty_domains.txt"
STALKERWARE_INDICATORS_FILE = DATA_DIR / "stalkerware_echap_indicators.csv"

# Minimal fallback in case the real files aren't available (e.g. a partial
# repository checkout): keeps the simulator runnable regardless.
_FALLBACK_PEGASUS_DOMAINS = ["cdn-edge-relay.net", "media-sync-cache.com", "static-assets-cdn.org"]
_FALLBACK_STALKERWARE = [("mspyonline.com", "mSpy"), ("hoverwatch.com", "Hoverwatch")]


def _root_domain(fqdn: str) -> str:
    parts = fqdn.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else fqdn


def load_pegasus_domains() -> list:
    """Domains historically documented as Pegasus infrastructure (Amnesty
    International), used as root domains in simulated pegasus_like
    sessions. They don't appear in the known-IoC baseline: see the
    comment in data/real_iocs/pegasus_amnesty_domains.txt for why (Pegasus
    C2 infrastructure rotates constantly, so a static historical list
    can't represent a newly-registered, not-yet-catalogued campaign)."""
    if not PEGASUS_DOMAINS_FILE.exists():
        return list(_FALLBACK_PEGASUS_DOMAINS)
    domains = []
    with open(PEGASUS_DOMAINS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                domains.append(line)
    return domains or list(_FALLBACK_PEGASUS_DOMAINS)


def load_stalkerware_indicators() -> dict:
    """Real indicators (domains/IPs) for mSpy, FlexiSpy, Hoverwatch (ECHAP
    / Coalition Against Stalkerware). Returns a dict with:
      - "domains": list of (fqdn, root_domain) tuples, the simulator's C2
        pool;
      - "ips": list of known IPv4 addresses;
      - "known_roots": set of known root domains, used by the IoC-based
        baseline (src/baseline/ioc_engine.py).
    """
    if not STALKERWARE_INDICATORS_FILE.exists():
        domains = [(d, _root_domain(d)) for d, _ in _FALLBACK_STALKERWARE]
        return {"domains": domains, "ips": [], "known_roots": {r for _, r in domains}}

    domains, ips = [], []
    with open(STALKERWARE_INDICATORS_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for row in reader:
            if row["type"] == "domain":
                domains.append((row["indicator"], _root_domain(row["indicator"])))
            elif row["type"] == "ipv4":
                ips.append(row["indicator"])

    if not domains:
        domains = [(d, _root_domain(d)) for d, _ in _FALLBACK_STALKERWARE]

    return {
        "domains": domains,
        "ips": ips,
        "known_roots": {root for _, root in domains},
    }
