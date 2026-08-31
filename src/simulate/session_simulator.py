"""
Synthetic traffic-session generator (30 minutes) for a device producing
either benign traffic or traffic infected with one of the simulated
spyware families.

With no physical capture testbed (Raspberry Pi + SpyGuard) or real spyware
samples available, this module generates STRUCTURALLY realistic data that
the rest of the pipeline (src/features/*) consumes exactly as it would
consume the real capture node's output:

  - capture.pcap  -> real TCP packets for the session's connections,
                      equivalent to a pcap captured by tcpdump/SpyGuard.
  - dns.jsonl     -> DNS query log, equivalent to an Unbound log enriched
                      offline with domain age and Tranco-list membership.
  - ssl.jsonl     -> TLS certificate metadata, equivalent to Zeek's ssl.log.
  - meta.json     -> ground truth (label, family) used for evaluation.

To switch to real data, replace this module with a real capture pipeline
while keeping the same output format: the rest of the pipeline (feature
extraction, classification, evaluation) requires no changes at all.
"""
from __future__ import annotations

import json
import random
import string
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from scapy.all import IP, TCP, Raw, wrpcap

from src.config import SESSION_DURATION_S
from src.simulate.real_iocs import load_pegasus_domains, load_stalkerware_indicators

# --------------------------------------------------------------------------- #
# Reference domains and infrastructure
# --------------------------------------------------------------------------- #

BENIGN_DOMAINS = [
    ("www.google.com", "google.com"),
    ("mail.google.com", "google.com"),
    ("www.wikipedia.org", "wikipedia.org"),
    ("www.youtube.com", "youtube.com"),
    ("web.whatsapp.com", "whatsapp.net"),
    ("api.telegram.org", "telegram.org"),
    ("play.googleapis.com", "googleapis.com"),
    ("graph.facebook.com", "facebook.com"),
    ("outlook.office365.com", "office365.com"),
    ("fonts.gstatic.com", "gstatic.com"),
    ("clientservices.googleapis.com", "googleapis.com"),
    ("firebaseinstallations.googleapis.com", "googleapis.com"),
]

# Pool of Pegasus-style "infection domains": root domains under which
# dynamic, high-entropy subdomains are generated. Loaded from
# data/real_iocs/pegasus_amnesty_domains.txt — domains historically
# documented by Amnesty International as Pegasus infrastructure (see
# src/simulate/real_iocs.py for source and licensing notes).
PEGASUS_C2_ROOTS = load_pegasus_domains()

# Static, low-effort commercial C2 domains (stalkerware pattern). Loaded
# from data/real_iocs/stalkerware_echap_indicators.csv — real indicators
# for mSpy/FlexiSpy/Hoverwatch published by ECHAP / Coalition Against
# Stalkerware (the same source used by the IoC baseline in
# src/baseline/ioc_engine.py).
STALKERWARE_C2_DOMAINS = load_stalkerware_indicators()["domains"]

PUBLIC_IP_POOL_BENIGN = [f"142.250.{a}.{b}" for a in (60, 70, 80) for b in (10, 20, 30, 40)]
PUBLIC_IP_POOL_MALICIOUS = [f"185.{a}.{b}.{c}" for a in (12, 45, 91) for b in (34, 88) for c in (5, 17, 200)]

_ALPHABET = string.ascii_lowercase + string.digits


def _rand_label(rng: random.Random, length: Optional[int] = None) -> str:
    length = length or rng.randint(16, 28)
    return "".join(rng.choices(_ALPHABET, k=length))


def _client_ip() -> str:
    return "10.42.0.42"  # the device under analysis on the isolated network


# --------------------------------------------------------------------------- #
# Intermediate data structures
# --------------------------------------------------------------------------- #

@dataclass
class FlowSpec:
    dst_ip: str
    dst_port: int
    src_port: int
    start_s: float
    n_packets: int
    iat_pattern: str  # "human" | "periodic"
    beacon_interval_s: float
    fwd_len_range: tuple
    bwd_len_range: tuple
    is_tls: bool
    sni: Optional[str]
    cert_ca_type: Optional[str]  # "commercial" | "free_ca" | "self_signed"
    tls_version: Optional[str]
    burst_upload: bool = False  # simulates an anomalous upload/exfiltration burst


@dataclass
class DnsSpec:
    t_s: float
    qname: str
    root_domain: str
    qtype: str  # "A" | "TXT"
    ttl: int
    answer_ip: str
    response_bytes: int
    domain_age_days: int
    in_tranco: bool


@dataclass
class SessionPlan:
    flows: list = field(default_factory=list)
    dns: list = field(default_factory=list)


LEGITIMATE_TELEMETRY_DOMAINS = [
    ("clientservices.googleapis.com", "googleapis.com"),
    ("firebaseinstallations.googleapis.com", "googleapis.com"),
]


def _add_legitimate_beaconing(rng: random.Random, plan: SessionPlan, duration_s: int) -> None:
    """Periodic but legitimate telemetry/keepalive traffic (push
    notifications, background sync): many real apps produce regular
    beaconing without being malicious. Adding it to benign traffic too
    prevents raw flow periodicity alone from becoming a trivial
    discriminator, leaving the DNS layer (entropy, domain age, reputation)
    as the feature set that actually carries incremental signal."""
    if rng.random() >= 0.7:
        return
    domain, root = rng.choice(LEGITIMATE_TELEMETRY_DOMAINS)
    dst_ip = rng.choice(PUBLIC_IP_POOL_BENIGN)
    beacon_interval = rng.uniform(60, 200)
    t = rng.uniform(5, 30)
    while t < duration_s - 10:
        plan.flows.append(FlowSpec(
            dst_ip=dst_ip, dst_port=443, src_port=rng.randint(40000, 60000),
            start_s=t, n_packets=rng.randint(6, 12), iat_pattern="periodic",
            beacon_interval_s=beacon_interval, fwd_len_range=(60, 150), bwd_len_range=(60, 200),
            is_tls=True, sni=domain, cert_ca_type="commercial", tls_version="TLS1.3",
        ))
        plan.dns.append(DnsSpec(
            t_s=max(0.0, t - rng.uniform(0.1, 0.5)), qname=domain, root_domain=root,
            qtype="A", ttl=rng.randint(120, 3600), answer_ip=dst_ip,
            response_bytes=rng.randint(48, 96),
            domain_age_days=rng.randint(1500, 9000), in_tranco=True,
        ))
        t += beacon_interval * rng.uniform(0.7, 1.3)


def _add_benign_background(rng: random.Random, plan: SessionPlan, duration_s: int, n_flows: int) -> None:
    _add_legitimate_beaconing(rng, plan, duration_s)
    for _ in range(n_flows):
        domain, root = rng.choice(BENIGN_DOMAINS)
        start = rng.uniform(0, duration_s - 30)
        dst_ip = rng.choice(PUBLIC_IP_POOL_BENIGN)
        plan.flows.append(FlowSpec(
            dst_ip=dst_ip, dst_port=443, src_port=rng.randint(40000, 60000),
            start_s=start, n_packets=rng.randint(10, 40), iat_pattern="human",
            beacon_interval_s=0.0, fwd_len_range=(60, 400), bwd_len_range=(200, 1400),
            is_tls=True, sni=domain, cert_ca_type="commercial", tls_version="TLS1.3",
        ))
        plan.dns.append(DnsSpec(
            t_s=max(0.0, start - rng.uniform(0.2, 1.5)), qname=domain, root_domain=root,
            qtype="A", ttl=rng.randint(120, 3600), answer_ip=dst_ip,
            response_bytes=rng.randint(48, 96),
            domain_age_days=rng.randint(1500, 9000), in_tranco=True,
        ))


# --------------------------------------------------------------------------- #
# Per-family session plans
# --------------------------------------------------------------------------- #

def _plan_benign(rng: random.Random, duration_s: int) -> SessionPlan:
    plan = SessionPlan()
    _add_benign_background(rng, plan, duration_s, rng.randint(18, 35))
    return plan


def _plan_pegasus(rng: random.Random, duration_s: int) -> SessionPlan:
    """Dynamic DNS infrastructure, high-entropy subdomains, self-signed /
    free-CA certificates: pre-2021 Pegasus pattern."""
    plan = SessionPlan()
    c2_root = rng.choice(PEGASUS_C2_ROOTS)
    c2_ip = rng.choice(PUBLIC_IP_POOL_MALICIOUS)
    beacon_interval = rng.uniform(45, 90)
    age_days = rng.randint(2, 25)
    t = rng.uniform(5, 30)
    while t < duration_s - 10:
        sub = _rand_label(rng, rng.randint(18, 30))
        deep = rng.random() < 0.3
        qname = f"{sub}.{_rand_label(rng, 8)}.{c2_root}" if deep else f"{sub}.{c2_root}"
        burst = rng.random() < 0.15
        plan.dns.append(DnsSpec(
            t_s=t, qname=qname, root_domain=c2_root, qtype="TXT" if burst else "A",
            ttl=rng.randint(30, 120), answer_ip=c2_ip,
            response_bytes=rng.randint(800, 2200) if burst else rng.randint(48, 110),
            domain_age_days=age_days, in_tranco=False,
        ))
        # The flow-level signal is deliberately kept weak and overlapping
        # with benign traffic (similar packet sizes, looser beaconing, a
        # barely-perceptible exfiltration burst in flow bytes): Pegasus's
        # dynamic C2 infrastructure is designed to be as indistinguishable
        # as possible at the application-traffic level, so it's the DNS
        # layer (entropy, domain age, TXT response size) that's meant to
        # provide the robust discriminating signal.
        plan.flows.append(FlowSpec(
            dst_ip=c2_ip, dst_port=443, src_port=rng.randint(40000, 60000),
            start_s=t + rng.uniform(0.1, 0.6),
            n_packets=rng.randint(12, 22) if burst else rng.randint(6, 16),
            iat_pattern="periodic", beacon_interval_s=beacon_interval,
            fwd_len_range=(60, 350), bwd_len_range=(150, 900),
            is_tls=True, sni=qname, cert_ca_type=rng.choice(["self_signed", "free_ca"]),
            tls_version=rng.choice(["TLS1.2", "TLS1.3"]), burst_upload=burst,
        ))
        t += beacon_interval * rng.uniform(0.65, 1.35)
    _add_benign_background(rng, plan, duration_s, rng.randint(3, 8))
    return plan


def _plan_graphite(rng: random.Random, duration_s: int) -> SessionPlan:
    """Direct-to-IP C2 communications, no associated DNS query: Paragon/
    Graphite pattern, deliberate evasion of DNS-based monitoring."""
    plan = SessionPlan()
    c2_ip = rng.choice(PUBLIC_IP_POOL_MALICIOUS)
    beacon_interval = rng.uniform(30, 70)
    t = rng.uniform(5, 30)
    while t < duration_s - 10:
        burst = rng.random() < 0.12
        plan.flows.append(FlowSpec(
            dst_ip=c2_ip, dst_port=rng.choice([443, 8443]), src_port=rng.randint(40000, 60000),
            start_s=t, n_packets=rng.randint(30, 70) if burst else rng.randint(6, 14),
            iat_pattern="periodic", beacon_interval_s=beacon_interval,
            fwd_len_range=(60, 220), bwd_len_range=(60, 260),
            is_tls=True, sni=None, cert_ca_type="self_signed",
            tls_version=rng.choice(["TLS1.2", "TLS1.3"]), burst_upload=burst,
        ))
        # no DNS query: direct connection to an IP address
        t += beacon_interval * rng.uniform(0.85, 1.15)
    _add_benign_background(rng, plan, duration_s, rng.randint(10, 20))
    return plan


def _plan_stalkerware(rng: random.Random, duration_s: int) -> SessionPlan:
    """Low-effort commercial spyware: static DNS infrastructure, less
    regular beaconing, periodic data uploads (analogous to CIC-AndMal2017
    / commercial stalkerware)."""
    plan = SessionPlan()
    domain, root = rng.choice(STALKERWARE_C2_DOMAINS)
    c2_ip = rng.choice(PUBLIC_IP_POOL_MALICIOUS)
    beacon_interval = rng.uniform(120, 300)
    age_days = rng.randint(90, 400)
    t = rng.uniform(10, 60)
    while t < duration_s - 10:
        burst = rng.random() < 0.2  # periodic exfiltration (location, contacts, ...)
        plan.dns.append(DnsSpec(
            t_s=t, qname=domain, root_domain=root, qtype="A",
            ttl=rng.randint(60, 300), answer_ip=c2_ip, response_bytes=rng.randint(48, 96),
            domain_age_days=age_days, in_tranco=False,
        ))
        fwd_range = (400, 1200) if burst else (60, 300)
        plan.flows.append(FlowSpec(
            dst_ip=c2_ip, dst_port=443, src_port=rng.randint(40000, 60000),
            start_s=t + rng.uniform(0.1, 0.6),
            n_packets=rng.randint(25, 55) if burst else rng.randint(8, 18),
            iat_pattern="periodic", beacon_interval_s=beacon_interval,
            fwd_len_range=fwd_range, bwd_len_range=(60, 260),
            is_tls=True, sni=domain, cert_ca_type="free_ca", tls_version="TLS1.2",
            burst_upload=burst,
        ))
        t += beacon_interval * rng.uniform(0.8, 1.2)
    _add_benign_background(rng, plan, duration_s, rng.randint(12, 22))
    return plan


_PLAN_BUILDERS = {
    "benign": _plan_benign,
    "pegasus_like": _plan_pegasus,
    "graphite_like": _plan_graphite,
    "stalkerware_like": _plan_stalkerware,
}


# --------------------------------------------------------------------------- #
# Rendering: session plan -> pcap / jsonl
# --------------------------------------------------------------------------- #

def _iat_sequence(rng: random.Random, n: int, pattern: str, beacon_interval: float) -> list:
    if pattern == "periodic":
        base = beacon_interval / max(n, 1)
        return [max(0.01, rng.gauss(base, base * 0.08)) for _ in range(n)]
    # "human": irregular intervals, typical of human browsing
    return [max(0.01, rng.lognormvariate(0.2, 0.9)) for _ in range(n)]


def _render_flow_packets(flow: FlowSpec, session_start: float, rng: random.Random) -> list:
    client = _client_ip()
    packets = []
    t = session_start + flow.start_s

    def add(src, dst, sport, dport, flags, length, ts):
        pkt = IP(src=src, dst=dst) / TCP(sport=sport, dport=dport, flags=flags)
        if length > 0:
            pkt = pkt / Raw(load=b"\x00" * length)
        pkt.time = ts
        packets.append(pkt)

    # three-way handshake
    add(client, flow.dst_ip, flow.src_port, flow.dst_port, "S", 0, t)
    t += rng.uniform(0.01, 0.05)
    add(flow.dst_ip, client, flow.dst_port, flow.src_port, "SA", 0, t)
    t += rng.uniform(0.01, 0.05)
    add(client, flow.dst_ip, flow.src_port, flow.dst_port, "A", 0, t)
    t += rng.uniform(0.01, 0.05)

    iats = _iat_sequence(rng, flow.n_packets, flow.iat_pattern, flow.beacon_interval_s)
    for i in range(flow.n_packets):
        forward = (i % 2 == 0)
        if forward:
            lo, hi = flow.fwd_len_range
            if flow.burst_upload and i == flow.n_packets - 2:
                lo, hi = max(lo, 700), max(hi, 1400)
            add(client, flow.dst_ip, flow.src_port, flow.dst_port, "PA", rng.randint(lo, hi), t)
        else:
            lo, hi = flow.bwd_len_range
            add(flow.dst_ip, client, flow.dst_port, flow.src_port, "PA", rng.randint(lo, hi), t)
        t += iats[i]

    # teardown (occasional RST instead of a clean close)
    if rng.random() < 0.08:
        add(client, flow.dst_ip, flow.src_port, flow.dst_port, "R", 0, t)
    else:
        add(client, flow.dst_ip, flow.src_port, flow.dst_port, "FA", 0, t)
        t += rng.uniform(0.01, 0.05)
        add(flow.dst_ip, client, flow.dst_port, flow.src_port, "FA", 0, t)
        t += rng.uniform(0.01, 0.05)
        add(client, flow.dst_ip, flow.src_port, flow.dst_port, "A", 0, t)

    return packets


def simulate_session(
    session_id: str,
    label: str,
    family: Optional[str],
    out_root: Path,
    duration_s: int = SESSION_DURATION_S,
    seed: Optional[int] = None,
) -> dict:
    """Generate a complete session (pcap + DNS log + TLS log + ground truth).

    ``label`` is "benign" or "malicious"; if "malicious", ``family`` must
    be one of the ``_PLAN_BUILDERS`` keys other than "benign".
    """
    rng = random.Random(seed)
    key = "benign" if label == "benign" else family
    if key not in _PLAN_BUILDERS:
        raise ValueError(f"Unknown family: {key}")
    plan = _PLAN_BUILDERS[key](rng, duration_s)

    session_dir = Path(out_root) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    session_start = 1_700_000_000.0  # fixed epoch, only used for pcap timestamp consistency

    all_packets = []
    ssl_records = []
    for flow in plan.flows:
        all_packets.extend(_render_flow_packets(flow, session_start, rng))
        if flow.is_tls:
            ssl_records.append({
                "ts": session_start + flow.start_s,
                "id.orig_h": _client_ip(), "id.orig_p": flow.src_port,
                "id.resp_h": flow.dst_ip, "id.resp_p": flow.dst_port,
                "version": flow.tls_version, "sni": flow.sni,
                "cert_ca_type": flow.cert_ca_type,
            })
    all_packets.sort(key=lambda p: p.time)

    pcap_path = session_dir / "capture.pcap"
    wrpcap(str(pcap_path), all_packets)

    dns_path = session_dir / "dns.jsonl"
    with open(dns_path, "w", encoding="utf-8") as f:
        for d in sorted(plan.dns, key=lambda x: x.t_s):
            rec = {
                "ts": session_start + d.t_s, "qname": d.qname, "root_domain": d.root_domain,
                "qtype": d.qtype, "ttl": d.ttl, "answer": d.answer_ip,
                "response_bytes": d.response_bytes, "domain_age_days": d.domain_age_days,
                "in_tranco": d.in_tranco,
            }
            f.write(json.dumps(rec) + "\n")

    ssl_path = session_dir / "ssl.jsonl"
    with open(ssl_path, "w", encoding="utf-8") as f:
        for rec in sorted(ssl_records, key=lambda x: x["ts"]):
            f.write(json.dumps(rec) + "\n")

    meta = {
        "session_id": session_id,
        "label": label,
        "family": family,
        "duration_s": duration_s,
        "n_flows": len(plan.flows),
        "n_dns": len(plan.dns),
        "pcap_path": str(pcap_path),
        "dns_path": str(dns_path),
        "ssl_path": str(ssl_path),
        "session_start": session_start,
    }
    with open(session_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta
