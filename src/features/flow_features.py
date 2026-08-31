"""
Flow-level feature extraction from pcap files: a Python equivalent of
CICFlowMeter, covering the same broad feature categories the tool is
known for — temporal, volumetric, TCP-flag, rate, and bulk-transfer
features (the bursty transmission pattern typical of data exfiltration).

Works on any pcap with valid-timestamp IP/TCP packets, whether produced
by the simulator (src/simulate) or a real capture from tcpdump/SpyGuard
on a physical testbed. It doesn't reproduce the original Java tool's 84
columns exactly, but covers the same categories with equivalent semantics.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from scapy.all import rdpcap, IP, TCP

ACTIVITY_GAP_S = 5.0  # gap between packets beyond which the flow is considered "idle"
MIN_PACKETS_PER_FLOW = 5  # minimum packet count for a flow to be considered informative
BULK_IAT_THRESHOLD_S = 1.0  # max gap between packets to stay within the same "bulk"
BULK_MIN_PACKETS = 4  # minimum packet count for a burst to count as a "bulk" (CICFlowMeter convention)


def _flow_key(pkt) -> tuple:
    ip, tcp = pkt[IP], pkt[TCP]
    endpoints = tuple(sorted([(ip.src, tcp.sport), (ip.dst, tcp.dport)]))
    return ("tcp", endpoints[0], endpoints[1])


def _safe_mean(values) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _safe_std(values) -> float:
    return float(np.std(values)) if len(values) else 0.0


def _safe_max(values) -> float:
    return float(np.max(values)) if len(values) else 0.0


def _safe_min(values) -> float:
    return float(np.min(values)) if len(values) else 0.0


def _compute_bulk_stats(times: list, lens: list) -> tuple:
    """Detects sequences ("bulks") of consecutive same-direction packets
    with non-zero payload and a small gap between them, and computes their
    average bytes, average packet count, and average byte rate. These
    "bulk" features capture the bursty transmission pattern typical of
    data exfiltrated in periodic blocks."""
    pairs = [(t, l) for t, l in sorted(zip(times, lens)) if l > 0]
    if not pairs:
        return 0.0, 0.0, 0.0

    bulks = []
    cur_n, cur_bytes, cur_start, prev_t = 0, 0, None, None
    for t, l in pairs:
        if prev_t is not None and (t - prev_t) <= BULK_IAT_THRESHOLD_S and cur_n > 0:
            cur_n += 1
            cur_bytes += l
        else:
            if cur_n >= BULK_MIN_PACKETS:
                bulks.append((cur_n, cur_bytes, max(prev_t - cur_start, 1e-6)))
            cur_n, cur_bytes, cur_start = 1, l, t
        prev_t = t
    if cur_n >= BULK_MIN_PACKETS:
        bulks.append((cur_n, cur_bytes, max(prev_t - cur_start, 1e-6)))

    if not bulks:
        return 0.0, 0.0, 0.0

    avg_bytes = _safe_mean([b for _, b, _ in bulks])
    avg_packets = _safe_mean([n for n, _, _ in bulks])
    avg_rate = _safe_mean([b / d for _, b, d in bulks])
    return avg_bytes, avg_packets, avg_rate


def extract_flows_from_pcap(pcap_path, t_max_s: float | None = None, session_start: float | None = None) -> pd.DataFrame:
    """Reconstructs the bidirectional flows (5-tuple) present in the pcap
    and computes their temporal, volumetric, TCP-flag, rate, and bulk
    features.

    If ``t_max_s`` is given, only packets with a timestamp within
    ``[session_start, session_start + t_max_s]`` are considered (used by
    the detection-latency metric).
    """
    packets = rdpcap(str(pcap_path))
    if session_start is not None and t_max_s is not None:
        packets = [p for p in packets if float(p.time) <= session_start + t_max_s]

    flows = defaultdict(list)
    origin = {}
    for pkt in packets:
        if IP not in pkt or TCP not in pkt:
            continue
        key = _flow_key(pkt)
        if key not in origin:
            origin[key] = (pkt[IP].src, pkt[TCP].sport)
        flows[key].append(pkt)

    rows = []
    for key, pkts in flows.items():
        if len(pkts) < MIN_PACKETS_PER_FLOW:
            continue
        pkts = sorted(pkts, key=lambda p: float(p.time))
        fwd_src, fwd_sport = origin[key]

        fwd_lens, bwd_lens = [], []
        fwd_times, bwd_times = [], []
        fwd_header_bytes, bwd_header_bytes = 0, 0
        flags_count = defaultdict(int)
        fwd_flags_count, bwd_flags_count = defaultdict(int), defaultdict(int)
        all_times, all_lens = [], []

        for p in pkts:
            ts = float(p.time)
            payload_len = len(bytes(p[TCP].payload))
            tcp_total_len = len(bytes(p[TCP]))
            header_len = max(tcp_total_len - payload_len, 0)

            all_times.append(ts)
            all_lens.append(payload_len)

            is_fwd = (p[IP].src == fwd_src and p[TCP].sport == fwd_sport)
            flag_letters = str(p[TCP].flags)
            for f in flag_letters:
                flags_count[f] += 1

            if is_fwd:
                fwd_lens.append(payload_len)
                fwd_times.append(ts)
                fwd_header_bytes += header_len
                for f in flag_letters:
                    fwd_flags_count[f] += 1
            else:
                bwd_lens.append(payload_len)
                bwd_times.append(ts)
                bwd_header_bytes += header_len
                for f in flag_letters:
                    bwd_flags_count[f] += 1

        all_times_sorted = sorted(all_times)
        duration = all_times_sorted[-1] - all_times_sorted[0]

        flow_iat = np.diff(all_times_sorted) if len(all_times_sorted) > 1 else np.array([0.0])
        fwd_times_sorted = sorted(fwd_times)
        bwd_times_sorted = sorted(bwd_times)
        fwd_iat = np.diff(fwd_times_sorted) if len(fwd_times_sorted) > 1 else np.array([0.0])
        bwd_iat = np.diff(bwd_times_sorted) if len(bwd_times_sorted) > 1 else np.array([0.0])

        active_periods, idle_periods = [], []
        run = 0.0
        for g in flow_iat:
            if g > ACTIVITY_GAP_S:
                if run > 0:
                    active_periods.append(run)
                idle_periods.append(g)
                run = 0.0
            else:
                run += g
        if run > 0:
            active_periods.append(run)

        total_bytes = sum(fwd_lens) + sum(bwd_lens)
        total_packets = len(fwd_lens) + len(bwd_lens)
        all_pkt_lens = fwd_lens + bwd_lens

        fwd_bulk_bytes, fwd_bulk_packets, fwd_bulk_rate = _compute_bulk_stats(fwd_times, fwd_lens)
        bwd_bulk_bytes, bwd_bulk_packets, bwd_bulk_rate = _compute_bulk_stats(bwd_times, bwd_lens)

        rows.append({
            # --- temporal ---
            "duration": duration,
            "flow_iat_mean": _safe_mean(flow_iat), "flow_iat_std": _safe_std(flow_iat),
            "flow_iat_min": _safe_min(flow_iat), "flow_iat_max": _safe_max(flow_iat),
            "fwd_iat_total": float(sum(fwd_iat)), "fwd_iat_mean": _safe_mean(fwd_iat),
            "fwd_iat_std": _safe_std(fwd_iat), "fwd_iat_max": _safe_max(fwd_iat), "fwd_iat_min": _safe_min(fwd_iat),
            "bwd_iat_total": float(sum(bwd_iat)), "bwd_iat_mean": _safe_mean(bwd_iat),
            "bwd_iat_std": _safe_std(bwd_iat), "bwd_iat_max": _safe_max(bwd_iat), "bwd_iat_min": _safe_min(bwd_iat),
            "active_mean": _safe_mean(active_periods), "active_std": _safe_std(active_periods),
            "active_max": _safe_max(active_periods), "active_min": _safe_min(active_periods),
            "idle_mean": _safe_mean(idle_periods), "idle_std": _safe_std(idle_periods),
            "idle_max": _safe_max(idle_periods), "idle_min": _safe_min(idle_periods),

            # --- volumetric ---
            "fwd_packets": len(fwd_lens), "bwd_packets": len(bwd_lens), "total_packets": total_packets,
            "fwd_bytes": sum(fwd_lens), "bwd_bytes": sum(bwd_lens), "total_bytes": total_bytes,
            "fwd_pkt_len_mean": _safe_mean(fwd_lens), "fwd_pkt_len_std": _safe_std(fwd_lens),
            "fwd_pkt_len_max": _safe_max(fwd_lens), "fwd_pkt_len_min": _safe_min(fwd_lens),
            "bwd_pkt_len_mean": _safe_mean(bwd_lens), "bwd_pkt_len_std": _safe_std(bwd_lens),
            "bwd_pkt_len_max": _safe_max(bwd_lens), "bwd_pkt_len_min": _safe_min(bwd_lens),
            "pkt_len_mean": _safe_mean(all_pkt_lens), "pkt_len_std": _safe_std(all_pkt_lens),
            "pkt_len_var": float(np.var(all_pkt_lens)) if all_pkt_lens else 0.0,
            "pkt_len_min": _safe_min(all_pkt_lens), "pkt_len_max": _safe_max(all_pkt_lens),
            "avg_packet_size": total_bytes / total_packets if total_packets else 0.0,
            "fwd_avg_segment_size": _safe_mean(fwd_lens), "bwd_avg_segment_size": _safe_mean(bwd_lens),
            "down_up_ratio": (len(bwd_lens) / len(fwd_lens)) if fwd_lens else 0.0,
            "act_data_pkt_fwd": sum(1 for l in fwd_lens if l > 0),
            "fwd_header_bytes": fwd_header_bytes, "bwd_header_bytes": bwd_header_bytes,

            # --- TCP flags (aggregate and per-direction) ---
            "syn_count": flags_count.get("S", 0), "ack_count": flags_count.get("A", 0),
            "fin_count": flags_count.get("F", 0), "rst_count": flags_count.get("R", 0),
            "psh_count": flags_count.get("P", 0), "urg_count": flags_count.get("U", 0),
            "fwd_psh_count": fwd_flags_count.get("P", 0), "bwd_psh_count": bwd_flags_count.get("P", 0),
            "fwd_urg_count": fwd_flags_count.get("U", 0), "bwd_urg_count": bwd_flags_count.get("U", 0),

            # --- rate ---
            "flow_byte_rate": total_bytes / duration if duration > 0 else float(total_bytes),
            "flow_packet_rate": total_packets / duration if duration > 0 else float(total_packets),
            "fwd_packet_rate": len(fwd_lens) / duration if duration > 0 else float(len(fwd_lens)),
            "bwd_packet_rate": len(bwd_lens) / duration if duration > 0 else float(len(bwd_lens)),

            # --- bulk (bursty exfiltration) ---
            "fwd_bulk_bytes_mean": fwd_bulk_bytes, "fwd_bulk_packets_mean": fwd_bulk_packets,
            "fwd_bulk_rate_mean": fwd_bulk_rate,
            "bwd_bulk_bytes_mean": bwd_bulk_bytes, "bwd_bulk_packets_mean": bwd_bulk_packets,
            "bwd_bulk_rate_mean": bwd_bulk_rate,
        })

    return pd.DataFrame(rows)


def aggregate_session_flow_features(df_flows: pd.DataFrame) -> dict:
    """Aggregates per-flow features into a single session-level vector:
    the unit of classification (and of the train/val/test split) is the
    session, not the individual flow."""
    if df_flows.empty:
        return {"n_flows": 0}

    agg = {"n_flows": len(df_flows)}
    for col in df_flows.columns:
        agg[f"{col}_mean"] = float(df_flows[col].mean())
        agg[f"{col}_std"] = float(df_flows[col].std(ddof=0))
        agg[f"{col}_max"] = float(df_flows[col].max())
    agg["total_bytes_sum"] = float(df_flows["total_bytes"].sum())
    agg["total_packets_sum"] = float(df_flows["total_packets"].sum())
    agg["rst_count_sum"] = float(df_flows["rst_count"].sum())
    agg["fwd_bulk_bytes_sum"] = float(df_flows["fwd_bulk_bytes_mean"].sum())
    return agg
