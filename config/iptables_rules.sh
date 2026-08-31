#!/bin/bash
# iptables rules for the capture node (Raspberry Pi 4). Run as root on the
# capture node after starting hostapd and Unbound.
set -euo pipefail

AP_IFACE="${AP_IFACE:-wlan0}"
RESOLVER_PORT="${RESOLVER_PORT:-5353}"

# Force all DNS queries (UDP and TCP), including ones aimed at alternative
# resolvers (e.g. 8.8.8.8, 1.1.1.1), toward the local Unbound resolver.
iptables -t nat -A PREROUTING -i "$AP_IFACE" -p udp --dport 53 -j REDIRECT --to-port "$RESOLVER_PORT"
iptables -t nat -A PREROUTING -i "$AP_IFACE" -p tcp --dport 53 -j REDIRECT --to-port "$RESOLVER_PORT"

# Best-effort block of well-known DoH provider IPs, to force the device to
# fall back to the classic, plaintext DNS resolver. This is a known
# methodological limitation: it only catches known DoH providers, and
# doesn't address DoH endpoints that aren't yet on this list.
DOH_IPS=(
    "1.1.1.1"       # Cloudflare
    "1.0.0.1"       # Cloudflare
    "8.8.8.8"       # Google
    "8.8.4.4"       # Google
    "45.90.28.0/24" # NextDNS
)

for ip in "${DOH_IPS[@]}"; do
    iptables -A OUTPUT -d "$ip" -p tcp --dport 443 -j DROP
done

echo "Rules applied on $AP_IFACE (DNS redirect -> :$RESOLVER_PORT, known DoH providers blocked)."
