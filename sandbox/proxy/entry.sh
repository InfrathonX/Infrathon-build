#!/bin/sh
# Gateway rules for one sandbox:
#   sandbox -> proxy IP :443/:80   names mapped in the sandbox's /etc/hosts; TLS is intercepted, paths filtered
#   sandbox -> other IP :443/:80   opaque passthrough, hostname (SNI) filter only
#   sandbox -> anything :53        forwarded (name resolution)
#   sandbox -> anything else       dropped
set -eu
: "${IBT_PROXY_IP:?}"
IN_IF="$(ip -o -4 addr show | awk -v ip="$IBT_PROXY_IP" 'index($4, ip"/")==1 {print $2}')"
OUT_IF="$(ip -o -4 route show default | awk '{print $5; exit}')"
[ -n "$IN_IF" ] && [ -n "$OUT_IF" ] && [ "$IN_IF" != "$OUT_IF" ] || {
  echo "cannot identify interfaces: in=$IN_IF out=$OUT_IF" >&2; ip -o addr >&2; ip route >&2; exit 1; }

# Set by the orchestrator with --sysctl, because /proc/sys is read-only here.
[ "$(cat /proc/sys/net/ipv4/ip_forward)" = 1 ] || { echo "ip_forward is off" >&2; exit 1; }

iptables -t nat -A PREROUTING -i "$IN_IF" -p tcp -m multiport --dports 80,443 -j REDIRECT --to-ports 8081
iptables -t nat -A POSTROUTING -o "$OUT_IF" -j MASQUERADE
iptables -P FORWARD DROP
iptables -A FORWARD -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A FORWARD -i "$IN_IF" -o "$OUT_IF" -p udp --dport 53 -j ACCEPT
iptables -A FORWARD -i "$IN_IF" -o "$OUT_IF" -p tcp --dport 53 -j ACCEPT
iptables -A INPUT -i "$IN_IF" -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT
iptables -A INPUT -i "$IN_IF" -p tcp -m multiport --dports 8081 -j ACCEPT
iptables -A INPUT -i "$IN_IF" -j DROP

# Generate the CA once and hand over only the public certificate.
mkdir -p /root/.mitmproxy
if [ ! -f /root/.mitmproxy/mitmproxy-ca-cert.pem ]; then
  (timeout 6 mitmdump --listen-host 127.0.0.1 --listen-port 18999 >/dev/null 2>&1 || true)
fi
cp /root/.mitmproxy/mitmproxy-ca-cert.pem /ca-out/ibt-ca.crt
chmod 0644 /ca-out/ibt-ca.crt

exec mitmdump \
  --mode transparent@8081 \
  --set connection_strategy=lazy --set upstream_cert=false \
  --set stream_large_bodies=1m --set block_global=false --set block_private=false \
  --set termlog_verbosity=warn --set flow_detail=0 \
  -s /opt/ibt/ibt_filter.py
