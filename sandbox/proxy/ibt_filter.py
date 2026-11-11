"""mitmproxy addon: policy and log for one sandbox's traffic.

Intercept: connections addressed to the gateway itself, which are names mapped
to it in the sandbox's /etc/hosts. The client trusts our CA, so paths are visible
and filtered.

Passthrough: everything else, including all traffic from nested containers, which
do not trust our CA. Only the hostname (SNI) is visible. A denied hostname is not
passed through: it falls into interception, so a client either fails certificate
validation or receives the 403 from request().
"""
from __future__ import annotations

import ipaddress
import json
import os
import re
import time

from mitmproxy import http, tls
from mitmproxy.proxy import server_hooks

GATEWAY_IP = os.environ["IBT_PROXY_IP"]
LOG_PATH = os.environ.get("IBT_PROXY_LOG", "/logs/egress.jsonl")
with open("/opt/ibt/denylist.json", encoding="utf-8") as fh:
    DENY = [re.compile(p, re.IGNORECASE) for p in json.load(fh)["patterns"]]
_log = open(LOG_PATH, "a", buffering=1, encoding="utf-8")


def _emit(**row) -> None:
    row["ts"] = round(time.time(), 3)
    _log.write(json.dumps(row, sort_keys=True) + "\n")


def _denied(text: str) -> str | None:
    for pattern in DENY:
        if pattern.search(text or ""):
            return pattern.pattern
    return None


def _private(host: str) -> bool:
    try:
        return not ipaddress.ip_address(host).is_global
    except ValueError:
        return False


def _intercepted(address) -> bool:
    return bool(address) and address[0] == GATEWAY_IP


def tls_clienthello(data: tls.ClientHelloData) -> None:
    sni = data.client_hello.sni or ""
    dest = data.context.server.address
    if _intercepted(dest):
        return
    hit = _denied(sni)
    if hit:
        _emit(event="deny_sni", sni=sni, rule=hit, mode="passthrough")
        return  # intercepted: the client cannot validate our certificate
    data.ignore_connection = True
    _emit(event="pass", sni=sni, dest=dest[0] if dest else None, mode="passthrough")


def server_connect(data: server_hooks.ServerConnectionHookData) -> None:
    address = data.server.address
    if _intercepted(address):
        # The client asked for a mapped name: the real destination is its SNI.
        if data.client.sni:
            data.server.address = (data.client.sni, 443)
        else:
            _emit(event="deny_no_sni", dest=address[0], port=address[1])
            data.server.error = "blocked by sandbox policy"
        return
    if address and _private(address[0]):
        _emit(event="deny_private", dest=address[0], port=address[1])
        data.server.error = "blocked by sandbox policy"


def request(flow: http.HTTPFlow) -> None:
    if _intercepted(flow.server_conn.address) and not flow.client_conn.sni:
        # Plain HTTP to a mapped name: route by the Host header.
        flow.request.host = flow.request.host_header or flow.request.host
        flow.request.port = 80
    host = flow.request.pretty_host
    hit = _denied(f"{host}{flow.request.path}")
    _emit(event="deny_http" if hit else "http", method=flow.request.method,
          host=host, path=flow.request.path[:400], rule=hit, mode="intercept")
    if hit:
        flow.response = http.Response.make(
            403, b"blocked by sandbox policy\n", {"content-type": "text/plain"})
