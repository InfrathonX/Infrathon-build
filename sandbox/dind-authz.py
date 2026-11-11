#!/usr/bin/env python3
"""Docker authorization plugin for the nested daemon.

The sandbox's dockerd has broad capabilities, so a nested container with the
same reach could leave the sandbox. SForge and the tasks need none of this.
"""
from __future__ import annotations

import base64
import http.server
import json
import os
import re
import socketserver

SOCKET = "/run/docker/plugins/ibt-authz.sock"
LOG = os.environ.get("IBT_AUTHZ_LOG", "/var/log/ibt-authz.jsonl")
ALLOWED_BIND_ROOTS = ("/work", "/tmp", "/opt/agent-runtimes", "/var/run/docker/netns")
DENIED_CAPS = {"ALL", "SYS_ADMIN", "SYS_MODULE", "SYS_RAWIO", "SYS_PTRACE", "SYS_BOOT",
               "DAC_READ_SEARCH", "BPF", "PERFMON", "MAC_ADMIN", "MAC_OVERRIDE", "SYSLOG", "NET_ADMIN"}
ALLOWED_DEVICE = re.compile(r"^/dev/(nvidia[-a-z0-9]*|nvidia-caps/.*|fuse)$")


def bind_ok(source: str) -> bool:
    if not source.startswith("/"):
        return True  # a named volume
    path = os.path.normpath(source)
    return any(path == root or path.startswith(root + "/") for root in ALLOWED_BIND_ROOTS)


def check_create(body: dict) -> str | None:
    hc = body.get("HostConfig") or {}
    if hc.get("Privileged"):
        return "privileged containers are not available in this environment"
    for key in ("PidMode", "UsernsMode", "CgroupnsMode", "UTSMode"):
        if hc.get(key) == "host":
            return f"{key}=host is not available in this environment"
    denied = {c.upper().removeprefix("CAP_") for c in (hc.get("CapAdd") or [])} & DENIED_CAPS
    if denied:
        return f"capability not available in this environment: {sorted(denied)}"
    for opt in hc.get("SecurityOpt") or []:
        if "unconfined" in opt:
            return f"security option not available in this environment: {opt}"
    for spec in hc.get("Binds") or []:
        source = spec.split(":", 1)[0]
        if not bind_ok(source):
            return f"bind mount source not available in this environment: {source} (use /work or /tmp)"
    for mount in hc.get("Mounts") or []:
        if mount.get("Type") == "bind" and not bind_ok(mount.get("Source", "")):
            return f"bind mount source not available in this environment: {mount.get('Source')} (use /work or /tmp)"
    for device in hc.get("Devices") or []:
        if not ALLOWED_DEVICE.match(device.get("PathOnHost", "")):
            return f"device not available in this environment: {device.get('PathOnHost')}"
    return None


def decide(method: str, uri: str, body: dict) -> str | None:
    path = re.sub(r"^/v[0-9.]+", "", uri.split("?", 1)[0])
    if method == "POST" and path == "/containers/create":
        return check_create(body)
    if method == "POST" and re.fullmatch(r"/containers/[^/]+/exec", path) and body.get("Privileged"):
        return "privileged exec is not available in this environment"
    if method == "POST" and path == "/volumes/create" and (body.get("DriverOpts") or {}):
        return "volume driver options are not available in this environment"
    if path.startswith("/plugins") and method != "GET":
        return "plugins are not available in this environment"
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def address_string(self) -> str:
        return "docker"

    def _reply(self, payload: dict) -> None:
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/vnd.docker.plugins.v1.2+json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.path == "/Plugin.Activate":
            return self._reply({"Implements": ["authz"]})
        if self.path == "/AuthZPlugin.AuthZRes":
            return self._reply({"Allow": True})
        if self.path != "/AuthZPlugin.AuthZReq":
            return self._reply({"Err": "unknown"})
        request = json.loads(raw or b"{}")
        body = {}
        if request.get("RequestBody"):
            try:
                body = json.loads(base64.b64decode(request["RequestBody"]))
            except (ValueError, TypeError):
                body = {}
        # The daemon sends "RequestUri"; a wrong key here would allow everything silently.
        method, uri = request["RequestMethod"], request["RequestUri"]
        reason = decide(method, uri, body if isinstance(body, dict) else {})
        if reason:
            with open(LOG, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"event": "deny", "method": method, "uri": uri[:300], "reason": reason}) + "\n")
            return self._reply({"Allow": False, "Msg": reason})
        return self._reply({"Allow": True})


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        request, _ = super().get_request()
        return request, ("docker", 0)


def main() -> None:
    os.makedirs(os.path.dirname(SOCKET), exist_ok=True)
    if os.path.exists(SOCKET):
        os.unlink(SOCKET)
    Server(SOCKET, Handler).serve_forever()


if __name__ == "__main__":
    main()
