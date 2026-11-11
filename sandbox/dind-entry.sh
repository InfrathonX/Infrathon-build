#!/bin/sh
# Start the sandbox's own dockerd, with the host contract SForge expects: the
# NVIDIA runtime is the default and GPUs are selected with NVIDIA_VISIBLE_DEVICES.
set -eu

# Writable /sys and cgroups are needed because this container is not --privileged.
mount -o remount,rw /sys 2>/dev/null || :
mount -o remount,rw /sys/fs/cgroup 2>/dev/null || :

# cgroup v2 nesting: move out of the root group so controllers can be delegated.
if [ -f /sys/fs/cgroup/cgroup.controllers ]; then
  mkdir -p /sys/fs/cgroup/init
  xargs -rn1 < /sys/fs/cgroup/cgroup.procs > /sys/fs/cgroup/init/cgroup.procs 2>/dev/null || :
  sed -e 's/ / +/g' -e 's/^/+/' < /sys/fs/cgroup/cgroup.controllers \
    > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || :
fi

if [ -n "${IBT_PROXY_IP:-}" ]; then
  # All traffic leaves through the gateway, and nothing else on the bridge is reachable.
  ip route replace default via "$IBT_PROXY_IP"
  OUT_IF="$(ip -o -4 route show default | awk '{print $5; exit}')"
  iptables -A OUTPUT -o "$OUT_IF" -d "$IBT_PROXY_IP" -j ACCEPT
  iptables -A OUTPUT -o "$OUT_IF" -d "$IBT_SUBNET" -j REJECT
  iptables -N DOCKER-USER 2>/dev/null || :
  iptables -I DOCKER-USER -o "$OUT_IF" -d "$IBT_SUBNET" -j REJECT
  iptables -I DOCKER-USER -o "$OUT_IF" -d "$IBT_PROXY_IP" -j ACCEPT

  # Names whose paths we filter resolve to the gateway. Containers nested under
  # this daemon get their own /etc/hosts and go through passthrough instead.
  for host in ${IBT_INTERCEPT_HOSTS:-}; do
    echo "$IBT_PROXY_IP $host" >> /etc/hosts
  done
  cp /ca/ibt-ca.crt /usr/local/share/ca-certificates/ibt-ca.crt
  update-ca-certificates >/dev/null
fi

mkdir -p /etc/docker /ibt-sock /run/docker/plugins
nvidia-ctk runtime configure --runtime=docker --set-as-default >/dev/null
python3 /usr/local/bin/dind-authz.py &
for _ in 1 2 3 4 5 6 7 8 9 10; do
  [ -S /run/docker/plugins/ibt-authz.sock ] && break
  sleep 0.2
done

# Data root must be a real filesystem (a volume): overlay on overlay fails.
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/etc/docker/daemon.json")
d = json.loads(p.read_text()) if p.exists() else {}
d.update({
    "exec-opts": ["native.cgroupdriver=cgroupfs"],
    "storage-driver": "overlay2",
    "hosts": ["unix:///ibt-sock/docker.sock"],
    "group": "docker",
    "features": {"buildkit": True},
    "dns": ["8.8.8.8"],
    "authorization-plugins": ["ibt-authz"],
})
p.write_text(json.dumps(d, indent=2))
PY
exec dockerd "$@"
