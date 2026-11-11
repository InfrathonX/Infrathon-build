# Egress gateway for one sandbox. Everything the sandbox sends leaves through here.
FROM python:3.12-slim
RUN apt-get update \
 && apt-get install -y --no-install-recommends iptables iproute2 ca-certificates \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir "mitmproxy==12.1.2"
COPY entry.sh ibt_filter.py /opt/ibt/
# The denylist is mounted per run at /opt/ibt/denylist.json.
RUN chmod 0755 /opt/ibt/entry.sh
ENTRYPOINT ["/opt/ibt/entry.sh"]
