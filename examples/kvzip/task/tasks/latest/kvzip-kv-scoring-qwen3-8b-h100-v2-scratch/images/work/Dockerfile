# Work image: the agent's workstation.  Holds the brief, the starting workspace,
# the read-only evaluation runtime and the public tests.  No Judge code, no hidden
# data, no reference solution.  The frozen model is mounted read-only at
# /opt/models/qwen3-8b at run time (it is not in the image).
FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-devel AS deps

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_TELEMETRY=1 \
    KVZIP_MODEL_DIR=/opt/models/qwen3-8b KVZIP_RUNTIME_DIR=/opt/kvzip/runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git jq procps ripgrep sudo util-linux xz-utils \
 && rm -rf /var/lib/apt/lists/* \
 && (id -u agent >/dev/null 2>&1 || useradd -m -u 1000 -s /bin/bash agent) \
 && echo 'agent ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/agent && chmod 440 /etc/sudoers.d/agent

RUN python3 -m pip install --no-cache-dir \
      transformers==4.57.1 tokenizers==0.22.2 huggingface-hub==0.36.2 safetensors==0.8.0 accelerate==1.15.0 \
      pandas==3.1.1 pyarrow==26.0.1 nvidia-ml-py==13.580.82 pytest==9.1.1

FROM deps AS work
COPY runtime/ /opt/kvzip/runtime/
COPY public_tests/ /workspace/public_tests/
COPY --from=public_suite public_suite.jsonl.gz /workspace/public_tests/data/public_suite.jsonl.gz
COPY workdir/ /app/workdir/
COPY AGENT_TASK.md /app/workdir/kvzip_lab/AGENT_TASK.md
RUN mkdir -p /opt/models/qwen3-8b /workspace/public_tests/results /logs/agent \
 && chown -R root:root /opt/kvzip /workspace/public_tests \
 && find /opt/kvzip /workspace/public_tests -type d -exec chmod 0555 {} + \
 && find /opt/kvzip /workspace/public_tests -type f -exec chmod 0444 {} + \
 && chmod 0555 /workspace/public_tests/run_public_tests.sh \
 && chown agent:agent /workspace/public_tests/results \
 && chmod 0755 /workspace/public_tests/results \
 && chown -R agent:agent /app/workdir /logs/agent \
 && python3 -c "import sys; sys.path.insert(0, '/opt/kvzip/runtime'); import kvzip_eval.harness, kvzip_eval.metrics"
USER agent
WORKDIR /app/workdir/kvzip_lab
