# One image for both sandbox roles: the inner dockerd ("dind") and the builder.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl gnupg git jq rsync patch util-linux iptables iproute2 \
      procps python3 python3-venv pigz xz-utils kmod less file unzip zip tmux \
      build-essential ripgrep fd-find openssh-client wget sudo \
 && install -m 0755 -d /etc/apt/keyrings \
 && curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc \
 && echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu noble stable" \
      > /etc/apt/sources.list.d/docker.list \
 && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
      | gpg --dearmor -o /etc/apt/keyrings/nvidia.gpg \
 && echo "deb [signed-by=/etc/apt/keyrings/nvidia.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" \
      > /etc/apt/sources.list.d/nvidia.list \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
      docker-ce docker-ce-cli containerd.io docker-buildx-plugin nvidia-container-toolkit \
 && rm -rf /var/lib/apt/lists/*
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin UV_NO_MODIFY_PATH=1 sh
RUN UV_TOOL_DIR=/opt/uv-tools UV_TOOL_BIN_DIR=/usr/local/bin UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    uv tool install --python 3.12 'huggingface_hub==1.23.0' \
 && chmod -R a+rX /opt/uv-tools
COPY dind-entry.sh dind-authz.py /usr/local/bin/
RUN chmod 0755 /usr/local/bin/dind-entry.sh /usr/local/bin/dind-authz.py
