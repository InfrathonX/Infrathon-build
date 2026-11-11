# Public Judge: the Work image plus the Judge code.  No holdout here;
# Dockerfile.private adds it on the host that runs the task.
ARG WORK_IMAGE=kvzip_kv_scoring.work.kvzip_kv_scoring_qwen3_8b_h100_scratch_v2:local
FROM ${WORK_IMAGE}
USER root
# Metric dependencies of the hidden suite's other workloads (kvzip_judge/hidden_workloads.py);
# kept out of the Work image, which knows only the public workloads.
RUN python3 -m pip install --no-cache-dir \
      rouge==1.0.1 nltk==3.10.4 tree-sitter==0.21.3 tree-sitter-languages==1.10.2
RUN rm -f /etc/sudoers.d/* \
 && for i in $(seq 0 15); do useradd -M -r -u $((20000 + i)) -s /usr/sbin/nologin "kvzscorer${i}"; done \
 && mkdir -p /opt/kvzip_judge/private/models/qwen3-8b /opt/kvzip_judge/private/released_gates
COPY kvzip_judge/ /opt/kvzip_judge/kvzip_judge/
COPY verify_kvzip.py /opt/kvzip_judge/verify_kvzip.py
RUN chown -R root:root /opt/kvzip_judge && chmod -R go-rwx /opt/kvzip_judge \
 && python3 -I -c "import sys; sys.path[:0]=['/opt/kvzip_judge','/opt/kvzip/runtime']; import kvzip_judge.run_judge, kvzip_judge.weights_check, tree_sitter_languages, rouge, nltk" \
 && python3 -I -m py_compile /opt/kvzip_judge/verify_kvzip.py && rm -rf /opt/kvzip_judge/__pycache__
ENV KVZIP_RUNTIME_DIR=/opt/kvzip/runtime
WORKDIR /
