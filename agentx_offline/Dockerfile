# AgentX offline test image.
#
# Build context = this directory (agentx_offline/). Prepare the cache first:
#   pwsh scripts/prepare_agentx_cache.ps1
#   bash scripts/prepare_agentx_cache.sh
#
# Build (dataset + tokenizer baked in, ~3.6 GB):
#   docker build -t agentx-offline:1.0.7 .
# Slim variant (mount /opt/hf at runtime instead):
#   docker build --build-arg WITH_CACHE=0 -t agentx-offline:slim .
FROM python:3.11-slim

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
ARG WITH_CACHE=1
ENV PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST} \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1

WORKDIR /opt/agentx

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -c "import agentx_cli, aiperf; print('agentx-harness ready')"

COPY runner.py report.py entrypoint.sh /opt/agentx/
COPY assets /opt/agentx/assets
RUN chmod +x /opt/agentx/entrypoint.sh

# Dataset Arrow cache + DeepSeek-V3 tokenizer (built by the prepare script).
COPY --chown=10001:10001 cache/ /opt/hf/
RUN if [ "$WITH_CACHE" != "1" ]; then rm -rf /opt/hf/datasets; fi

RUN useradd --create-home --uid 10001 agentx && \
    mkdir -p /results && chown -R agentx:agentx /results /opt/agentx
USER agentx
VOLUME ["/results"]

ENTRYPOINT ["/opt/agentx/entrypoint.sh"]
