# =============================================================================
# GPT-Zero v3 — 纯离线 Mock 推理服务镜像
# =============================================================================
# 构建期：只访问清华 PyPI 镜像（可用 --network none 之外的正常网络构建）。
# 运行期：**零网络依赖**（不装 curl、不拉模型、不访问外网；HEALTHCHECK 用标准库）。
#
#   docker build -t gptzero:v3 .
#   docker build -t gptzero:v3 --build-arg WITH_TRANSFORMERS=1 .   # 精确 token 计数
#   docker run --rm -p 8090:8090 gptzero:v3                        # = mock
#   docker run --rm --network none gptzero:v3 smoke                # 离线自检
#
# 多阶段：builder 里把依赖装到 /install，runtime 只从 builder 拷贝 /usr/local，
# 因此最终镜像不含 pip 缓存、wheel 包、构建中间层。
# =============================================================================
ARG BASE_IMAGE=python:3.11-slim

# -----------------------------------------------------------------------------
# 阶段 1：依赖构建（联网，仅 build 期）
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS builder

ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn
# 1 = 额外安装 transformers（约 +160MB），用于精确 token 计数；0 = 用 heuristic 回退
ARG WITH_TRANSFORMERS=0

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# 先拷依赖清单，依赖不变时命中层缓存（改代码不会触发重装依赖）
COPY docker/requirements-runtime.txt docker/requirements-tokenizer.txt /build/

# 全部依赖都是 manylinux cp311 wheel → 不需要 gcc/g++/python3-dev
RUN set -eux; \
    pip install --upgrade -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" pip setuptools wheel; \
    pip install --prefix=/install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        -r /build/requirements-runtime.txt; \
    if [ "${WITH_TRANSFORMERS}" = "1" ]; then \
        pip install --prefix=/install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
            -r /build/requirements-tokenizer.txt; \
    fi; \
    # 运行期要做 `pip install -e .`（离线、--no-build-isolation），需要 setuptools/wheel 在镜像里
    pip install --prefix=/install -i "${PIP_INDEX_URL}" --trusted-host "${PIP_TRUSTED_HOST}" \
        --upgrade setuptools wheel; \
    # 剔除测试/缓存类垃圾，减小体积
    find /install -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    find /install -type d -name 'tests' -prune -exec rm -rf {} + || true

# -----------------------------------------------------------------------------
# 阶段 2：运行镜像
# -----------------------------------------------------------------------------
FROM ${BASE_IMAGE} AS runtime

LABEL org.opencontainers.image.title="gptzero" \
      org.opencontainers.image.description="Offline OpenAI-compatible mock LLM inference service for agentic benchmarks (GPT-Zero v3)" \
      org.opencontainers.image.version="3.x" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/gpt-zero/gpt-zero"

# 运行期环境：unbuffered 输出、不写 pyc、不联网（HF 相关全部离线）
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_HOME=/tmp/hf \
    GPTZERO_HOST=0.0.0.0 \
    GPTZERO_PORT=8090 \
    APP_DIR=/app

# 依赖（含 setuptools/wheel，供离线 editable 安装使用）
COPY --from=builder /install /usr/local

# 非 root 用户
RUN set -eux; \
    groupadd --gid 10001 gptzero; \
    useradd --uid 10001 --gid gptzero --create-home --shell /bin/bash gptzero; \
    mkdir -p /data /app/results /tmp/matplotlib; \
    chown -R gptzero:gptzero /data /tmp/matplotlib

WORKDIR /app

# 源码整体拷入（.dockerignore 已排除 .venv / .git / results / dist / __pycache__）
COPY --chown=gptzero:gptzero . /app

# 离线可编辑安装：--no-deps 避免构建期重新解析依赖图，--no-build-isolation 避免下载构建后端
RUN set -eux; \
    pip install --no-deps --no-build-isolation -e /app; \
    chmod +x /app/docker/entrypoint.sh; \
    find /app -maxdepth 3 -type d -name '__pycache__' -prune -exec rm -rf {} + ; \
    chown -R gptzero:gptzero /app /home/gptzero

USER gptzero:gptzero

EXPOSE 8090

# 用标准库探活，不额外装 curl / wget
HEALTHCHECK --interval=10s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python3", "-c", "import os,sys,urllib.request;\
p=os.environ.get('GPTZERO_PORT','8090');\
r=urllib.request.urlopen('http://127.0.0.1:'+str(p)+'/health', timeout=2);\
sys.exit(0 if r.status==200 else 1)"]

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["mock"]
