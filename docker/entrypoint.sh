#!/usr/bin/env bash
# GPT-Zero v4 容器入口
# ---------------------------------------------------------------------------
# 子命令：
#   mock   启动离线 Mock 推理服务（默认，监听 0.0.0.0:8090）
#   bench  运行压测 harness（python -m bench ...）
#   smoke  容器内自检：起 mock → 打端点 → 退出码 0/1
#   shell  进入交互 shell（调试用）
#   pytest 在容器内跑单元测试（调试用）
#
# mock 的默认参数全部可用环境变量覆盖（CLI 参数仍然优先，因为追加在后面）：
#   GPTZERO_HOST            监听地址           默认 0.0.0.0
#   GPTZERO_PORT            监听端口           默认 8090
#   GPTZERO_MODEL           模型名（逗号分隔）  默认不传（用服务端默认列表）
#   GPTZERO_TTFT_MS         TTFT 基准（ms）     默认 180
#   GPTZERO_TPOT_MS         TPOT 基准（ms）     默认 8
#   GPTZERO_CACHE_BLOCKS    前缀缓存块数         默认 8192
#   GPTZERO_TOKENIZER       auto|heuristic|<dir> 默认 auto
#   GPTZERO_LOG_FILE        JSONL 请求日志      默认 /data/requests.jsonl（未挂卷时退化为 /tmp/…）
#   GPTZERO_ARGS            额外参数（原样拼到命令行末尾）
# ---------------------------------------------------------------------------
set -euo pipefail

APP_DIR="${APP_DIR:-/app}"
PORT="${GPTZERO_PORT:-8090}"
HOST="${GPTZERO_HOST:-0.0.0.0}"

log() { printf '[entrypoint] %s\n' "$*" >&2; }

usage() {
  cat >&2 <<'EOF'
用法: entrypoint.sh <mock|bench|smoke|shell|pytest> [args...]

  mock   [gptzero CLI 参数...]   启动 mock 服务（默认动作）
  bench  [bench CLI 参数...]     例: bench run --url http://mock:8090 --profile coding
  smoke                          容器内端到端自检
  shell                          交互 shell
  pytest [pytest 参数...]        容器内单元测试

环境变量: GPTZERO_HOST / GPTZERO_PORT / GPTZERO_MODEL / GPTZERO_TTFT_MS /
          GPTZERO_TPOT_MS / GPTZERO_CACHE_BLOCKS / GPTZERO_TOKENIZER /
          GPTZERO_LOG_FILE / GPTZERO_ARGS
EOF
}

# 模块缺失时给出可操作的提示（bench 并行开发时，镜像按契约先行支持）
need_module() {
  local mod="$1"
  if ! python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('$mod') else 1)" 2>/dev/null; then
    log "错误：镜像内缺少 '$mod' 模块（python -m $mod 不可用）。"
    log "请重新构建镜像以纳入最新代码："
    log "  scripts/offline_image.ps1 -NoCache   # 或 docker build -t gptzero:v4 ."
    exit 3
  fi
}

cmd="${1:-mock}"
if [[ $# -gt 0 ]]; then shift; fi

case "$cmd" in
  mock)
    log_file="${GPTZERO_LOG_FILE:-/data/requests.jsonl}"
    # 日志目录不可写时退化为 /tmp，保证 --network none 下也能跑起来；
    # 用 >> 只创建/追加，不截断历史日志（容器重启后仍能追溯之前的请求）
    if ! mkdir -p "$(dirname "$log_file")" 2>/dev/null || ! : >>"$log_file" 2>/dev/null; then
      fallback="/tmp/gptzero-requests.jsonl"
      log "警告：$log_file 不可写，改用 $fallback"
      log_file="$fallback"
      mkdir -p /tmp
      : >>"$log_file" 2>/dev/null || true
    fi

    args=(
      --host "$HOST"
      --port "$PORT"
      --ttft-ms "${GPTZERO_TTFT_MS:-180}"
      --tpot-ms "${GPTZERO_TPOT_MS:-8}"
      --cache-blocks "${GPTZERO_CACHE_BLOCKS:-8192}"
      --tokenizer "${GPTZERO_TOKENIZER:-auto}"
      --log-file "$log_file"
    )
    if [[ -n "${GPTZERO_MODEL:-}" ]]; then
      args+=(--model "$GPTZERO_MODEL")
    fi

    if [[ -n "${GPTZERO_ARGS:-}" ]]; then
      # 有意做 shell 分词：允许 GPTZERO_ARGS="--ttft-ms 50 --jitter 0"
      read -r -a _extra <<<"${GPTZERO_ARGS}"
      args+=("${_extra[@]}")
    fi

    log "启动 mock: ${HOST}:${PORT}  log=${log_file}  tokenizer=${GPTZERO_TOKENIZER:-auto}"
    exec python -m gptzero "${args[@]}" "$@"
    ;;

  bench)
    need_module bench
    exec python -m bench "$@"
    ;;

  smoke)
    if [[ ! -f "$APP_DIR/scripts/smoke.sh" ]]; then
      log "错误：找不到 $APP_DIR/scripts/smoke.sh"
      exit 3
    fi
    exec bash "$APP_DIR/scripts/smoke.sh" "$@"
    ;;

  pytest)
    exec python -m pytest -q "$@" .
    ;;

  shell)
    exec bash -l
    ;;

  -h|--help|help)
    usage
    exit 0
    ;;

  *)
    log "未知子命令: $cmd"
    usage
    exit 2
    ;;
esac
