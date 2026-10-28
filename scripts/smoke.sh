#!/usr/bin/env bash
# =============================================================================
# GPT-Zero v3 容器内自检（smoke test）
# =============================================================================
# 作用：起一个 mock（或复用已在跑的），逐项校验契约里的关键端点，退出码 0/1。
# 只依赖 Python 标准库 + bash，容器 `--network none` 也能跑 → 证明纯离线可用。
#
#   docker run --rm --network none gptzero:v3 smoke
#   bash scripts/smoke.sh                      # 本机（默认 8090）
#   SMOKE_NO_START=1 bash scripts/smoke.sh     # 不自起服务，只校验已在跑的服务
#   GPTZERO_PORT=9090 bash scripts/smoke.sh
#
# 覆盖：/health、/v1/models、/v1/chat/completions（非流式 + 流式 SSE + usage）、
#       /metrics（vllm:prefix_cache_* 等 vLLM 命名）、/api/stats（契约 §3.5 字段）
# =============================================================================
set -uo pipefail

PORT="${GPTZERO_PORT:-8090}"
HOST="${GPTZERO_HOST:-127.0.0.1}"
# 0.0.0.0 是「监听全部网卡」，客户端要连 127.0.0.1
CLIENT_HOST="$HOST"
case "$HOST" in 0.0.0.0|"::"|"") CLIENT_HOST="127.0.0.1" ;; esac
BASE="${SMOKE_BASE_URL:-http://${CLIENT_HOST}:${PORT}}"
START_TIMEOUT="${SMOKE_START_TIMEOUT:-60}"
NO_START="${SMOKE_NO_START:-0}"
LOG_FILE="${SMOKE_LOG_FILE:-/tmp/smoke-server.log}"

SERVER_PID=""

cleanup() {
  if [[ -n "$SERVER_PID" ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

# 选一个可用的 python（容器里是 python3，Windows Git-Bash 里可能是 python）
PY="${PYTHON:-}"
if [[ -z "$PY" ]]; then
  for cand in python3 python; do
    if command -v "$cand" >/dev/null 2>&1; then PY="$cand"; break; fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "[smoke] 错误：找不到 python，无法运行检查" >&2
  exit 1
fi

is_up() {
  "$PY" - "$BASE" <<'EOF' >/dev/null 2>&1
import sys, urllib.request
url = sys.argv[1].rstrip('/') + '/health'
try:
    with urllib.request.urlopen(url, timeout=2) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
EOF
}

echo "=================================================================="
echo " GPT-Zero v3 smoke test   target=${BASE}"
echo "=================================================================="

if is_up; then
  echo "[smoke] 复用已在运行的服务（不重复启动）"
else
  if [[ "$NO_START" == "1" ]]; then
    echo "[smoke] 失败：SMOKE_NO_START=1 但 ${BASE} 上没有在跑的服务" >&2
    exit 1
  fi
  echo "[smoke] 启动 mock：port=${PORT} log=${LOG_FILE}"
  "$PY" -m gptzero \
    --host "$HOST" --port "$PORT" \
    --ttft-ms 8 --tpot-ms 1 --jitter 0 \
    --log-file "$LOG_FILE" \
    >"${LOG_FILE}.stdout" 2>&1 &
  SERVER_PID=$!

  deadline=$((SECONDS + START_TIMEOUT))
  until is_up; do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
      echo "[smoke] 失败：服务进程提前退出，日志尾部：" >&2
      tail -n 40 "${LOG_FILE}.stdout" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "[smoke] 失败：${START_TIMEOUT}s 内 /health 未就绪，日志尾部：" >&2
      tail -n 40 "${LOG_FILE}.stdout" >&2 || true
      exit 1
    fi
    sleep 0.5
  done
  echo "[smoke] 服务已就绪（pid=${SERVER_PID}）"
fi

TOKENIZER_HINT="${GPTZERO_TOKENIZER:-auto}"
"$PY" - "$BASE" "$TOKENIZER_HINT" <<'PYEOF'
"""纯标准库的端点校验：health / models / chat(stream+non-stream) / metrics / stats."""
import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1].rstrip("/")
TOKENIZER_HINT = sys.argv[2] if len(sys.argv) > 2 else "auto"

FAILS = []
CHECKS = []


def ok(name, detail=""):
    CHECKS.append((True, name, detail))
    print(f"  [PASS] {name}" + (f" — {detail}" if detail else ""))


def bad(name, detail):
    CHECKS.append((False, name, detail))
    FAILS.append(f"{name}: {detail}")
    print(f"  [FAIL] {name} — {detail}")


def request(path, payload=None, timeout=60, headers=None):
    url = BASE + path
    data = None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    if payload is not None:
        data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data, headers=hdrs,
                                 method="POST" if payload is not None else "GET")
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(path):
    with request(path) as r:
        return r.status, json.loads(r.read().decode())


print("\n[1/7] GET /health")
try:
    status, body = get_json("/health")
    if status == 200 and body.get("status") == "ok":
        ok("/health", f'status={body.get("status")}')
    else:
        bad("/health", f"status={status} body={body}")
except Exception as exc:  # noqa: BLE001
    bad("/health", f"{type(exc).__name__}: {exc}")

print("\n[2/7] GET /v1/models（OpenAI 列表格式）")
model_id = "gpt-3.5-turbo"
try:
    status, body = get_json("/v1/models")
    items = body.get("data") or []
    if status == 200 and isinstance(items, list) and items and "id" in items[0]:
        model_id = items[0]["id"]
        ok("/v1/models", f"{len(items)} 个模型，首个 id={model_id} object={items[0].get('object')}")
    else:
        bad("/v1/models", f"status={status} body={str(body)[:200]}")
except Exception as exc:  # noqa: BLE001
    bad("/v1/models", f"{type(exc).__name__}: {exc}")

print("\n[3/7] POST /v1/chat/completions（非流式）")
prompt = "smoke: write a short function that adds two integers"
try:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 24,
        "temperature": 0,
    }
    with request("/v1/chat/completions", payload) as r:
        body = json.loads(r.read().decode())
    choice = (body.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = body.get("usage") or {}
    problems = []
    if choice.get("finish_reason") not in ("stop", "length", "tool_calls"):
        problems.append(f"finish_reason={choice.get('finish_reason')}")
    if not isinstance(msg.get("content"), str):
        problems.append("message.content 不是字符串")
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if not isinstance(usage.get(key), int):
            problems.append(f"usage.{key} 缺失")
    if problems:
        bad("chat/non-stream", "; ".join(problems))
    else:
        ok("chat/non-stream",
           f'finish={choice["finish_reason"]} completion_tokens={usage["completion_tokens"]}')
except Exception as exc:  # noqa: BLE001
    bad("chat/non-stream", f"{type(exc).__name__}: {exc}")

print("\n[4/7] POST /v1/chat/completions（流式 SSE + include_usage）")
try:
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 16,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    deltas = 0
    content = []
    saw_done = False
    saw_usage = False
    events = 0
    with request("/v1/chat/completions", payload) as r:
        ctype = r.headers.get("Content-Type", "")
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                saw_done = True
                break
            events += 1
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if chunk.get("usage") and not choices:
                saw_usage = True
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content"):
                    deltas += 1
                    content.append(delta["content"])
    problems = []
    if "event-stream" not in ctype:
        problems.append(f"Content-Type={ctype}")
    if not saw_done:
        problems.append("缺少 data: [DONE]")
    if deltas < 1:
        problems.append("没有收到 delta.content 增量")
    if not saw_usage:
        problems.append("include_usage=true 未追加 usage chunk")
    if problems:
        bad("chat/stream", "; ".join(problems))
    else:
        ok("chat/stream", f"{events} 个 chunk，{deltas} 个增量帧，usage chunk 已出现")
except Exception as exc:  # noqa: BLE001
    bad("chat/stream", f"{type(exc).__name__}: {exc}")

print("\n[5/7] GET /metrics（vLLM 命名子集）")
try:
    with request("/metrics") as r:
        text = r.read().decode()
    required = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:kv_cache_usage_perc",
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prefix_cache_queries_total",
        "vllm:prefix_cache_hits_total",
        "vllm:prefix_cache_evictions_total",
        "vllm:time_to_first_token_seconds_sum",
        "vllm:time_to_first_token_seconds_count",
        "vllm:time_per_output_token_seconds_sum",
        "vllm:e2e_request_latency_seconds_sum",
        "gptzero:tool_calls_total",
        "gptzero:cache_hit_tokens_total",
    ]
    # 直方图/摘要族的 # HELP 只出现在族名上（vllm:x_seconds），不在 _sum/_count 上
    def family(name):
        for suf in ("_bucket", "_count", "_sum"):
            if name.endswith(suf) and not name.endswith("_total"):
                return name[: -len(suf)]
        return name

    missing = [name for name in required if name not in text]
    no_help = [
        name for name in required
        if f"# HELP {family(name)}" not in text and f"# HELP {name}" not in text
    ]
    no_type = [
        name for name in required
        if f"# TYPE {family(name)}" not in text and f"# TYPE {name}" not in text
    ]
    if missing:
        bad("/metrics", f"缺指标: {', '.join(missing[:6])}")
    elif no_help:
        bad("/metrics", f"缺 # HELP 行: {', '.join(no_help[:6])}")
    elif no_type:
        bad("/metrics", f"缺 # TYPE 行: {', '.join(no_type[:6])}")
    else:
        ok("/metrics", f"{len(required)} 个必需指标齐全，含 # HELP/# TYPE")

    # prefix_cache_* 必须是可解析的数值
    for name in ("vllm:prefix_cache_queries_total", "vllm:prefix_cache_hits_total"):
        for line in text.splitlines():
            if line.startswith(name + " "):
                float(line.split()[1])
                break
    ok("metrics/prefix_cache", "vllm:prefix_cache_* 数值可解析")
except Exception as exc:  # noqa: BLE001
    bad("/metrics", f"{type(exc).__name__}: {exc}")

print("\n[6/7] GET /api/stats（契约 §3.5 稳定字段）")
try:
    status, stats = get_json("/api/stats")
    scalars = [
        "uptime_s", "total_requests", "active_requests", "completed_requests",
        "failed_requests", "prompt_tokens", "completion_tokens", "total_tokens",
        "tool_calls", "rps", "prompt_tokens_per_s", "completion_tokens_per_s",
        "cache_hit_tokens", "cache_query_tokens", "cache_hit_rate",
    ]
    dicts = ["ttft_ms", "tpot_ms", "itl_ms", "e2e_ms", "prompt_tokens_hist", "completion_tokens_hist"]
    missing = [k for k in scalars if k not in stats]
    missing += [k for k in dicts if not isinstance(stats.get(k), dict)]
    badkeys = []
    for k in dicts:
        d = stats.get(k) or {}
        for p in ("p50", "p90", "p99", "mean", "max"):
            if p not in d:
                badkeys.append(f"{k}.{p}")
    if not isinstance(stats.get("models"), dict):
        missing.append("models")
    if not isinstance(stats.get("cache"), dict):
        missing.append("cache")
    if missing:
        bad("/api/stats", f"缺字段: {', '.join(missing[:8])}")
    elif badkeys:
        bad("/api/stats", f"分位键缺: {', '.join(badkeys[:6])}")
    else:
        ok("/api/stats", f"15 标量 + 6 分位字典 + models/cache 齐全；total_requests={stats['total_requests']}")
except Exception as exc:  # noqa: BLE001
    bad("/api/stats", f"{type(exc).__name__}: {exc}")

print("\n[7/7] 前缀缓存生效（两次相同长 prompt → cache_hit_tokens 增长）")
try:
    long_prompt = "prefix-cache smoke: " + ("x" * 4000)
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": long_prompt}],
        "max_tokens": 8,
    }
    for _ in range(2):
        with request("/v1/chat/completions", payload) as r:
            r.read()
    _, stats = get_json("/api/stats")
    hit = stats.get("cache_hit_tokens", 0)
    rate = stats.get("cache_hit_rate", 0)
    if hit > 0:
        ok("prefix-cache", f"cache_hit_tokens={hit} rate={rate}")
    else:
        bad("prefix-cache", f"两次相同 prompt 后 cache_hit_tokens 仍为 {hit}")
except Exception as exc:  # noqa: BLE001
    bad("prefix-cache", f"{type(exc).__name__}: {exc}")

print("\n" + "=" * 66)
passed = sum(1 for c in CHECKS if c[0])
print(f" 结果: {passed}/{len(CHECKS)} 项通过   (tokenizer={TOKENIZER_HINT})")
if FAILS:
    print(" 失败项:")
    for f in FAILS:
        print(f"   - {f}")
    print("=" * 66)
    sys.exit(1)
print(" 结论: SMOKE PASS — 离线 mock 服务契约自检全部通过")
print("=" * 66)
sys.exit(0)
PYEOF

rc=$?
echo
if (( rc == 0 )); then
  echo "[smoke] PASS (exit 0)"
else
  echo "[smoke] FAIL (exit ${rc})"
  if [[ -f "${LOG_FILE}.stdout" ]]; then
    echo "[smoke] 服务端日志尾部：" >&2
    tail -n 30 "${LOG_FILE}.stdout" >&2 || true
  fi
fi
exit "$rc"
