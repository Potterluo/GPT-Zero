---
title: GPT-Zero · 纯离线多协议 Mock LLM
---

# GPT-Zero

**纯离线**的 Mock 推理服务：一套引擎同时支持 **OpenAI Chat Completions / OpenAI Responses / Anthropic Messages** 三种调用形式（JSON + SSE），每个协议都有真实的 **fake 工具调用**，外加精确 token 计数、64-token 块 prefix cache、vLLM 风格 Prometheus 指标与开箱即用的压测 harness。

> 它**不是**模型，不产生有意义的回答。它复现的是**可观测行为**：token 计数、TTFT/TPOT、排队、缓存命中、工具调用帧、成功率——让推理栈、网关、调度器、监控面板在没有 GPU 的环境里也能被压测和回归。
>
> 协议适配形态参考 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)（只取协议转换思路，不做 tokenizer 转换）。

[![CI](https://github.com/Potterluo/GPT-Zero/actions/workflows/ci.yml/badge.svg)](https://github.com/Potterluo/GPT-Zero/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/gpt-zero)](https://pypi.org/project/gpt-zero/)
[![Docker (GHCR)](https://img.shields.io/badge/ghcr.io-potterluo%2Fgpt--zero-blue)](https://github.com/Potterluo/GPT-Zero/pkgs/container/gpt-zero)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](https://github.com/Potterluo/GPT-Zero/blob/main/LICENSE)

## 三种调用形式

| 协议 | 端点 | 流式 | 工具调用形态 |
|---|---|---|---|
| OpenAI Chat Completions | `POST /v1/chat/completions` | `data:` chunk + `[DONE]` | `delta.tool_calls` 参数分片 |
| OpenAI Responses | `POST /v1/responses` · `GET /v1/responses/{id}` | `response.created` → `output_item.added` → `*_delta` → `response.completed` | `function_call`（`call_id`） |
| Anthropic Messages | `POST /v1/messages` | `message_start` → `ping` → `content_block_*` → `message_stop` | `tool_use`（`toolu_*` id）+ `input_json_delta` |

三协议共用同一引擎：`tool_choice` 语义互通、token 计数/前缀缓存/TTFT/TPOT 同口径、工具结果回填后 agent 循环正常终止。

## 快速开始

=== "pip 安装"

    ```bash
    pip install gpt-zero
    gptzero --port 8090 --tokenizer deepseek_v3_tokenizer
    # 或从源码
    pip install -e ".[dev]"
    ```

=== "Docker"

    ```bash
    docker run --rm -p 8090:8090 ghcr.io/potterluo/gpt-zero:latest
    ```

```bash
# 冒烟
curl http://127.0.0.1:8090/health
curl -X POST http://127.0.0.1:8090/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"gpt-3.5-turbo","messages":[{"role":"user","content":"hi"}],"max_tokens":32}'
```

## 典型用法

- **Claude Code / Anthropic SDK** → `ANTHROPIC_BASE_URL=http://127.0.0.1:8090`，走 `/v1/messages`；
- **OpenAI SDK / Codex 类** → `base_url=http://127.0.0.1:8090/v1`，走 `/v1/chat/completions` 或 `/v1/responses`；
- **压测** → `python -m bench sweep --url http://127.0.0.1:8090 --profiles chat,coding,agentic`，一条命令出表 + 出图。

## 站内导航

- [架构与实现](ARCHITECTURE.md) — 协议适配层、引擎、缓存模拟、延迟模型、指标口径
- [压测与指标](BENCHMARK.md) — bench harness、profile、分位数与图表读法
- [Docker 离线镜像](DOCKER.md) — 镜像构建、断网自检、compose 用法
- [发布流程](RELEASE.md) — PyPI / GHCR / Pages 的发布链路与一次性配置

完整 README、压测结果与图表见 [GitHub 仓库](https://github.com/Potterluo/GPT-Zero)。
