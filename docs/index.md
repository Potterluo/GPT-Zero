---
title: GPT-Zero 文档总览
---

# GPT-Zero 文档总览

这里是 GPT-Zero 的**文档区**；产品介绍、安装与在线演示见[落地页](https://potterluo.github.io/GPT-Zero/)。

GPT-Zero 是一个纯离线的多协议 Mock LLM 推理服务：一套引擎同时支持 **OpenAI Chat Completions / OpenAI Responses / Anthropic Messages** 三种调用形式（JSON + SSE），每个协议都有真实的 fake 工具调用，内置精确 token 计数、64-token 块前缀缓存、vLLM 风格 Prometheus 指标与闭环压测 harness。

## 站内文档

| 页面 | 内容 |
|---|---|
| [架构与实现](ARCHITECTURE.md) | 组件图、请求生命周期、协议适配层（protocols.py）、前缀缓存模拟、延迟模型、指标口径 |
| [压测与指标](BENCHMARK.md) | bench harness 设计、四种工作负载 profile、指标定义、图表读法 |
| [Docker 离线镜像](DOCKER.md) | 镜像里有什么、运行期零网络原理、断网 smoke、compose 用法、known-good 命令 |
| [发布流程](RELEASE.md) | tag → PyPI（openai-mock）/ GHCR 多架构镜像 / Pages 的发布链路与一次性配置 |

## 快速链接

- GitHub 仓库：<https://github.com/Potterluo/GPT-Zero>
- PyPI：<https://pypi.org/project/openai-mock/>（`pip install openai-mock`）
- GHCR 镜像：`ghcr.io/potterluo/gpt-zero:{版本,latest}`（amd64 + arm64）
- 给 LLM/回答引擎的站点摘要：<https://potterluo.github.io/GPT-Zero/llms.txt>
