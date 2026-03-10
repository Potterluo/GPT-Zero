#!/usr/bin/env python3
"""
Mock LLM API - OpenAI 兼容 Mock 服务器
"""
import argparse
import json
import time
import uuid
from typing import AsyncGenerator

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI(title="Mock LLM", version="1.1.0")
# 配置
DEFAULT_REPLY = "你好！我是测试助手。"
SUPPORTED_MODELS = ["gpt-3.5-turbo", "qwen3-32b", "claude-3-opus", "deepseek-r1"]


def make_chunk(id: str, model: str, content: str = "", finish: bool = False) -> str:
    """生成 SSE chunk"""
    choice = {"index": 0, "delta": {"content": content} if content else {}, 
              "finish_reason": "stop" if finish else None}
    data = {"id": id, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [choice]}
    return f"data: {json.dumps(data)}\n\n"


async def stream_response(messages: list, model: str) -> AsyncGenerator[str, None]:
    """流式响应生成器"""
    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    content = messages[-1].get("content", "")[:50] if messages else DEFAULT_REPLY
    
    # role
    yield make_chunk(chunk_id, model)
    
    # content chunks (模拟逐字输出)
    for i in range(0, len(content), 2):
        yield make_chunk(chunk_id, model, content[i:i+2])
    
    # finish
    yield make_chunk(chunk_id, model, finish=True)
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """OpenAI 兼容的聊天完成接口"""
    data = await request.json()
    model = data.get("model", "gpt-3.5-turbo")
    stream = data.get("stream", False)
    messages = data.get("messages", [])
    
    if model not in SUPPORTED_MODELS:
        model = "test-model"
    
    if stream:
        return StreamingResponse(
            stream_response(messages, model),
            media_type="text/event-stream"
        )
    
    # 非流式响应
    content = messages[-1].get("content", "")[:50] if messages else DEFAULT_REPLY
    return JSONResponse({
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": "stop"
        }],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20}
    })


@app.get("/v1/models")
async def list_models():
    """模型列表"""
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "mock"} 
                 for m in SUPPORTED_MODELS]
    }


@app.get("/health")
async def health():
    """健康检查"""
    return {"status": "ok"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", "-p", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port)