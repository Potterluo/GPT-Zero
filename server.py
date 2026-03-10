# mock_llm_fastapi.py
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json, time, uuid, asyncio, logging
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uvicorn
import argparse

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mock LLM API",
    description="一个模拟的LLM API服务器，用于测试和开发",
    version="1.0.0"
)

# 添加CORS支持
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 配置常量
DEFAULT_REPLY = {
    "content": "你好！我是测试助手。",
    "reasoning_content": "模拟推理中..."
}

# 支持的模型列表
SUPPORTED_MODELS = {
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "gpt-4": "GPT-4",
    "gpt-4-turbo": "GPT-4 Turbo",
    "claude-3-opus": "Claude 3 Opus",
    "claude-3-sonnet": "Claude 3 Sonnet",
    "test-model": "测试模型"
}

# 动态回复模板
RESPONSE_TEMPLATES = {
    "default": "这是一个基于请求的动态回复：{user_message}",
    "code": "我理解您需要代码帮助。基于您的请求：{user_message}",
    "creative": "让我来创作一些内容：{user_message}",
    "analysis": "分析结果：{user_message}"
}

def generate_response(messages: List[Dict], model: str) -> str:
    """根据消息历史生成回复"""
    try:
        if not messages:
            return DEFAULT_REPLY["content"]

        last_message = messages[-1].get("content", "")

        # 安全地处理字符串内容
        try:
            content_str = str(last_message) if last_message else ""
        except (UnicodeEncodeError, UnicodeDecodeError):
            content_str = last_message.encode('utf-8', errors='ignore').decode('utf-8')

        # 根据模型和消息内容选择回复模板
        if "代码" in content_str or "code" in content_str.lower():
            template = RESPONSE_TEMPLATES["code"]
        elif "分析" in content_str or "analysis" in content_str.lower():
            template = RESPONSE_TEMPLATES["analysis"]
        elif "创作" in content_str or "creative" in content_str.lower():
            template = RESPONSE_TEMPLATES["creative"]
        else:
            template = RESPONSE_TEMPLATES["default"]

        return template.format(user_message=content_str[:100])  # 限制长度

    except Exception as e:
        logger.error(f"生成回复时出错: {e}")
        return DEFAULT_REPLY["content"]

async def stream_generator(data: dict, model: str):
    """生成流式响应"""
    try:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())
        messages = data.get("messages", [])

        # 生成回复内容
        content = generate_response(messages, model)

        logger.info(f"开始流式响应: {chunk_id}, 模型: {model}")

        # role chunk
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

        # content chunks
        chunk_size = 3
        for i in range(0, len(content), chunk_size):
            chunk_content = content[i:i+chunk_size]
            yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'content': chunk_content}, 'finish_reason': None}]})}\n\n"
            await asyncio.sleep(0.03)  # 模拟网络延迟

        # reasoning_content for supported models
        if "claude" in model.lower() or "gpt-4" in model.lower():
            reasoning = DEFAULT_REPLY.get("reasoning_content", "")
            for i in range(0, len(reasoning), chunk_size):
                chunk_content = reasoning[i:i+chunk_size]
                yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {'reasoning_content': chunk_content}, 'finish_reason': None}]})}\n\n"
                await asyncio.sleep(0.02)

        # done
        yield f"data: {json.dumps({'id': chunk_id, 'object': 'chat.completion.chunk', 'created': created, 'model': model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"流式响应完成: {chunk_id}")

    except Exception as e:
        logger.error(f"流式生成错误: {e}")
        # 即使出错也要发送完成标记
        yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat(request: Request):
    """聊天完成API端点"""
    try:
        data = await request.json()
        model = data.get("model", "gpt-3.5-turbo")
        stream = data.get("stream", False)
        messages = data.get("messages", [])

        # 验证模型
        if model not in SUPPORTED_MODELS:
            logger.warning(f"未知模型: {model}, 使用默认模型")
            model = "test-model"

        logger.info(f"收到请求 - 模型: {model}, 流式: {stream}, 消息数: {len(messages)}")

        # 生成回复内容
        response_content = generate_response(messages, model)

        if stream:
            return StreamingResponse(
                stream_generator(data, model),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "Connection": "keep-alive"
                }
            )
        else:
            response_data = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(str(messages)) * 2,
                    "completion_tokens": len(response_content) * 2,
                    "total_tokens": len(str(messages)) * 2 + len(response_content) * 2
                }
            }

            # 为支持的模型添加推理内容
            if "claude" in model.lower() or "gpt-4" in model.lower():
                response_data["choices"][0]["message"]["reasoning_content"] = DEFAULT_REPLY.get("reasoning_content", "")

            logger.info(f"非流式响应完成: {response_data['id']}")
            return JSONResponse(response_data)

    except json.JSONDecodeError:
        logger.error("无效的JSON数据")
        raise HTTPException(status_code=400, detail="Invalid JSON data")
    except Exception as e:
        logger.error(f"处理请求时出错: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

# 添加健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "timestamp": int(time.time()),
        "models": list(SUPPORTED_MODELS.keys())
    }

# 添加模型列表端点
@app.get("/v1/models")
async def list_models():
    """列出支持的模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "mock-llm"
            }
            for model_id in SUPPORTED_MODELS.keys()
        ]
    }

# 根路径
@app.get("/")
async def root():
    """根路径信息"""
    return {
        "message": "Mock LLM API Server",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/v1/chat/completions",
            "models": "/v1/models",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock LLM API服务器")
    parser.add_argument(
        "--port", 
        "-p", 
        type=int, 
        default=8080, 
        help="服务器监听端口 (默认: 8080)"
    )
    parser.add_argument(
        "--host",
        "-H",
        type=str,
        default="0.0.0.0",
        help="服务器监听地址 (默认: 0.0.0.0)"
    )
    args = parser.parse_args()
    
    logger.info(f"启动Mock LLM API服务器于 {args.host}:{args.port}...")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")