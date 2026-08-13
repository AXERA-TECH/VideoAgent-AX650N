from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from transformers import AutoTokenizer
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()
# 1. 初始化 FastAPI
app = FastAPI(title="Qwen3 Tokenizer API", version="1.0")

# 2. 全局加载 Tokenizer (避免每次请求都加载)
# 如果是本地文件，请替换为本地路径，例如 "./qwen3_tokenizer"
MODEL_PATH = os.getenv("Tokenizer_MODEL_PATH", "")
print("MODEL_PATH :",MODEL_PATH )
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=False)

# 3. 定义请求数据结构
class EncodeRequest(BaseModel):
    text: str
    add_special_tokens: bool = True  # 是否添加 <\|im_start\|> 等特殊标记

class DecodeRequest(BaseModel):
    token_ids: List[int]
    skip_special_tokens: bool = True # 是否跳过特殊标记

class BatchEncodeRequest(BaseModel):
    texts: List[str]                # 接收文本列表
    padding: bool = False            # 是否自动填充到最长序列
    truncation: bool = True         # 是否截断超长文本
    max_length: Optional[int] = None # 最大长度限制
    add_special_tokens: bool = True

# 4. 定义 API 路由

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {"status": "running", "model": tokenizer.name_or_path}

@app.post("/encode")
async def encode_text(request: EncodeRequest):
    """
    将文本转换为 Token IDs
    """
    try:
        # 调用 tokenizer
        encoding = tokenizer(
            request.text, 
            add_special_tokens=request.add_special_tokens,
            return_tensors="pt" # 返回 PyTorch 张量，也可以选 "np"
        )
        
        token_ids = encoding.input_ids[0].tolist()
        
        return {
            "text": request.text,
            "token_ids": token_ids,
            "count": len(token_ids), # 返回长度，方便前端计算
            "special_tokens_added": request.add_special_tokens
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_encode")
async def batch_encode(request: BatchEncodeRequest):
    """
    批量将文本列表转换为 Token IDs
    """
    try:
        # 调用 tokenizer，直接传入列表即可实现批量处理
        encoding = tokenizer(
            request.texts,
            padding=request.padding,
            truncation=request.truncation,
            max_length=request.max_length,
            add_special_tokens=request.add_special_tokens,
            return_tensors="pt" 
        )
        
        # 将 PyTorch Tensor 转换为 Python 列表以便 JSON 序列化
        # input_ids 的形状通常是 [batch_size, sequence_length]
        batch_token_ids = encoding.input_ids.tolist()
        
        # 计算每个文本的长度
        lengths = [len(tokens) for tokens in batch_token_ids]
        
        return {
            "batch_size": len(request.texts),
            "token_ids_batch": batch_token_ids,
            "lengths": lengths
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/decode")
async def decode_tokens(request: DecodeRequest):
    """
    将 Token IDs 还原为文本
    """
    try:
        text = tokenizer.decode(
            request.token_ids, 
            skip_special_tokens=request.skip_special_tokens
        )
        return {
            "token_ids": request.token_ids,
            "text": text,
            "count": len(request.token_ids)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 5. 启动服务
if __name__ == "__main__":
    # 启动命令：uvicorn tokenizer_api:app --host 0.0.0.0 --port 8001
    import uvicorn
    port = int(os.getenv("Tokenizer_API_PORT", 8007))
    uvicorn.run(app, host="0.0.0.0", port=port)
