import os
import sys
import logging
import tempfile
import torch
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()



# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Qwen3VL Embedding API",
    description="API for Qwen3VL Embedding model",
    version="1.0.0"
)

# --- 加载模型 ---
logger.info("Loading Qwen3VL Embedding model...")

try:
    # 添加路径以导入 Qwen3VLEmbedder
    # sys.path.insert(0, '/data/huangjie/video_agent/videoRAG/black/videorag/mytools')
    from qwen3_vl_embedding import Qwen3VLEmbedder

    model_path = os.getenv("EMBEDDING_MODEL_PATH", "")

    embedding_model = Qwen3VLEmbedder(
        model_name_or_path=model_path,
        max_length=8192,
        max_frames=16,
        fps=10
    )

    logger.info("Qwen3VL Embedding model loaded successfully!")

except Exception as e:
    logger.error(f"Failed to load model: {e}")
    embedding_model = None

# --- 数据模型 ---
class EmbeddingRequest(BaseModel):
    text: Optional[str] = None
    image: Optional[str] = None  # 图像URL或本地路径
    video: Optional[str] = None  # 视频路径
    normalize: bool = True

class BatchEmbeddingRequest(BaseModel):
    documents: List[Dict[str, Any]]  # 文档列表
    normalize: bool = True

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    shape: List[int]
    count: int

# --- API 端点 ---

@app.post("/v1/embed/text")
async def embed_text(
    text: str = Form(...),
    normalize: bool = Form(True)
) -> Dict[str, Any]:
    """为文本生成嵌入向量"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        documents = [{"text": text}]
        embeddings = embedding_model.process(documents, normalize=normalize)

        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "count": embeddings.shape[0],
            "text": text
        }

    except Exception as e:
        logger.error(f"Text embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embed/image")
async def embed_image(
    image_path: str = Form(...),
    normalize: bool = Form(True)
) -> Dict[str, Any]:
    """为图像生成嵌入向量"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not image_path:
        raise HTTPException(status_code=400, detail="Image path required")

    try:
        documents = [{"image": image_path}]
        embeddings = embedding_model.process(documents, normalize=normalize)

        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "count": embeddings.shape[0],
            "image_path": image_path
        }

    except Exception as e:
        logger.error(f"Image embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embed/video")
async def embed_video(
    video_path: str = Form(...),
    normalize: bool = Form(True)
) -> Dict[str, Any]:
    """为视频生成嵌入向量"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not video_path:
        raise HTTPException(status_code=400, detail="Video path required")

    try:
        documents = [{"video": video_path}]
        embeddings = embedding_model.process(documents, normalize=normalize)

        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "count": embeddings.shape[0],
            "video_path": video_path
        }

    except Exception as e:
        logger.error(f"Video embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embed/batch")
async def embed_batch(request: BatchEmbeddingRequest) -> Dict[str, Any]:
    """批量生成嵌入向量"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not request.documents:
        raise HTTPException(status_code=400, detail="Documents required")

    try:
        embeddings = embedding_model.process(request.documents, normalize=request.normalize)

        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "count": embeddings.shape[0],
            "document_count": len(request.documents)
        }

    except Exception as e:
        logger.error(f"Batch embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/embed/multimodal")
async def embed_multimodal(request: EmbeddingRequest) -> Dict[str, Any]:
    """生成多模态嵌入向量（支持文本、图像、视频）"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # 构建文档
    document = {}
    if request.text:
        document["text"] = request.text
    if request.image:
        document["image"] = request.image
    if request.video:
        document["video"] = request.video

    if not document:
        raise HTTPException(status_code=400, detail="At least one of text, image, or video is required")

    try:
        embeddings = embedding_model.process([document], normalize=request.normalize)

        return {
            "embeddings": embeddings.tolist(),
            "shape": list(embeddings.shape),
            "count": embeddings.shape[0],
            "document": document
        }

    except Exception as e:
        logger.error(f"Multimodal embedding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/similarity")
async def compute_similarity(
    text1: str = Form(...),
    text2: str = Form(...)
) -> Dict[str, Any]:
    """计算两个文本的相似度"""

    if embedding_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 获取两个文本的嵌入
        embeddings = embedding_model.process(
            [{"text": text1}, {"text": text2}],
            normalize=True
        )

        # 计算余弦相似度
        similarity = torch.nn.functional.cosine_similarity(
            embeddings[0].unsqueeze(0),
            embeddings[1].unsqueeze(0)
        ).item()

        return {
            "similarity": float(similarity)
        }

    except Exception as e:
        logger.error(f"Similarity computation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/v1/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "model_loaded": embedding_model is not None,
        "model_name": "Qwen3VL-Embedding-2B"
    }

@app.get("/v1/info")
async def model_info():
    """获取模型信息"""
    if embedding_model is None:
        return {"error": "Model not loaded"}

    return {
        "model_name": "Qwen3VL-Embedding-2B",
        "model_path": model_path,
        "supported_modalities": ["text", "image", "video"],
        "max_length": 8192,
        "max_frames": 16,
        "fps": 10
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("EMBEDDING_API_PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)
