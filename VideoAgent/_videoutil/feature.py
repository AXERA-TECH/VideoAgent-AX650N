import os
import numpy as np

from PIL import Image
from decord import VideoReader, cpu

from .._llm import Qwen3VLEmbedder

_MAX_FRAMES = 5  # DashScope qwen3-vl-embedding 每次 image batch 上限为 5

model = None


def _get_model():
    global model
    if model is None:
        model = Qwen3VLEmbedder()
    return model


def _load_frames(video_path: str, max_frames: int = _MAX_FRAMES):
    """用 decord 均匀采样至多 max_frames 帧，返回 PIL Image 列表。"""
    vr = VideoReader(video_path, ctx=cpu(0))
    total = len(vr)
    indices = np.linspace(0, total - 1, min(total, max_frames), dtype=int)
    return [Image.fromarray(vr[i].asnumpy()) for i in indices]


def encode_video_segments(video_path):
    frames = _load_frames(video_path)
    documents = [{"video": frames}]
    embedder = _get_model()
    embeddings = embedder.embed_multimodal(documents)
    return embeddings


def encode_string_query(query: str):
    documents = [{"text": query}]
    embedder = _get_model()
    embeddings = embedder.embed_multimodal(documents)
    return embeddings
