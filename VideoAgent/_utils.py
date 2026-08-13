import asyncio
import html
import json
import logging
import os
import re
import numbers
from dataclasses import dataclass
from functools import wraps
from hashlib import md5
from typing import Any, Union
from PIL import Image
import base64
from typing import Literal
import io


import numpy as np
logger = logging.getLogger("nano-graphrag")

def write_json(json_obj, file_name):
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_obj, f, indent=2, ensure_ascii=False)


def load_json(file_name):
    """
    从指定文件中加载JSON数据并返回解析后的Python对象。

    参数:
        file_name (str): 要读取的JSON文件的路径。

    返回:
        dict/list/None: 如果文件存在且成功解析，返回JSON数据对应的Python对象（通常是字典或列表）；
                        如果文件不存在，返回None。
    """
    # 检查文件是否存在，若不存在则直接返回None
    if not os.path.exists(file_name):
        return None
    
    # 打开文件并使用UTF-8编码读取内容，然后解析为JSON对象
    with open(file_name, encoding="utf-8") as f:
        return json.load(f)
    
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    model_name: str
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        # Had to fix this as the embedding function took only one named argument put it's passed in
        # positionally, now we need to pass both
        kwargs['model_name'] = self.model_name
        
        # If there are positional arguments, convert them to keyword arguments
        if args:
            # Assuming the first positional argument is always 'texts'
            if len(args) == 1 and isinstance(args[0], list):
                kwargs['texts'] = args[0]
            else:
                raise ValueError("Unexpected positional arguments. Expected a single list of texts")
        # Call the function with the updated keyword arguments
        return await self.func(**kwargs)   
    
def clean_output(raw_text):
    # 先移除成对的 <think>...</think>
    cleaned = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
    # 再移除残留的单独 <think> 或 </think> 标签
    cleaned = re.sub(r'</?think>', '', cleaned)
    return cleaned


def compute_mdhash_id(content, prefix: str = ""):
    return prefix + md5(content.encode()).hexdigest()


def always_get_an_event_loop() -> asyncio.AbstractEventLoop:
    try:
        # If there is already an event loop, use it.
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If in a sub-thread, create a new event loop.
        logger.info("Creating a new event loop in a sub-thread.")
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")