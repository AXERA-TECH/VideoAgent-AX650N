import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI


def _pil_to_base64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class Qwen3VLModel:
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None, **kwargs):
        self.model_name = model_name or os.getenv("VLM_MODEL_NAME", "Qwen3-VL-2B-Instruct")
        base_url = base_url or os.getenv("VLM_API_BASE_URL", "http://localhost:8001/v1")
        api_key = api_key or os.getenv("VLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _convert_content(self, content: list) -> list:
        """将本地内容（PIL Image、视频帧列表）转换为 API 兼容的格式。"""
        converted = []
        for item in content:
            if item["type"] == "text":
                converted.append(item)
            elif item["type"] == "image":
                img = item["image"]
                if isinstance(img, Image.Image):
                    b64 = _pil_to_base64(img)
                    converted.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
                else:
                    converted.append(item)
            elif item["type"] == "video":
                # 将视频帧列表转换为多个 image_url 消息
                frames = item["video"]
                if isinstance(frames, list):
                    for frame in frames:
                        if isinstance(frame, Image.Image):
                            b64 = _pil_to_base64(frame)
                            converted.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                            })
                else:
                    converted.append(item)
            else:
                converted.append(item)
        return converted

    def generate_result(
        self,
        messages,
        image_paths=None,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        **kwargs
    ) -> str:
        converted_messages = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, list):
                content = self._convert_content(content)
            converted_messages.append({"role": msg["role"], "content": content})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=converted_messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content
