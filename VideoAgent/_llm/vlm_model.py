import os
import base64
from io import BytesIO
from PIL import Image
from openai import OpenAI

from .._utils import _pil_to_base64



class Qwen3VLModel:
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None, **kwargs):
        self.model_name = model_name or os.getenv("VLM_MODEL_NAME", "Qwen3-VL-2B-Instruct")
        base_url = base_url or os.getenv("VLM_API_BASE_URL", "http://localhost:8001/v1")

        api_key = api_key or os.getenv("VLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=base_url, api_key=api_key)



    def generate_result(
        self,
        messages,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        **kwargs
    ) -> str:


        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        return response.choices[0].message.content

    def generate_result_stream(self, messages, max_new_tokens: int = 1024) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            stream=True,
        )
        result = ""
        for chunk in response:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                result += delta.content
        print()
        return result
