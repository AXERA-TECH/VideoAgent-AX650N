import os
from openai import OpenAI
from modelscope.hub.file_download import model_file_download
from .._utils import logger

class Qwen3:
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "Qwen3-4B-Instruct-2507")
        self.base_url = base_url or os.getenv("LLM_API_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key, timeout=250000.0)


    def generate_result(self, messages, max_new_tokens: int = 16384) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            extra_body={"enable_thinking": False},
        )
        return response.choices[0].message.content

    def generate_result_stream(self, messages, max_new_tokens: int = 32768):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            stream=True,
            extra_body={"enable_thinking": False},
        )
        for chunk in response:
            delta = chunk.choices[0].delta
            # 遍历 delta 所有属性，收集所有非空字符串字段
            text = None

            for field in ("content", "reasoning_content", "reasoning", "thinking", "text"):
                val = getattr(delta, field, None)
                
                if isinstance(val, str) and val:
                    text = val
                    break
            # 兜底：如果以上都没有，遍历所有属性找字符串
            if text is None:
                for attr in dir(delta):
                    if attr.startswith("_"):
                        continue
                    try:
                        val = getattr(delta, attr)
                        if isinstance(val, str) and val:
                            text = val
                            break
                    except Exception:
                        pass
            if text:
                yield text
            if chunk.choices[0].finish_reason:
                finish = chunk.choices[0].finish_reason
                if finish == "length":
                    yield "\n\n[⚠️ 输出因达到长度限制({} tokens)被截断]".format(max_new_tokens)
                break

    def generate_result_stream2(self, messages, max_new_tokens: int = 16384):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
           
            stream=True,
            
        )
        for ev in response:
            delta = getattr(ev.choices[0], "delta", None)
            # print(delta)
            if delta and getattr(delta, "content", None):
                print(delta.content, end="", flush=True)
        print(" ")



        

