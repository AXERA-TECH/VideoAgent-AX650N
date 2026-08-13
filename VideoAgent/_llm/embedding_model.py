import os
from openai import OpenAI
from modelscope.hub.file_download import model_file_download
from openai._types import NOT_GIVEN, NotGiven
from openai.types.chat import ChatCompletionMessageParam
from openai.types.create_embedding_response import CreateEmbeddingResponse

from typing import Literal


class Qwen3VLEmbedderC:
    def __init__(self, model_name: str = None, base_url: str = None, api_key: str = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL_NAME", "Qwen3-4B-Instruct-2507")
        self.base_url = base_url or os.getenv("EMBEDDING_API_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "EMPTY")
        self.client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def create_chat_embeddings(
        self,
        messages: list[ChatCompletionMessageParam],
        model: str,
        encoding_format: Literal["base64", "float"] | NotGiven = NOT_GIVEN,
        continue_final_message: bool = False,
        add_special_tokens: bool = False,
    ) -> CreateEmbeddingResponse:
        """
        Convenience function for accessing vLLM's Chat Embeddings API,
        which is an extension of OpenAI's existing Embeddings API.
        """
        return self.client.post(
            "/embeddings",
            cast_to=CreateEmbeddingResponse,
            body={
                "messages": messages,
                "model": model,
                "encoding_format": encoding_format,
                "continue_final_message": continue_final_message,
                "add_special_tokens": add_special_tokens,
            },
        )
    
    def embedding_gen(self, message_input):
        """
        Convenience function for accessing vLLM's Text Embeddings API,
        which is an extension of OpenAI's existing Embeddings API.
        """
        response =  self.create_chat_embeddings(
            messages = message_input,
            model=self.model_name,
            encoding_format="float",
            continue_final_message=False,
            add_special_tokens=True,
        )

        return response.data[0].embedding
    
    
    
    
    