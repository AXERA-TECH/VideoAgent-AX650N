import torch
import requests
import tempfile
import os
from typing import List, Dict, Any, Optional, Union
from PIL import Image
import logging
from dotenv import load_dotenv
load_dotenv()
logger = logging.getLogger(__name__)

class Qwen3VLEmbedder:
    """Qwen3VL Embedding API 客户端"""

    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        timeout: int = 300
    ):
        """
        初始化 API 客户端

        Args:
            base_url: API 服务地址
            timeout: 请求超时时间（秒）
        """
        # self.base_url = base_url
        self.timeout = timeout
        self.base_url = base_url or os.getenv("EMBEDDING_API_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("EMBEDDING_API_KEY", "EMPTY")

        

    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送 HTTP 请求"""
        url = f"{self.base_url}{endpoint}"
        kwargs['timeout'] = self.timeout

        try:
            if method == "POST":
                response = requests.post(url, **kwargs)
            elif method == "GET":
                response = requests.get(url, **kwargs)
            else:
                raise ValueError(f"Unsupported method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            raise RuntimeError(f"API request failed: {e}")

    def embed_text(self, text: str, normalize: bool = True) -> torch.Tensor:
        """为文本生成嵌入向量

        Args:
            text: 输入文本
            normalize: 是否进行 L2 归一化

        Returns:
            嵌入向量张量
        """
        data = {"text": text, "normalize": normalize}
        result = self._request("POST", "/embed/text", data=data)

        embeddings = torch.tensor(result["embeddings"], dtype=torch.float32)
        return embeddings

    def embed_image(self, image_path: str, normalize: bool = True) -> torch.Tensor:
        """为图像生成嵌入向量

        Args:
            image_path: 图像URL或本地路径
            normalize: 是否进行 L2 归一化

        Returns:
            嵌入向量张量
        """
        data = {"image_path": image_path, "normalize": normalize}
        result = self._request("POST", "/embed/image", data=data)

        embeddings = torch.tensor(result["embeddings"], dtype=torch.float32)
        return embeddings

    def embed_video(self, video_path: str, normalize: bool = True) -> torch.Tensor:
        """为视频生成嵌入向量

        Args:
            video_path: 视频本地路径
            normalize: 是否进行 L2 归一化

        Returns:
            嵌入向量张量
        """
        data = {"video_path": video_path, "normalize": normalize}
        result = self._request("POST", "/embed/video", data=data)

        embeddings = torch.tensor(result["embeddings"], dtype=torch.float32)
        return embeddings

    def embed_batch(self, documents: List[Dict[str, Any]], normalize: bool = True) -> torch.Tensor:
        """批量生成嵌入向量

        Args:
            documents: 文档列表，每个文档可包含 text, image, video 等
            normalize: 是否进行 L2 归一化

        Returns:
            嵌入向量张量 (N, embedding_dim)
        """
        # 处理文档中的PIL Image对象
        processed_documents = []
        temp_files = []

        try:
            for doc in documents:
                processed_doc = dict(doc)

                # 如果image是PIL Image对象，保存为临时文件
                if "image" in processed_doc and isinstance(processed_doc["image"], Image.Image):
                    temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                    temp_path = temp_file.name
                    temp_file.close()

                    processed_doc["image"].save(temp_path)
                    temp_files.append(temp_path)
                    processed_doc["image"] = temp_path

                # 如果video是列表（帧列表），转换为临时文件列表
                if "video" in processed_doc and isinstance(processed_doc["video"], list):
                    video_paths = []
                    for frame in processed_doc["video"]:
                        if isinstance(frame, Image.Image):
                            temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
                            temp_path = temp_file.name
                            temp_file.close()

                            frame.save(temp_path)
                            temp_files.append(temp_path)
                            video_paths.append(temp_path)
                        else:
                            video_paths.append(frame)

                    processed_doc["video"] = video_paths

                processed_documents.append(processed_doc)

            json_data = {"documents": processed_documents, "normalize": normalize}
            result = self._request("POST", "/embed/batch", json=json_data)

            embeddings = torch.tensor(result["embeddings"], dtype=torch.float32)
            return embeddings

        finally:
            # 清理临时文件
            for temp_path in temp_files:
                try:
                    os.remove(temp_path)
                except:
                    pass

    def embed_multimodal(
        self,
        document_or_text: Union[Dict[str, Any], List[Dict[str, Any]], str, None] = None,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video: Optional[str] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """生成多模态嵌入向量

        支持两种调用方式：
        1. embed_multimodal([{"text": "...", "image": "..."}]) - 传递document或documents列表
        2. embed_multimodal(text="...", image="...") - 使用关键字参数

        Args:
            document_or_text: 单个document dict、documents列表、或文本字符串
            text: 输入文本（当使用关键字参数方式时）
            image: 图像URL或本地路径
            video: 视频本地路径
            normalize: 是否进行 L2 归一化

        Returns:
            嵌入向量张量
        """
        # 如果传入了document或documents（作为第一个位置参数），则转发给embed_batch
        if document_or_text is not None:
            if isinstance(document_or_text, dict):
                # 单个document
                return self.embed_batch([document_or_text], normalize=normalize)
            elif isinstance(document_or_text, list):
                # 多个documents
                return self.embed_batch(document_or_text, normalize=normalize)
            elif isinstance(document_or_text, str):
                # 字符串被视为文本
                text = document_or_text

        # 使用关键字参数方式
        data = {
            "text": text,
            "image": image,
            "video": video,
            "normalize": normalize
        }

        json_data = {k: v for k, v in data.items() if v is not None}
        result = self._request("POST", "/embed/multimodal", json=json_data)

        embeddings = torch.tensor(result["embeddings"], dtype=torch.float32)
        return embeddings

    def compute_similarity(self, text1: str, text2: str) -> float:
        """计算两个文本的相似度

        Args:
            text1: 第一个文本
            text2: 第二个文本

        Returns:
            相似度分数（0-1）
        """
        data = {"text1": text1, "text2": text2}
        result = self._request("POST", "/similarity", data=data)

        return result["similarity"]

    def health_check(self) -> Dict[str, Any]:
        """检查服务健康状态

        Returns:
            健康检查结果
        """
        return self._request("GET", "/health")

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息

        Returns:
            模型信息
        """
        return self._request("GET", "/info")


if __name__ == "__main__":
    # 使用示例
    client = Qwen3VLEmbedder(base_url="http://localhost:8007")

    # 检查服务
    print("Health check:", client.health_check())
    print("Model info:", client.get_model_info())

    # 测试文本嵌入
    try:
        embeddings = client.embed_text("这是一个测试文本")
        print(f"Text embedding shape: {embeddings.shape}")
    except Exception as e:
        print(f"Error: {e}")
