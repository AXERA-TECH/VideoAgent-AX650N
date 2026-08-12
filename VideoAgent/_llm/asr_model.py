from openai import OpenAI
import re
import os
from dotenv import load_dotenv
load_dotenv()

class ASRModel:
    """语音识别模型封装"""

    def __init__(self, base_url: str = None,
                 api_key: str = "not-needed",
                 model: str = "sensevoice-small"):
        """
        初始化 ASR 模型

        Args:
            base_url: API 服务地址
            api_key: API 密钥（本地服务不需要真实的 key）
            model: 使用的模型名称
        """

        self.base_url = base_url or os.getenv("ASR_API_BASE_URL", "http://localhost:8000/v1")
        self.api_key = api_key or os.getenv("ASR_API_KEY", "EMPTY")
        # self.base_url = base_url
        # self.api_key = api_key
        self.model = model or os.getenv("ASR_MODEL_NAME", "sensevoice-small")
        # self.model = model
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def transcribe(self, audio_file_path: str, language: str = "zh",
                   response_format: str = "json"):
        """
        转录音频文件

        Args:
            audio_file_path: 音频文件路径
            language: 语言代码 (zh, en, ja, ko, auto)
            response_format: 响应格式

        Returns:
            转录结果对象
        """
        with open(audio_file_path, "rb") as audio_file:
            transcription = self.client.audio.transcriptions.create(
                model=self.model,
                file=audio_file,
                language=language,
                response_format=response_format
            )

        # 清理返回的文本中的特殊标记（作为备份处理）
        if hasattr(transcription, 'text'):
            transcription.text = self._clean_text(transcription.text)

        return transcription

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理 SenseVoice 返回的文本中的特殊标记"""
        # 移除所有 <|....|> 格式的标记
        text = re.sub(r'<\|[^|]*\|>', '', text)

        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()

        return text


if __name__ == "__main__":
    # 使用示例
    asr = ASRModel()
    result = asr.transcribe("/data/huangjie/video_agent/videoRAG/black/videorag/mytools/testoutput/_cache/911/1770889667985-8-160-180.mp3", language="en")

    print("识别文本:", result.text)
    print("检测语言:", result.language)
    print("时间戳段:", result.segments)