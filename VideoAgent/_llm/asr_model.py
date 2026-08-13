import re
import os
import json
import logging
import subprocess
import tempfile
import librosa
import numpy as np
import requests
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


@dataclass
class ASRResult:
    """ASR 识别结果"""
    text: str
    lang: str = ""
    emotion: str = ""
    event: str = ""
    timestamps: list = None


class ASRModel:
    """语音识别模型封装，调用 asr_server 的 /asr 接口"""

    def __init__(self, base_url: str = None, language: str = "auto", sample_rate: int = 16000):
        """
        初始化 ASR 模型

        Args:
            base_url: ASR 服务地址，如 "http://localhost:8000"
            language: 默认语言代码 (auto, zh, en, ja, ko, yue)
            sample_rate: 音频采样率，默认 16000 Hz
        """
        self.base_url = (base_url or os.getenv("ASR_API_BASE_URL", "http://localhost:8000")).rstrip('/')
        self.language = language or os.getenv("ASR_LANGUAGE", "auto")
        self.sample_rate = sample_rate
        self.asr_endpoint = f"{self.base_url}/asr"

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """
        加载音频文件并重采样到目标采样率

        Args:
            audio_path: 音频文件路径

        Returns:
            音频数组
        """
        try:
            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            return waveform
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise

    def transcribe(self, audio_file_path: str, language: Optional[str] = None) -> ASRResult:
        """
        转录音频文件

        Args:
            audio_file_path: 音频文件路径
            language: 语言代码，不提供则使用默认值

        Returns:
            ASRResult 对象，包含 text 字段
        """
        lang = language or self.language

        try:
            # 验证文件存在
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")

            # logger.info(f"Transcribing audio file: {audio_file_path}")

            # 加载音频文件
            waveform = self._load_audio(audio_file_path)

            # 转换为列表格式用于 API 调用
            audio_data = waveform.tolist()

            # logger.info(f"Calling ASR API: {self.asr_endpoint}")
            # logger.info(f"Audio length: {len(audio_data)}, Sample rate: {self.sample_rate}, Language: {lang}")

            # 调用 asr_server 的 /asr 接口
            response = requests.post(
                self.asr_endpoint,
                json={
                    "audio_data": audio_data,
                    "sample_rate": self.sample_rate,
                    "language": lang
                },
                timeout=300
            )

            if response.status_code != 200:
                error_detail = response.json().get("detail", response.text) if response.headers.get("content-type") == "application/json" else response.text
                logger.error(f"ASR API error: {response.status_code} - {error_detail}")
                raise RuntimeError(f"ASR API error: {response.status_code} - {error_detail}")

            result_data = response.json()
            text = result_data.get("text", "")

            # 清理文本中的特殊标记
            text = self._clean_text(text)

            return ASRResult(text=text)

        except FileNotFoundError as e:
            logger.error(f"File error: {e}")
            raise
        except requests.RequestException as e:
            logger.error(f"Failed to connect to ASR API: {e}")
            raise RuntimeError(f"Failed to connect to ASR API at {self.asr_endpoint}: {e}")
        except Exception as e:
            logger.error(f"ASR transcription error: {e}")
            raise

    @staticmethod
    def _clean_text(text: str) -> str:
        """清理 SenseVoice 返回的文本中的特殊标记"""
        # 移除所有 <|....|> 格式的标记
        text = re.sub(r'<\|[^|]*\|>', '', text)
        # 清理多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        return text



class SherpaOnnxASRClient:
    """调用 Sherpa-ONNX FastAPI 服务的客户端"""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or os.getenv("SHERPA_ASR_URL", "http://10.126.102.211:8016")).rstrip("/")

    def _parse_response(self, response) -> ASRResult:
        if response.status_code != 200:
            raise RuntimeError(f"Sherpa ASR error: {response.status_code} - {response.text}")
        data = response.json()
        # print("lang: ", data.get("lang", ""))
        return ASRResult(
            text=data.get("text", ""),
            lang=data.get("lang", ""),
            emotion=data.get("emotion", ""),
            event=data.get("event", ""),
            timestamps=data.get("timestamps", []),
        )

    def transcribe(self, audio_path: str) -> ASRResult:
        """通过 /asr/file 接口上传音频文件"""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        with open(audio_path, "rb") as f:
            response = requests.post(
                f"{self.base_url}/asr/file",
                files={"file": f},
                timeout=120,
            )
        return self._parse_response(response)

    def transcribe_audio_data(self, audio_data: list, sample_rate: int = 16000) -> ASRResult:
        """通过 /asr 接口发送音频数组数据"""
        response = requests.post(
            f"{self.base_url}/asr",
            json={"audio_data": audio_data, "sample_rate": sample_rate},
            timeout=120,
        )
        return self._parse_response(response)
