import os
import re
import json
import tempfile
import logging
import subprocess
import soundfile as sf
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Body, UploadFile, File
from fastapi.responses import JSONResponse
import numpy as np
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# 全局 ASR 实例
asr_engine = None


class SherpaASREngine:
    """sherpa-onnx-offline 命令行引擎封装"""

    def __init__(
        self,
        model_dir: str = None,
        model_file: str = None,
        tokens_file: str = None,
        sherpa_bin: str = None,
        vad: str = None,
        provider: str = "axera",
    ):
        base = model_dir or os.getenv("SHERPA_MODEL_DIR", os.path.dirname(os.path.abspath(__file__)))
        self.model_file = model_file or os.getenv("SHERPA_MODEL_FILE", os.path.join(base, "ax650", "model-10-seconds.axmodel"))
        self.tokens_file = tokens_file or os.path.join(base, "tokens.txt")
        self.sherpa_bin = sherpa_bin or os.path.join(
            base,
            os.getenv("SHERPA_BIN_DIR", "sherpa-onnx-v1.12.20-axera-ax650-linux-aarch64-shared"),
            "bin",
            "sherpa-onnx-offline",
        )
        self.provider = provider or os.getenv("SHERPA_PROVIDER", "axera")
        # self.vad = vad or os.getenv("vad-model", "/root/huangjie/AXERA-TECH/SenseVoice/silero_vad.onnx")

        if os.path.exists(self.sherpa_bin):
            os.chmod(self.sherpa_bin, 0o755)

    def run(self, audio_path: str) -> dict:
        """执行识别命令，返回解析后的 JSON 结果"""
        cmd = [
            self.sherpa_bin,
            # f"--silero-vad-model={self.vad}",
            f"--sense-voice-model={self.model_file}",
            f"--tokens={self.tokens_file}",
            f"--provider={self.provider}",
            audio_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            logger.error(f"sherpa-onnx failed: {result.stderr}")
            raise RuntimeError(f"sherpa-onnx ASR failed: {result.stderr}")
        print("result: ", result)


        
        # 解析输出中的 JSON 行
        for line in reversed(result.stderr.strip().splitlines()):
            line = line.strip()
            if line.startswith("{"):
                # text=line.json().get("text", "")
                # lang=line.json().get("lang", "")
                print("lang: ", line)
                return json.loads(line)

        return {"text": "", "lang": "", "timestamps": []}


def clean_text(text: str) -> str:
    """清理文本中的特殊标记"""
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr_engine
    logger.info("Loading Sherpa-ONNX ASR engine...")
    try:
        asr_engine = SherpaASREngine()
        logger.info("Sherpa-ONNX ASR engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load Sherpa-ONNX ASR engine: {str(e)}")
        raise
    yield


app = FastAPI(title="Sherpa-ONNX ASR Server", description="SenseVoice ASR via sherpa-onnx-offline", lifespan=lifespan)



@app.post("/asr", summary="Recognize speech from raw audio data")
async def recognize_speech(
    audio_data: List[float] = Body(..., embed=True, description="Audio data as list of floats"),
    sample_rate: Optional[int] = Body(16000, description="Audio sample rate in Hz"),
):
    """接收 numpy 数组格式的音频数据并返回识别结果"""
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not loaded")

    try:
        np_audio = np.array(audio_data, dtype=np.float32)
        if np_audio.ndim != 1 or len(np_audio) == 0:
            raise HTTPException(status_code=400, detail="Audio data must be a non-empty 1D array")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            sf.write(tmp_path, np_audio, sample_rate)

        try:
            result = asr_engine.run(tmp_path)
            result["text"] = clean_text(result.get("text", ""))
            return JSONResponse(content=result)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/asr/file", summary="Recognize speech from uploaded audio file")
async def recognize_file(file: UploadFile = File(..., description="Audio file (wav, mp3, etc.)")):
    """接收音频文件并返回识别结果"""
    if asr_engine is None:
        raise HTTPException(status_code=503, detail="ASR engine not loaded")

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            content = await file.read()
            tmp.write(content)

        try:
            result = asr_engine.run(tmp_path)
            result["text"] = clean_text(result.get("text", ""))
            return JSONResponse(content=result)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": asr_engine is not None}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("SHERPA_ASR_API_PORT", 8006))
    uvicorn.run(app, host="0.0.0.0", port=port)
