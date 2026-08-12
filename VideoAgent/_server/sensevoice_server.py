import os
import tempfile
import json
import logging
import re
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import time

from funasr import AutoModel
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(
    title="SenseVoice OpenAI Compatible API",
    description="A wrapper for SenseVoice model compatible with OpenAI Whisper API.",
    version="1.0.0"
)

logger.info("Loading SenseVoice model...")
try:
    model_path = os.getenv("ASR_MODEL_PATH", "")
    model = AutoModel(
        model=model_path,
        device="cuda",
        disable_update=True,
        trust_remote_code=True
    )
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

class OpenAIWhisperResponse(BaseModel):
    text: str
    language: str
    duration: float
    segments: Optional[List[Dict[str, Any]]] = None

def clean_text(text: str) -> str:
    """清理 SenseVoice 返回的文本中的特殊标记"""
    text = re.sub(r'<\|[^|]*\|>', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def format_segments(raw_result: Dict) -> List[Dict]:
    """将 SenseVoice 的输出转换为 OpenAI 的 segments 格式"""
    segments = []
    if isinstance(raw_result, list) and len(raw_result) > 0:
        res = raw_result[0]
    else:
        res = raw_result

    text = res.get("text", "")
    timestamps = res.get("timestamp", [])
    text = clean_text(text)

    if not timestamps:
        return []

    if len(timestamps) == 1:
        segments.append({
            "id": 0,
            "start": float(timestamps[0][0]),
            "end": float(timestamps[0][1]),
            "text": text,
            "avg_logprob": -0.5,
            "compression_ratio": 1.0,
            "no_speech_prob": 0.0
        })
    else:
        for i, ts in enumerate(timestamps):
            segments.append({
                "id": i,
                "start": float(ts[0]),
                "end": float(ts[1]),
                "text": text,
                "avg_logprob": -0.5,
                "compression_ratio": 1.0,
                "no_speech_prob": 0.0
            })

    return segments

@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(..., description="Audio file (wav, mp3, etc.)"),
    model_name: Optional[str] = Form(None),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    timestamp_granularities: Optional[str] = Form(None)
):
    if model_name is None:
        model_name = "sensevoice-small"

    if not file.filename.lower().endswith(('.wav', '.mp3', '.flac', '.m4a', '.ogg')):
        pass

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_path = tmp_file.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary file: {str(e)}")

    try:
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        lang_map = {
            "zh": "zh", "en": "en", "ja": "ja", "ko": "ko", 
            "chinese": "zh", "english": "en", "japanese": "ja", "korean": "ko",
            "auto": "auto", None: "auto"
        }
        sense_lang = lang_map.get(language, "auto")

        res = model.generate(
            input=tmp_path,
            language=sense_lang,
            use_itn=True,
            batch_size_s=60
        )

        if not res:
            raise HTTPException(status_code=400, detail="No result generated")

        result_data = res[0]
        full_text = result_data.get("text", "")

        full_text = clean_text(full_text)

        response_data = {
            "text": full_text,
            "language": result_data.get("lang", sense_lang),
            "duration": result_data.get("timestamp", [[0,0]])[0][1] if result_data.get("timestamp") else 0.0,
            "segments": format_segments(result_data),
            "model": "sensevoice-small",
            "task": "transcribe"
        }

        return JSONResponse(content=response_data)

    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/v1/audio/translations")
async def translate_audio(
    file: UploadFile = File(...),
    **kwargs
):
    """
    OpenAI 的 translate 接口是将任意语言翻译为英文。
    SenseVoice 主要是 ASR。如果需要翻译，通常需要级联一个翻译模型。
    此处暂时返回转录文本，并在 text 前标注 [Translation not supported natively, returning transcription]
    或者你可以集成一个 NMT 模型在此处调用。
    """
    raise HTTPException(status_code=501, detail="Translation task is not directly supported by SenseVoice alone. Please use transcription + separate translation model.")

@app.get("/health")
async def health_check():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("ASR_API_PORT", 8007))
    uvicorn.run(app, host="0.0.0.0", port=port)