"""Perception server for VideoAgent on AX650N: SenseVoice ASR + PP-OCRv6, via pyaxengine.

Runs on-chip (AX650 board) or on an AXCL card; the HTTP server is the standard library.
Either part can be disabled.

    python3 servers/perception_server.py --port 8013 \\
        --asr-dir /path/to/SenseVoice     # https://huggingface.co/AXERA-TECH/SenseVoice  (python/ + sensevoice_ax650/)
        --ocr-dir /path/to/PPOCR_v6       # https://huggingface.co/AXERA-TECH/PPOCR_v6    (ppocrv6_ax.py + axmodel/ax650/)
    AXCL_DEVICE_ID=0                      # AXCL card index (ignored on-chip)

API:
    POST /asr/file   audio (multipart field "file" or raw body, any format ffmpeg reads) -> {"text"}
    POST /ocr        image (multipart field "file" or raw body)                           -> {"texts": [{"text", "score", "box"}]}
    GET  /health     -> {"status", "asr", "ocr"}

Dependencies: numpy, pyaxengine; ASR: kaldi-native-fbank (+ ffmpeg for non-wav audio);
OCR: opencv-python, shapely, pyclipper, pyyaml.
"""
import argparse
import contextlib
import email.parser
import email.policy
import io
import json
import os
import re
import subprocess
import sys
import threading
import types
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

SR = 16000
_TAG_RE = re.compile(r"<\|[^|]*\|>")
_asr = None
_ocr = None
_npu_lock = threading.Lock()  # the NPU sessions are not thread-safe: serialise requests


def _select_axcl_device():
    """pyaxengine only honours the AXCL card index via provider_options (not the providers tuple)."""
    device = os.getenv("AXCL_DEVICE_ID")
    if device is None:
        return
    import axengine as axe
    orig = axe.InferenceSession
    if getattr(orig, "_va_patched", False):
        return

    def session(path, *a, **kw):
        kw.setdefault("provider_options", [{"device_id": int(device)}])
        return orig(path, *a, **kw)

    session._va_patched = True
    axe.InferenceSession = session


def load_asr(model_dir: str, language: str):
    # SenseVoiceAx.py imports torch/librosa only for streaming and file loading, neither is used here
    for name in ("torch", "librosa"):
        sys.modules.setdefault(name, types.ModuleType(name))
    sys.path.insert(0, os.path.join(model_dir, "python"))
    _select_axcl_device()
    from SenseVoiceAx import SenseVoiceAx

    root = os.path.join(model_dir, "sensevoice_ax650")
    with contextlib.redirect_stdout(sys.stderr):  # axengine prints to stdout
        model = SenseVoiceAx(os.path.join(root, "sensevoice.axmodel"), os.path.join(root, "am.mvn"),
                             os.path.join(root, "tokens.txt"),
                             os.path.join(root, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
                             max_seq_len=256, beam_size=1, streaming=False)
    model.default_language = language
    return model


def load_ocr(model_dir: str, npu: str):
    sys.path.insert(0, model_dir)
    _select_axcl_device()
    import ppocrv6_ax

    ax = os.path.join(model_dir, "axmodel", "ax650")
    with contextlib.redirect_stdout(sys.stderr):
        return ppocrv6_ax.PPOCRv6Onnx(
            det_onnx=os.path.join(ax, f"det_{npu}.axmodel"), rec_onnx=os.path.join(ax, f"rec_{npu}.axmodel"),
            char_dict=os.path.join(model_dir, "onnx", "rec_inference.yml"),
            det_db_box_thresh=0.45, drop_score=0.5)


def decode_audio(data: bytes) -> np.ndarray:
    """Audio bytes -> mono float32 @16 kHz. 16 kHz mono PCM wav is read directly, anything else via ffmpeg."""
    try:
        with wave.open(io.BytesIO(data)) as w:
            if w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2:
                return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0
    except (wave.Error, EOFError):
        pass
    p = subprocess.run([os.getenv("FFMPEG_BIN", "ffmpeg"), "-loglevel", "error", "-i", "pipe:0",
                        "-ac", "1", "-ar", str(SR), "-f", "s16le", "pipe:1"], input=data, capture_output=True)
    if p.returncode != 0:
        raise ValueError(f"cannot decode audio: {p.stderr.decode(errors='replace')[-300:]}")
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768.0


def transcribe(audio: np.ndarray) -> str:
    if audio.size < SR // 10:
        return ""
    with _npu_lock, contextlib.redirect_stdout(sys.stderr):
        text = _asr.infer_waveform(audio, _asr.default_language)
    return re.sub(r"\s+", " ", _TAG_RE.sub("", text)).strip()


def ocr(data: bytes) -> list[dict]:
    import cv2
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cannot decode image")
    with _npu_lock, contextlib.redirect_stdout(sys.stderr):
        res = _ocr(img)
    items = res[0] if isinstance(res, tuple) else res
    return [{"text": r["text"], "score": float(r["confidence"]), "box": r["box"]} for r in items or []]


class Handler(BaseHTTPRequestHandler):
    def _json(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _payload(self) -> bytes:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        ctype = self.headers.get("Content-Type", "")
        if not ctype.startswith("multipart/form-data"):
            return body
        msg = email.parser.BytesParser(policy=email.policy.HTTP).parsebytes(
            b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + body)
        for part in msg.iter_parts():
            if part.get_param("name", header="content-disposition") in ("file", "audio", "image"):
                return part.get_payload(decode=True)
        raise ValueError('multipart field "file" missing')

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self._json(200, {"status": "ok", "asr": _asr is not None, "ocr": _ocr is not None})
        else:
            self._json(404, {"detail": "not found"})

    def do_POST(self):
        path = self.path.rstrip("/")
        try:
            if path in ("/asr/file", "/asr"):
                if _asr is None:
                    return self._json(503, {"detail": "ASR not enabled (--asr-dir)"})
                return self._json(200, {"text": transcribe(decode_audio(self._payload()))})
            if path == "/ocr":
                if _ocr is None:
                    return self._json(503, {"detail": "OCR not enabled (--ocr-dir)"})
                return self._json(200, {"texts": ocr(self._payload())})
            self._json(404, {"detail": "not found"})
        except ValueError as e:
            self._json(400, {"detail": str(e)})
        except Exception as e:
            self._json(500, {"detail": f"{path} failed: {e}"})

    def log_message(self, fmt, *args):  # keep the console quiet
        pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("PERCEPTION_PORT", 8013)))
    ap.add_argument("--asr-dir", default=os.getenv("SENSEVOICE_DIR", ""), help="SenseVoice model dir ('' = off)")
    ap.add_argument("--ocr-dir", default=os.getenv("PPOCR_DIR", ""), help="PPOCR_v6 model dir ('' = off)")
    ap.add_argument("--language", default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    ap.add_argument("--ocr-npu", default="npu1", choices=["npu1", "npu3"],
                    help="npu1: 1 NPU core (leaves the others to the VLM on a shared chip); npu3: all cores")
    args = ap.parse_args()
    if not args.asr_dir and not args.ocr_dir:
        ap.error("enable at least one of --asr-dir / --ocr-dir")
    if args.asr_dir:
        _asr = load_asr(os.path.abspath(args.asr_dir), args.language)
    if args.ocr_dir:
        _ocr = load_ocr(os.path.abspath(args.ocr_dir), args.ocr_npu)
    print(f"perception server on http://{args.host}:{args.port}  asr={_asr is not None} ocr={_ocr is not None}",
          flush=True)
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
