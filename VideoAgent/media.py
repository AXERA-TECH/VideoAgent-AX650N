"""ffmpeg helpers: probe, one-pass frame/audio extraction, clip export."""
import json
import os
import shutil
import subprocess

import numpy as np


def ffmpeg_bin() -> str:
    env = os.getenv("FFMPEG_BIN")
    if env:
        return env
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("ffmpeg not found: install it (apt install ffmpeg) or set FFMPEG_BIN")


def _run(cmd: list[str]):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{os.path.basename(cmd[0])} failed: {p.stderr.strip()[-800:]}")
    return p


def probe(path: str) -> dict:
    """-> {"duration": float, "has_audio": bool, "width": int, "height": int}"""
    ffprobe = ffmpeg_bin().replace("ffmpeg", "ffprobe")
    if shutil.which(ffprobe) or os.path.exists(ffprobe):
        out = _run([ffprobe, "-v", "error", "-show_entries",
                    "format=duration:stream=codec_type,width,height", "-of", "json", path]).stdout
        info = json.loads(out)
        streams = info.get("streams", [])
        video = next((s for s in streams if s.get("codec_type") == "video"), {})
        return {
            "duration": float(info.get("format", {}).get("duration") or 0),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "width": int(video.get("width") or 0),
            "height": int(video.get("height") or 0),
        }
    # imageio-ffmpeg ships no ffprobe: fall back to OpenCV
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"duration": n / fps, "has_audio": True, "width": w, "height": h}


def extract_frames(path: str, out_dir: str, fps: float, max_side: int) -> list[tuple[float, str]]:
    """Decode the video once and dump frames at `fps`. Returns [(timestamp, jpg_path)]."""
    os.makedirs(out_dir, exist_ok=True)
    scale = (f"scale='if(gt(iw,ih),min({max_side},iw),-2)':'if(gt(iw,ih),-2,min({max_side},ih))'")
    # start half an interval in: sample the middle of each 1/fps slot, not its (often black) edge
    offset = 0.5 / fps
    _run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{offset:.3f}", "-i", path,
          "-vf", f"fps={fps},{scale}", "-q:v", "3", os.path.join(out_dir, "%06d.jpg")])
    files = sorted(f for f in os.listdir(out_dir) if f.endswith(".jpg"))
    return [(offset + i / fps, os.path.join(out_dir, f)) for i, f in enumerate(files)]


def extract_audio(path: str, wav_path: str, sample_rate: int = 16000) -> np.ndarray | None:
    """Whole audio track as mono float32 @16k (None if the video has no audio)."""
    try:
        _run([ffmpeg_bin(), "-y", "-loglevel", "error", "-i", path, "-vn", "-ac", "1",
              "-ar", str(sample_rate), "-f", "s16le", wav_path])
    except RuntimeError:
        return None
    pcm = np.fromfile(wav_path, dtype=np.int16)
    os.remove(wav_path)
    return pcm.astype(np.float32) / 32768.0 if pcm.size else None


def write_wav(path: str, samples: np.ndarray, sample_rate: int = 16000):
    import wave
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def export_clip(path: str, start: float, end: float, out_path: str) -> str:
    if os.path.exists(out_path):
        return out_path
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    _run([ffmpeg_bin(), "-y", "-loglevel", "error", "-ss", f"{start:.3f}", "-to", f"{end:.3f}",
          "-i", path, "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
          "-movflags", "+faststart", out_path])
    return out_path


def fmt_time(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"
