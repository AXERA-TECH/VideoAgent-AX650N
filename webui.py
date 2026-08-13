import hashlib
import json
import logging
import os
import queue as _queue
import re
import subprocess
import tempfile
import threading
import time
import warnings
import shutil
import atexit

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

# ---------- 基础环境配置 ----------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_temp_dir = os.path.join(_script_dir, ".gradio_temp")
os.makedirs(_temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _temp_dir
tempfile.gettempdir = lambda: _temp_dir

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from VideoAgent import QueryParam, VideoRAG
from VideoAgent.prompt import PROMPTS
from VideoAgent.query import _result_query_stream
from VideoAgent._utils import clean_output

# ---------- 并行处理补丁：解决 Spliting Video / Saving Video Segments 串行慢、CPU 利用率低的问题 ----------
from concurrent.futures import ProcessPoolExecutor, as_completed
import VideoAgent.vidrag_pipeline as _pipeline_mod
import VideoAgent._videoutil.split as _split_mod

_VIDEO_WORKERS = int(os.getenv("VIDEO_SPLIT_WORKERS", os.cpu_count() or 4))


def _extract_audio_seg(args):
    """Worker: 提取单个视频片段的音频（进程池调用）"""
    video_path, start, end, output_path = args
    from moviepy.video.io.VideoFileClip import VideoFileClip
    try:
        with VideoFileClip(video_path) as video:
            subvideo = video.subclip(start, end)
            subaudio = subvideo.audio
            subaudio.write_audiofile(
                output_path, codec='pcm_s16le', fps=16000,
                nbytes=2, verbose=False, logger=None,
            )
        return True
    except Exception:
        return False


def _save_video_seg(args):
    """Worker: 提取并编码单个视频片段（进程池调用）"""
    video_path, start, end, output_path = args
    from moviepy.video.io.VideoFileClip import VideoFileClip
    with VideoFileClip(video_path) as video:
        subvideo = video.subclip(start, end)
        subvideo.write_videofile(
            output_path, codec='libx264',
            ffmpeg_params=['-threads', '0'],
            verbose=False, logger=None,
        )
    return True


def _parallel_split_video(
    video_path, working_dir, segment_length,
    num_frames_per_segment, audio_output_format='mp3',
):
    """split_video 的并行版本：先算元数据，再用进程池并行提取音频"""
    import shutil as _shutil
    import numpy as _np
    from tqdm import tqdm as _tqdm
    from moviepy.video.io.VideoFileClip import VideoFileClip as _VideoFileClip

    unique_timestamp = str(int(time.time() * 1000))
    video_name = os.path.basename(video_path).split('.')[0]
    cache_dir = os.path.join(working_dir, '_cache', video_name)
    if os.path.exists(cache_dir):
        _shutil.rmtree(cache_dir)
    os.makedirs(cache_dir, exist_ok=False)

    segment_index2name, segment_times_info = {}, {}
    with _VideoFileClip(video_path) as video:
        total_length = int(video.duration)
        start_times = list(range(0, total_length, segment_length))
        if len(start_times) > 1 and (total_length - start_times[-1]) < 5:
            start_times = start_times[:-1]

        for idx, start in enumerate(start_times):
            end = (min(start + segment_length, total_length)
                   if start != start_times[-1] else total_length)
            frame_times = _np.linspace(0, end - start, num_frames_per_segment, endpoint=False)
            frame_times += start
            sid = str(idx)
            segment_index2name[sid] = f"{unique_timestamp}-{idx}-{start}-{end}"
            segment_times_info[sid] = {"frame_times": frame_times, "timestamp": (start, end)}

    # 并行提取音频
    tasks = []
    for idx in segment_index2name:
        s, e = segment_times_info[idx]["timestamp"]
        out = os.path.join(cache_dir, f'{segment_index2name[idx]}.{audio_output_format}')
        tasks.append((video_path, s, e, out))

    nw = min(_VIDEO_WORKERS, len(tasks)) if tasks else 1
    with ProcessPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(_extract_audio_seg, t): t for t in tasks}
        for f in _tqdm(as_completed(futs), total=len(futs), desc=f"Spliting Video {video_name}"):
            try:
                f.result()
            except Exception:
                pass

    return segment_index2name, segment_times_info


def _parallel_saving_video_segments(
    video_name, video_path, working_dir, segment_index2name,
    segment_times_info, error_queue, video_output_format='mp4',
):
    """saving_video_segments 的并行版本：进程池并行编码每个片段"""
    from tqdm import tqdm as _tqdm2

    try:
        cache_dir = os.path.join(working_dir, '_cache', video_name)
        tasks = []
        for idx in segment_index2name:
            s, e = segment_times_info[idx]["timestamp"]
            out = os.path.join(cache_dir, f'{segment_index2name[idx]}.{video_output_format}')
            tasks.append((video_path, s, e, out))

        nw = min(_VIDEO_WORKERS, len(tasks)) if tasks else 1
        with ProcessPoolExecutor(max_workers=nw) as ex:
            futs = {ex.submit(_save_video_seg, t): t for t in tasks}
            for f in _tqdm2(as_completed(futs), total=len(futs), desc=f"Saving Video Segments {video_name}"):
                try:
                    f.result()
                except Exception as e:
                    raise RuntimeError(f"Error in saving_video_segments: {str(e)}")
    except Exception as e:
        error_queue.put(f"Error in saving_video_segments:\n {str(e)}")
        raise RuntimeError


def _parallel_speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    """speech_to_text 的并行版本：线程池并行调用 ASR"""
    from concurrent.futures import ThreadPoolExecutor, as_completed as _ac
    from tqdm import tqdm as _tqdm3

    cache_dir = os.path.join(working_dir, '_cache', video_name)
    transcripts = {}

    def _transcribe_one(idx_name):
        idx, seg_name = idx_name
        audio_file = os.path.join(cache_dir, f"{seg_name}.{audio_output_format}")
        if not os.path.exists(audio_file):
            return idx, ""
        from VideoAgent._videoutil.asr import OnnxASRClient
        result = OnnxASRClient.transcribe(audio_file)
        if isinstance(result, tuple):
            text = result[0] if result else ""
        elif hasattr(result, 'text'):
            text = result.text
        else:
            text = str(result)
        return idx, text

    items = list(segment_index2name.items())
    nw = min(_VIDEO_WORKERS, len(items)) if items else 1
    with ThreadPoolExecutor(max_workers=nw) as ex:
        futs = {ex.submit(_transcribe_one, item): item for item in items}
        for f in _tqdm3(_ac(futs), total=len(futs), desc=f"Speech Recognition {video_name}"):
            idx, text = f.result()
            transcripts[idx] = text

    return transcripts


# 用并行版本替换 pipeline 模块中的串行实现
_pipeline_mod.split_video = _parallel_split_video
_pipeline_mod.saving_video_segments = _parallel_saving_video_segments
_pipeline_mod.speech_to_text = _parallel_speech_to_text
_split_mod.split_video = _parallel_split_video
_split_mod.saving_video_segments = _parallel_saving_video_segments

# 同样替换 asr 模块中的引用
import VideoAgent._videoutil.asr as _asr_mod
_asr_mod.speech_to_text = _parallel_speech_to_text

# ---------- 样式表 (CSS) ----------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --radius-sm: 6px;
    --radius-md: 8px;
    --radius-lg: 10px;
    --radius-xl: 12px;
    --color-bg: #f1f5f9;
    --color-surface: #ffffff;
    --color-border: #e2e8f0;
    --color-border-light: #f1f5f9;
    --color-primary: #6366f1;
    --color-primary-hover: #4f46e5;
    --color-text: #1e293b;
    --color-text-muted: #64748b;
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-lg: 0 4px 12px rgba(0,0,0,0.06), 0 2px 4px rgba(0,0,0,0.04);
}

.gradio-container {
    background: var(--color-bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--color-text) !important;
    max-width: 100% !important;
}

/* ---- 标题 ---- */
.app-title {
    text-align: center;
    padding: 6px 0 2px 0;
}
.app-title h1 {
    margin: 0;
    font-size: 18px;
    font-weight: 700;
    background: linear-gradient(135deg, #4338ca 0%, #6366f1 50%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -0.02em;
}

/* ---- Tabs 导航栏 ---- */
.tabs {
    gap: 0 !important;
}
.tabs > .tab-nav {
    gap: 4px !important;
    padding: 0 8px !important;
}
.tabs > .tab-nav > button {
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 9px 22px !important;
    border-radius: var(--radius-lg) var(--radius-lg) 0 0 !important;
    transition: all 0.2s ease !important;
    color: var(--color-text-muted) !important;
    background: transparent !important;
    border: none !important;
}
.tabs > .tab-nav > button.selected {
    color: var(--color-primary) !important;
    background: var(--color-surface) !important;
    box-shadow: 0 -1px 3px rgba(0,0,0,0.04) !important;
}

/* ---- 卡片 ---- */
.card-style {
    border-radius: var(--radius-lg) !important;
    border: 1px solid var(--color-border) !important;
    padding: 16px !important;
    background: var(--color-surface) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: box-shadow 0.2s ease !important;
    margin-bottom: 10px !important;
}
.card-style:hover {
    box-shadow: var(--shadow-lg) !important;
}

/* ---- 分区标题 ---- */
.section-label {
    font-weight: 600;
    font-size: 12px;
    color: var(--color-primary);
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    opacity: 0.85;
}
.section-label::before {
    content: '';
    display: inline-block;
    width: 3px;
    height: 14px;
    background: linear-gradient(180deg, #6366f1, #8b5cf6);
    border-radius: var(--radius-sm);
}

/* ---- 按钮 ---- */
.gradio-container .gr-button-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%) !important;
    border: none !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: var(--radius-md) !important;
    padding: 6px 18px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 3px rgba(79,70,229,0.25) !important;
}
.gradio-container .gr-button-primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 14px rgba(79,70,229,0.35) !important;
    filter: brightness(1.06);
}
.gradio-container .gr-button-secondary {
    border: 1px solid var(--color-border) !important;
    background: var(--color-surface) !important;
    color: #475569 !important;
    font-weight: 500 !important;
    font-size: 13px !important;
    border-radius: var(--radius-md) !important;
    padding: 6px 18px !important;
    transition: all 0.15s ease !important;
}
.gradio-container .gr-button-secondary:hover {
    background: #f8fafc !important;
    border-color: #94a3b8 !important;
    color: #334155 !important;
}

/* ---- 输入框全局 ---- */
.gradio-container input,
.gradio-container textarea {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--color-border) !important;
    font-size: 13px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    background: var(--color-surface) !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus {
    outline: none !important;
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
}

/* ---- 搜索面板 ---- */
.search-toolbar {
    padding: 14px 16px !important;
    margin-bottom: 8px;
}
.search-query textarea {
    font-size: 14px !important;
    line-height: 1.6 !important;
    min-height: 68px !important;
    border: 1px solid var(--color-border) !important;
    background: #fafbff !important;
    border-radius: var(--radius-lg) !important;
    padding: 12px 14px !important;
}
.search-query textarea:focus {
    background: var(--color-surface) !important;
    border-color: var(--color-primary) !important;
    box-shadow: 0 0 0 4px rgba(99,102,241,0.06) !important;
}
.search-actions {
    margin-top: 10px;
    justify-content: flex-end;
    gap: 10px;
}
.search-actions .gr-button {
    min-height: 44px !important;
    font-size: 15px !important;
    border-radius: var(--radius-lg) !important;
    min-width: 130px;
}
.search-panel {
    margin-top: 0;
    gap: 10px !important;
}

/* ---- 控制台输出框 (暗色主题) ---- */
.console-font,
.console-font > div {
    border-radius: var(--radius-lg) !important;
}
.console-font textarea {
    border-radius: var(--radius-lg) !important;
    overflow-y: auto !important;
}
.console-font textarea {
    font-family: 'JetBrains Mono', 'Fira Code', ui-monospace, SFMono-Regular, Menlo, Monaco, monospace !important;
    font-size: 12px !important;
    line-height: 1.5 !important;
    background: #0f172a !important;
    color: #e2e8f0 !important;
    border: 1px solid #1e293b !important;
    padding: 14px !important;
}
.console-font textarea:focus {
    border-color: #334155 !important;
    box-shadow: 0 0 0 2px rgba(99,102,241,0.15) !important;
    outline: none !important;
}

/* ---- 结果展示框 ---- */
.result-box {
    min-height: 360px;
    max-height: 360px;
    overflow: auto;
}
.result-box textarea {
    min-height: 360px !important;
    max-height: 360px !important;
}

/* ---- 视频预览 ---- */
.video-box {
    border-radius: var(--radius-lg) !important;
    overflow: hidden !important;
    border: 1px solid var(--color-border);
    box-shadow: var(--shadow-sm);
    background: #f8fafc !important;
}
.video-box video {
    border-radius: var(--radius-lg) !important;
}

/* ---- 画廊 ---- */
.clip-gallery {
    border: 1px solid var(--color-border);
    border-radius: var(--radius-lg);
    padding: 8px;
    background: var(--color-surface);
    box-shadow: var(--shadow-sm);
    min-height: 360px;
}
.clip-gallery .grid-wrap {
    gap: 8px !important;
}
.clip-gallery img,
.clip-gallery video {
    border-radius: var(--radius-md) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
.clip-gallery img:hover,
.clip-gallery video:hover {
    transform: scale(1.03);
    box-shadow: 0 4px 20px rgba(0,0,0,0.1) !important;
}

/* ---- 文件上传区域 ---- */
.gradio-container .file-preview {
    border-radius: var(--radius-md) !important;
    border: 2px dashed #cbd5e1 !important;
    background: #f8fafc !important;
    transition: all 0.2s ease !important;
    padding: 8px !important;
}
.gradio-container .file-preview:hover {
    border-color: var(--color-primary) !important;
    background: #fafbff !important;
}

/* ---- 设置面板 ---- */
.settings-group {
    margin-bottom: 20px;
}
.settings-section-title {
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #334155 !important;
    margin-bottom: 12px !important;
    padding-bottom: 8px !important;
    border-bottom: 2px solid var(--color-border);
}
.config-card {
    background: #f8fafc !important;
    border-radius: var(--radius-md) !important;
    padding: 14px !important;
    border: 1px solid var(--color-border) !important;
    margin-bottom: 12px;
}
.param-row {
    display: flex !important;
    gap: 14px !important;
    margin-bottom: 12px !important;
}
.param-col {
    flex: 1 !important;
    display: flex !important;
    flex-direction: column !important;
}
.param-label {
    font-size: 12px !important;
    font-weight: 500 !important;
    color: #475569 !important;
    margin-bottom: 4px !important;
}
.param-info {
    font-size: 11px !important;
    color: #94a3b8 !important;
    margin-top: 2px !important;
}
.apply-btn-container {
    text-align: center;
    margin-top: 20px;
}

/* ---- Accordion ---- */
.gradio-accordion {
    border-radius: var(--radius-md) !important;
    border: 1px solid var(--color-border) !important;
    margin-bottom: 6px !important;
    overflow: hidden !important;
    background: #ffffff !important;
}
.gradio-accordion:last-child {
    margin-bottom: 0 !important;
}
.gradio-accordion .label-wrap {
    padding: 9px 14px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #475569 !important;
    background: transparent !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
}
.gradio-accordion[open] > .label-wrap {
    border-radius: var(--radius-md) var(--radius-md) 0 0 !important;
    border-bottom: 1px solid var(--color-border) !important;
}

/* ---- Number Input ---- */
.gradio-container input[type="number"] {
    font-size: 13px !important;
    padding: 7px 10px !important;
    border-radius: var(--radius-md) !important;
}

/* ---- 滚动条 ---- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 10px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* ---- Footer ---- */
footer { display: none !important; }

/* ---- Tab 内容区 ---- */
.tabs > .tabitem {
    padding-top: 8px !important;
}

/* ---- 表单行间距 ---- */
.form {
    gap: 10px !important;
}
"""

# ---------- 全局状态控制 ----------
_videorag: VideoRAG | None = None
_rag_lock = threading.Lock()

# 添加清理缓存函数
def cleanup_temp_dir():
    """清理Gradio临时目录，但保留working_dir中的视频文件"""
    try:
        if os.path.exists(_temp_dir):
            # 只删除非working_dir的临时文件
            for item in os.listdir(_temp_dir):
                item_path = os.path.join(_temp_dir, item)
                # 跳过处理过的视频文件，只清理临时上传文件
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    # 递归删除子目录
                    shutil.rmtree(item_path)
            print(f"已清理Gradio临时目录: {_temp_dir}")
    except Exception as e:
        print(f"清理临时目录时出错: {e}")

# 注册退出时清理函数
atexit.register(cleanup_temp_dir)

_RAG_ENV_MAP = {
    "video_segment_length": "VIDEORAG_VIDEO_SEGMENT_LENGTH",
    "rough_num_frames_per_segment": "VIDEORAG_ROUGH_NUM_FRAMES_PER_SEGMENT",
    "retrieval_topk_chunks": "VIDEORAG_RETRIEVAL_TOPK_CHUNKS",
    "query_better_than_threshold": "VIDEORAG_QUERY_BETTER_THAN_THRESHOLD",
    "chunk_token_size": "VIDEORAG_CHUNK_TOKEN_SIZE",
    "segment_retrieval_top_k": "VIDEORAG_SEGMENT_RETRIEVAL_TOP_K",
}


def _read_int_env(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _read_float_env(key: str, default: float) -> float:
    try:
        return float(os.getenv(key, str(default)).strip())
    except Exception:
        return default


def _load_rag_runtime_settings() -> dict:
    return {
        "video_segment_length": _read_int_env(_RAG_ENV_MAP["video_segment_length"], 20),
        "rough_num_frames_per_segment": _read_int_env(_RAG_ENV_MAP["rough_num_frames_per_segment"], 10),
        "retrieval_topk_chunks": _read_int_env(_RAG_ENV_MAP["retrieval_topk_chunks"], 2),
        "query_better_than_threshold": _read_float_env(_RAG_ENV_MAP["query_better_than_threshold"], 0.2),
        "chunk_token_size": _read_int_env(_RAG_ENV_MAP["chunk_token_size"], 1000),
        "segment_retrieval_top_k": _read_int_env(_RAG_ENV_MAP["segment_retrieval_top_k"], 3),
    }


_rag_runtime_settings = _load_rag_runtime_settings()


def _get_rag(working_dir: str) -> VideoRAG:
    global _videorag
    with _rag_lock:
        need_rebuild = _videorag is None or _videorag.working_dir != working_dir
        if not need_rebuild and _videorag is not None:
            for k, v in _rag_runtime_settings.items():
                if getattr(_videorag, k, None) != v:
                    need_rebuild = True
                    break
        if need_rebuild:
            _videorag = VideoRAG(working_dir=working_dir, **_rag_runtime_settings)
    return _videorag


def _read_indexed_videos(working_dir: str) -> list[str]:
    kv_path = os.path.join(working_dir, "kv_store_video_path.json")
    if not os.path.exists(kv_path):
        return []
    try:
        with open(kv_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return list(data.keys())
    except Exception:
        return []


def _fmt_video_list(videos: list[str]) -> str:
    if not videos:
        return "📦 暂无已索引视频"
    return "\n".join(f"• {v}" for v in sorted(videos))


def _get_path_from_file(file_obj):
    if file_obj is None:
        return None
    if isinstance(file_obj, str):
        return file_obj
    if isinstance(file_obj, os.PathLike):
        return os.fspath(file_obj)
    if isinstance(file_obj, dict):
        for k in ("path", "name"):
            v = file_obj.get(k)
            if isinstance(v, str) and v.strip():
                return v
        return None
    for attr in ("path", "name"):
        v = getattr(file_obj, attr, None)
        if isinstance(v, str) and v.strip():
            return v
    return None


def _parse_clock_text(clock_text: str) -> float:
    t = clock_text.strip()
    parts = t.split(":")
    if len(parts) == 2:
        mm, ss = parts
        return int(mm) * 60 + int(ss)
    if len(parts) == 3:
        hh, mm, ss = parts
        return int(hh) * 3600 + int(mm) * 60 + int(ss)
    raise ValueError(f"无效时间格式: {clock_text}")


_TIME_TOKEN_RE = r"[0-9]{1,2}:[0-9]{1,2}(?::[0-9]{1,2})?"
_REF_LINE_RE = re.compile(
    rf"^\s*(?:[-*•]\s*)?(?:\[(?P<idx>\d+)\]\s*)?(?:\d+[.)、]\s*)?"
    rf"(?:(?:\*\*)?(?:reference|参考)(?:\*\*)?\s*[:：]\s*)?"
    rf"(?P<video>[^,，]+?)\s*[,，]\s*(?P<start>{_TIME_TOKEN_RE})\s*[,，]\s*(?P<end>{_TIME_TOKEN_RE})\s*$",
    flags=re.IGNORECASE,
)


def _is_reference_header(line: str) -> bool:
    s = line.strip()
    s = re.sub(r"\*\*", "", s)
    s = s.replace("：", ":")
    return bool(
        re.match(r"^#{1,6}\s*(reference|参考)\s*:?\s*$", s, flags=re.IGNORECASE)
        or re.match(r"^(reference|参考)\s*:?\s*$", s, flags=re.IGNORECASE)
    )


def _parse_reference_line(line: str):
    normalized = line.strip().replace("，", ",")
    if not normalized:
        return None

    # 兼容“说明文字 ... 参考：video, t1, t2”
    tail = re.search(r"(?:reference|参考)\s*[:：]\s*(.+)$", normalized, flags=re.IGNORECASE)
    if tail:
        normalized = tail.group(1).strip()

    idx_text = None
    m_idx = re.match(r"^\s*\[(\d+)\]\s*(.*)$", normalized)
    if m_idx:
        idx_text = m_idx.group(1)
        normalized = m_idx.group(2).strip()

    normalized = re.sub(r"^\s*[-*•]\s*", "", normalized)
    normalized = re.sub(r"^(?:\d+[.)、])\s*", "", normalized)
    normalized = re.sub(r"^(?:\*\*)?(?:reference|参考)(?:\*\*)?\s*[:：]\s*", "", normalized, flags=re.IGNORECASE)

    # 兼容: [sanguo, 0:0:20, 0:0:40] / - [sanguo, ...]
    if normalized.startswith("[") and normalized.endswith("]"):
        normalized = normalized[1:-1].strip()

    m = _REF_LINE_RE.match(normalized)
    if m:
        idx_text = idx_text or m.group("idx")
        video_name = m.group("video").strip().strip("`").strip('"').strip("'").strip("[]")
        start_text = m.group("start")
        end_text = m.group("end")
    else:
        # 兜底：按逗号切分，最后两段视为时间
        parts = [p.strip() for p in normalized.split(",") if p.strip()]
        if len(parts) < 3:
            return None
        video_name = ",".join(parts[:-2]).strip().strip("`").strip('"').strip("'").strip("[]")
        start_text, end_text = parts[-2], parts[-1]

    if not video_name:
        return None
    try:
        start = _parse_clock_text(start_text)
        end = _parse_clock_text(end_text)
    except Exception:
        return None
    if end <= start:
        end = start + 1

    return {
        "ref_id": int(idx_text) if idx_text else None,
        "video_name": video_name,
        "start": float(start),
        "end": float(end),
        "start_text": start_text,
        "end_text": end_text,
    }


def _dedup_and_fill_ref_id(items: list[dict]):
    uniq = []
    seen = set()
    for it in items:
        key = (it["video_name"].lower(), int(it["start"]), int(it["end"]))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(it)

    for i, it in enumerate(uniq, start=1):
        if it["ref_id"] is None:
            it["ref_id"] = i
    return sorted(uniq, key=lambda x: x["ref_id"])


def _extract_reference_items(answer: str):
    lines = answer.splitlines()
    in_ref = False
    from_ref_section = []
    from_all_lines = []

    for raw in lines:
        line = raw.strip()
        if _is_reference_header(line):
            in_ref = True
            continue

        if in_ref and re.match(r"^#{1,6}\s+\S+", line) and not _is_reference_header(line):
            in_ref = False

        parsed = _parse_reference_line(line)
        if not parsed:
            continue

        from_all_lines.append(parsed)
        if in_ref:
            from_ref_section.append(parsed)

    # 优先使用“Reference/参考”章节中的条目；没有再全局兜底
    if from_ref_section:
        return _dedup_and_fill_ref_id(from_ref_section)
    return _dedup_and_fill_ref_id(from_all_lines)


def _resolve_video_path(rag: VideoRAG, referenced_name: str):
    def _normalize_video_name(name: str) -> str:
        n = (name or "").strip().strip("`").strip('"').strip("'").strip("[]")
        n = n.replace("：", ":")
        # 兼容 "video_name: sanguo" / "video: xxx" / "视频名: xxx"
        n = re.sub(r"^(video_name|video|视频名|文件名|name)\s*:\s*", "", n, flags=re.IGNORECASE).strip()
        n = re.sub(r"^\s*[-*•]\s*", "", n)
        return os.path.splitext(n)[0]

    path_map = rag.video_path_db._data
    normalized = _normalize_video_name(referenced_name)
    if normalized in path_map:
        return normalized, path_map[normalized]

    raw = (referenced_name or "").strip()
    raw_wo_ext = _normalize_video_name(raw)
    for k, v in path_map.items():
        if k == raw_wo_ext:
            return k, v
        if k.lower() == raw.lower() or k.lower() == raw_wo_ext.lower():
            return k, v
        # 兜底：包含关系匹配，避免“video_name: xxx”这类前缀导致 miss
        if raw_wo_ext.lower() in k.lower() or k.lower() in raw_wo_ext.lower():
            return k, v
    return None, None


def _export_clip(video_path: str, start: float, end: float, working_dir: str, cache_key: str):
    clip_dir = os.path.join(working_dir, "_webui_query_clips")
    os.makedirs(clip_dir, exist_ok=True)
    video_stem = os.path.splitext(os.path.basename(video_path))[0]
    clip_name = f"{video_stem}_{int(start)}_{int(end)}_{cache_key[:8]}.mp4"
    clip_path = os.path.join(clip_dir, clip_name)

    if os.path.exists(clip_path):
        return clip_path

    # 使用与 split.py 相同的 ffmpeg 选择逻辑
    try:
        import imageio_ffmpeg
        _ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        _ffmpeg = os.getenv("FFMPEG_BIN", "ffmpeg")

    cmd = [
        _ffmpeg,
        "-y",
        "-ss",
        f"{start:.3f}",
        "-to",
        f"{end:.3f}",
        "-i",
        video_path,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        "-loglevel",
        "error",
        clip_path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or "ffmpeg 裁剪失败")
    return clip_path


def _upsert_env_file(env_path: str, updates: dict[str, str]):
    key_pattern = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=")
    lines = []
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

    found = set()
    new_lines = []
    for line in lines:
        m = key_pattern.match(line)
        if not m:
            new_lines.append(line)
            continue
        k = m.group(1)
        if k in updates:
            safe_v = str(updates[k]).replace('"', '\\"')
            new_lines.append(f'{k} = "{safe_v}"\n')
            found.add(k)
        else:
            new_lines.append(line)

    for k, v in updates.items():
        if k in found:
            continue
        safe_v = str(v).replace('"', '\\"')
        new_lines.append(f'{k} = "{safe_v}"\n')

    with open(env_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)


def apply_system_settings(
    llm_base_url: str,
    llm_api_key: str,
    llm_model_name: str,
    vlm_base_url: str,
    vlm_api_key: str,
    vlm_model_name: str,
    embedding_base_url: str,
    embedding_api_key: str,
    embedding_model_name: str,
    video_segment_length: float,
    rough_num_frames_per_segment: float,
    retrieval_topk_chunks: float,
    query_better_than_threshold: float,
    chunk_token_size: float,
    segment_retrieval_top_k: float,  # 新增参数
):
    global _rag_runtime_settings, _videorag

    llm_base_url = (llm_base_url or "").strip()
    llm_api_key = (llm_api_key or "").strip()
    llm_model_name = (llm_model_name or "").strip()
    vlm_base_url = (vlm_base_url or "").strip()
    vlm_api_key = (vlm_api_key or "").strip()
    vlm_model_name = (vlm_model_name or "").strip()
    embedding_base_url = (embedding_base_url or "").strip()
    embedding_api_key = (embedding_api_key or "").strip()
    embedding_model_name = (embedding_model_name or "").strip()

    required = {
        "LLM_API_BASE_URL": llm_base_url,
        "LLM_API_KEY": llm_api_key,
        "LLM_MODEL_NAME": llm_model_name,
        "VLM_API_BASE_URL": vlm_base_url,
        "VLM_API_KEY": vlm_api_key,
        "VLM_MODEL_NAME": vlm_model_name,
        "EMBEDDING_API_BASE_URL": embedding_base_url,
        "EMBEDDING_API_KEY": embedding_api_key,
        "EMBEDDING_MODEL_NAME": embedding_model_name,
    }
    empties = [k for k, v in required.items() if not v]
    if empties:
        return "❌ 以下配置不能为空：\n- " + "\n- ".join(empties)

    try:
        video_segment_length = int(video_segment_length)
        rough_num_frames_per_segment = int(rough_num_frames_per_segment)
        retrieval_topk_chunks = int(retrieval_topk_chunks)
        chunk_token_size = int(chunk_token_size)
        query_better_than_threshold = float(query_better_than_threshold)
        segment_retrieval_top_k = int(segment_retrieval_top_k)  # 新增转换
    except Exception as e:
        return f"❌ 参数类型错误，请检查数值配置：{e}"

    if video_segment_length <= 0 or rough_num_frames_per_segment <= 0:
        return "❌ video_segment_length 和 rough_num_frames_per_segment 必须 > 0"
    if retrieval_topk_chunks <= 0:
        return "❌ retrieval_topk_chunks 必须 > 0"
    if chunk_token_size <= 0:
        return "❌ chunk_token_size 必须 > 0"
    if segment_retrieval_top_k <= 0:  # 新增验证
        return "❌ segment_retrieval_top_k 必须 > 0"
    if not (0 <= query_better_than_threshold <= 1):
        return "❌ query_better_than_threshold 建议设置在 [0, 1] 区间"

    updates = {
        **required,
        _RAG_ENV_MAP["video_segment_length"]: str(video_segment_length),
        _RAG_ENV_MAP["rough_num_frames_per_segment"]: str(rough_num_frames_per_segment),
        _RAG_ENV_MAP["retrieval_topk_chunks"]: str(retrieval_topk_chunks),
        _RAG_ENV_MAP["query_better_than_threshold"]: str(query_better_than_threshold),
        _RAG_ENV_MAP["chunk_token_size"]: str(chunk_token_size),
        _RAG_ENV_MAP["segment_retrieval_top_k"]: str(segment_retrieval_top_k),  # 新增
    }
    for k, v in updates.items():
        os.environ[k] = v

    env_path = os.path.join(_script_dir, ".env")
    _upsert_env_file(env_path, updates)

    applied_logs = []
    try:
        import VideoAgent.query as query_mod

        if hasattr(query_mod, "qwen3_model"):
            query_mod.qwen3_model.model_name = llm_model_name
            query_mod.qwen3_model.base_url = llm_base_url
            query_mod.qwen3_model.api_key = llm_api_key
            query_mod.qwen3_model.client = OpenAI(base_url=llm_base_url, api_key=llm_api_key)
            applied_logs.append("✅ LLM 客户端已热更新")
        else:
            applied_logs.append("⚠️ LLM 客户端未找到，需重启生效")
    except Exception as e:
        applied_logs.append(f"⚠️ LLM 热更新失败，需重启生效：{e}")

    try:
        import VideoAgent._videoutil.caption as caption_mod

        if hasattr(caption_mod, "model"):
            caption_mod.model.model_name = vlm_model_name
            caption_mod.model.client = OpenAI(base_url=vlm_base_url, api_key=vlm_api_key)
            applied_logs.append("✅ VLM 客户端已热更新")
        else:
            applied_logs.append("⚠️ VLM 客户端未找到，需重启生效")
    except Exception as e:
        applied_logs.append(f"⚠️ VLM 热更新失败，需重启生效：{e}")

    try:
        from VideoAgent._llm import Qwen3VLEmbedderC
        import VideoAgent._videoutil.feature as feature_mod

        feature_mod.model = Qwen3VLEmbedderC(
            model_name=embedding_model_name,
            base_url=embedding_base_url,
            api_key=embedding_api_key,
        )
        applied_logs.append("✅ Embedding 客户端已热更新")
    except Exception as e:
        applied_logs.append(f"⚠️ Embedding 热更新失败，需重启生效：{e}")

    _rag_runtime_settings = {
        "video_segment_length": video_segment_length,
        "rough_num_frames_per_segment": rough_num_frames_per_segment,
        "retrieval_topk_chunks": retrieval_topk_chunks,
        "query_better_than_threshold": query_better_than_threshold,
        "chunk_token_size": chunk_token_size,
        "segment_retrieval_top_k": segment_retrieval_top_k,  # 新增
    }
    _videorag = None
    applied_logs.append("✅ VideoAgent 参数已更新（下次索引/查询将按新参数实例化）")

    return (
        "🎉 系统设置已保存到 .env 并应用。\n"
        + "\n".join(applied_logs)
        + "\n\n若你替换了模型结构差异较大的服务，建议重启 webui 以确保完全生效。"
    )


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.q: _queue.Queue[str] = _queue.Queue()

    def emit(self, record: logging.LogRecord):
        self.q.put(self.format(record))


# ---------- 核心交互函数 ----------
def refresh_video_list(working_dir: str) -> str:
    return _fmt_video_list(_read_indexed_videos(working_dir))


def index_videos(video_files, working_dir: str, progress=gr.Progress()):
    if not working_dir.strip():
        yield "❌ 错误：工作目录不能为空。", None
        return
    if not video_files:
        yield "❌ 错误：未选择视频文件。", None
        return

    os.makedirs(working_dir, exist_ok=True)
    
    # 创建processed目录
    processed_dir = os.path.join(working_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)
    
    # 检查processed目录中已存在的视频文件
    existing_videos = set()
    for file in os.listdir(processed_dir):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')):
            video_name_without_ext = os.path.splitext(file)[0]
            existing_videos.add(video_name_without_ext)
    
    # 处理视频文件并获取处理后的路径
    video_paths = []
    original_to_processed_map = {}  # 映射原始路径到处理后路径
    skipped_videos = []  # 记录跳过的视频
    
    for f in video_files:
        original_path = _get_path_from_file(f)
        if original_path and os.path.exists(original_path):
            # 获取视频名称（不含扩展名）
            original_video_name = os.path.basename(original_path)
            video_name_without_ext = os.path.splitext(original_video_name)[0]
            
            # 检查是否已存在同名视频
            if video_name_without_ext in existing_videos:
                skipped_videos.append(original_video_name)
                continue
            
            # 使用preprocess_video处理原始视频
            from VideoAgent._videoutil.split import preprocess_video
            video_output_path = preprocess_video(
                original_path,
                target_width=384,  # 使用与VideoRAG类相同的默认值
                target_height=384,
                target_fps=5,
                video_output_format="mp4"
            )
            
            # 将处理后的视频移动到working_dir/processed目录
            video_name = os.path.basename(video_output_path)
            target_path = os.path.join(processed_dir, video_name)
            
            # 如果目标文件已存在（理论上不应该发生），则添加时间戳避免覆盖
            if os.path.exists(target_path):
                name, ext = os.path.splitext(video_name)
                timestamp = int(time.time())
                target_path = os.path.join(processed_dir, f"{name}_{timestamp}{ext}")
            
            import shutil
            shutil.move(video_output_path, target_path)  # 使用move而不是copy，因为原文件不再需要
            video_paths.append(target_path)  # 现在使用处理后的路径调用insert_video
            original_to_processed_map[original_path] = target_path  # 保存映射关系
    
    if skipped_videos:
        log_lines = [f"⚠️ 以下视频已在processed目录中存在，跳过处理：{', '.join(skipped_videos)}"]
    else:
        log_lines = []
    
    if not video_paths:
        if skipped_videos:
            yield f"⚠️ 所有视频均已在processed目录中存在，无需重复处理。\n{chr(10).join(log_lines)}", None
        else:
            yield "❌ 错误：上传文件不可用，请重新上传后重试。", None
        return

    cap = _LogCapture()
    cap.setFormatter(logging.Formatter("%(asctime)s | %(message)s", "%H:%M:%S"))
    root_logger = logging.getLogger()
    root_logger.addHandler(cap)

    events: _queue.Queue = _queue.Queue()
    done_flag: dict[str, str | bool] = {}
    total = len(video_paths)

    def run():
        try:
            rag = _get_rag(working_dir)
            for i, path in enumerate(video_paths, start=1):
                events.put(("start", {"index": i, "path": path}))
                rag.insert_video(video_path_list=[path])
                
                # 获取video_name以便更新video_path_db
                video_name = os.path.basename(path).split('.')[0]
                
                # 使用always_get_an_event_loop获取事件循环并运行异步任务
                from VideoAgent._utils import always_get_an_event_loop
                loop = always_get_an_event_loop()
                
                async def update_video_path():
                    await rag.video_path_db.upsert({video_name: path})
                    await rag.video_path_db.index_done_callback()
                
                loop.run_until_complete(update_video_path())
                
                events.put(("done", {"index": i, "path": path}))
            done_flag["ok"] = True
        except Exception as e:
            done_flag["error"] = str(e)

    worker = threading.Thread(target=run, daemon=True)
    worker.start()

    current_video = video_paths[0] if video_paths else None
    log_lines.extend([f"🎬 准备索引 {total} 个视频..."])
    last_log_snapshot = ""
    last_video = None

    while worker.is_alive() or not events.empty() or not cap.q.empty():
        changed = False
        while True:
            try:
                event_type, value = events.get_nowait()
                if event_type == "start":
                    i = value["index"]
                    current_video = value["path"]
                    progress(float(i - 1) / total, desc=f"正在索引: {os.path.basename(current_video)}")
                    log_lines.append(f"🎞️ [{i}/{total}] 开始索引：{os.path.basename(current_video)}")
                elif event_type == "done":
                    i = value["index"]
                    current_video = value["path"]
                    progress(float(i) / total, desc=f"已完成: {os.path.basename(current_video)}")
                    log_lines.append(f"✅ [{i}/{total}] 完成索引：{os.path.basename(current_video)}")
                else:
                    log_lines.append(value)
                changed = True
            except _queue.Empty:
                break

        while True:
            try:
                log_line = cap.q.get_nowait()
                log_lines.append(log_line)
                changed = True
            except _queue.Empty:
                break

        snapshot = "\n".join(log_lines[-120:])
        if changed or snapshot != last_log_snapshot or current_video != last_video:
            last_log_snapshot = snapshot
            last_video = current_video
            yield snapshot, current_video
        else:
            time.sleep(0.2)

    root_logger.removeHandler(cap)
    if skipped_videos:
        final_log = f"{last_log_snapshot}\n🎉 完成索引。跳过的视频：{', '.join(skipped_videos)}"
    else:
        final_log = f"{last_log_snapshot}\n🎉 全部索引完成。"
        
    if "error" in done_flag:
        final_log = f"{last_log_snapshot}\n❌ 索引失败：{done_flag['error']}"
    yield final_log, current_video


def query_videos(query_text: str, working_dir: str):
    if not query_text.strip():
        yield "❌ 请输入问题", []
        return

    # Immediate first yield so Gradio registers the generator as started
    yield "⏳ 正在检索相关内容...", []

    try:
        rag = _get_rag(working_dir)
        qparam = QueryParam()
        qparam.naive_max_token_for_text_unit = int(
            getattr(rag, "chunk_token_size", qparam.naive_max_token_for_text_unit)
        )

        # Phase 1: retrieve context (blocking, can take a while)
        sys_prompt = rag._prepare_query_context(query=query_text, param=qparam)
        if sys_prompt == PROMPTS["fail_response"]:
            yield sys_prompt, []
            return

        # Phase 2: stream the LLM answer, filtering <think>...</think> blocks
        accumulated = ""
        try:
            for chunk in _result_query_stream(query_text, sys_prompt):
                accumulated += chunk
                display = clean_output(accumulated)
                if display:
                    yield display, []
        except Exception as stream_err:
            if accumulated:
                yield f"{clean_output(accumulated)}\n\n❌ 流式输出中断: {stream_err}", []
            else:
                yield f"❌ 生成失败: {stream_err}", []
            return

        answer = clean_output(accumulated)

        # --- Parse reference clips (after full answer is available) ---
        refs = _extract_reference_items(answer)
        if not refs:
            yield (
                f"{answer}\n\nℹ️ 未解析到可播放片段（请确保答案包含\"参考/Reference: 视频名, 开始时间, 结束时间\"格式）。",
                [],
            )
            return

        gallery_items = []
        warnings = []

        for ref in refs:
            resolved_name, video_path = _resolve_video_path(rag, ref["video_name"])
            if not video_path:
                warnings.append(f"[{ref['ref_id']}] 未匹配到视频：{ref['video_name']}")
                continue

            cache_key = hashlib.md5(
                f"{resolved_name}|{ref['start']}|{ref['end']}|{query_text}|{ref['ref_id']}".encode("utf-8")
            ).hexdigest()
            try:
                clip_path = _export_clip(
                    video_path=video_path,
                    start=ref["start"],
                    end=ref["end"],
                    working_dir=working_dir,
                    cache_key=cache_key,
                )
            except Exception as e:
                warnings.append(f"[{ref['ref_id']}] 裁剪失败：{e}")
                continue

            label = f"[{ref['ref_id']}] {resolved_name}  {ref['start_text']} - {ref['end_text']}"
            gallery_items.append((clip_path, label))

        if not gallery_items:
            warn_text = ("\n".join(f"- {w}" for w in warnings)) if warnings else "- 未生成任何可播放片段"
            yield (
                f"{answer}\n\n⚠️ 参考片段解析完成，但无法生成可播放视频：\n{warn_text}",
                [],
            )
            return

        if warnings:
            answer = answer + "\n\n⚠️ 部分片段处理失败：\n" + "\n".join(f"- {w}" for w in warnings[:5])
        yield answer, gallery_items
    except Exception as e:
        yield f"❌ 查询异常: {e}", []


# ---------- UI 界面构建 ----------
DEFAULT_WORKING_DIR = os.path.join(_script_dir, "working_dir")

with gr.Blocks(
    title="VideoRAG",
    theme=gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="indigo",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f1f5f9",
        body_background_fill_dark="#0f172a",
        block_background_fill="#ffffff",
        block_border_color="#e2e8f0",
        block_border_width="1px",
        block_radius="12px",
        button_primary_background_fill="linear-gradient(135deg, #4f46e5 0%, #6366f1 100%)",
        button_primary_background_fill_hover="linear-gradient(135deg, #4338ca 0%, #4f46e5 100%)",
        button_primary_border_color="transparent",
        button_primary_text_color="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_border_color="#cbd5e1",
        button_secondary_text_color="#475569",
        button_medium_radius="8px",
        input_background_fill="#ffffff",
        input_border_color="#e2e8f0",
        input_border_color_focus="#818cf8",
        input_radius="8px",
    ),
    css=custom_css,
) as demo:
    # gr.State 必须在 Blocks 上下文内创建，避免出现 KeyError: 0
    working_dir_state = gr.State(DEFAULT_WORKING_DIR)

    gr.HTML(
        """
        <div class="app-title">
            <h1>VideoAgent 控制台</h1>
        </div>
        """
    )

    with gr.Tabs():
        with gr.Tab("索引"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">上传视频</div>')
                        video_upload = gr.File(
                            file_count="multiple",
                            file_types=["video"],
                            type="filepath",
                            height=140,
                        )
                        with gr.Row():
                            index_btn = gr.Button("开始索引", variant="primary", size="sm")
                            refresh_btn = gr.Button("刷新", size="sm")
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">日志</div>')
                        index_log = gr.Textbox(
                            show_label=False,
                            interactive=False,
                            lines=16,
                            elem_classes="console-font",
                            placeholder="等待任务...",
                        )

                with gr.Column(scale=4):
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">当前视频</div>')
                        index_video_preview = gr.Video(
                            show_label=False,
                            height=240,
                            interactive=False,
                            elem_classes="video-box",
                        )
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">已索引</div>')
                        indexed_list = gr.Textbox(
                            show_label=False,
                            value=refresh_video_list(DEFAULT_WORKING_DIR),
                            lines=6,
                            max_lines=8,
                            interactive=False,
                        )

        with gr.Tab("检索"):
            with gr.Group(elem_classes=["card-style", "search-toolbar"]):
                query_input = gr.Textbox(
                    placeholder="输入检索问题",
                    show_label=False,
                    lines=2,
                    elem_classes="search-query",
                )
                with gr.Row(equal_height=True, elem_classes="search-actions"):
                    query_btn = gr.Button("开始检索", variant="primary", size="lg")
                    clear_query = gr.Button("清空", variant="secondary", size="lg")

            with gr.Row(equal_height=False, elem_classes="search-panel"):
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">检索结果</div>')
                        response_box = gr.Textbox(
                            show_label=False,
                            value="等待检索...",
                            lines=20,
                            max_lines=30,
                            interactive=False,
                            elem_classes="console-font result-box",
                        )
                with gr.Column(scale=6):
                    with gr.Group(elem_classes="card-style"):
                        gr.HTML('<div class="section-label">相关片段（点击播放）</div>')
                        query_clip_gallery = gr.Gallery(
                            show_label=False,
                            columns=2,
                            height=360,
                            interactive=False,
                            elem_classes="clip-gallery",
                        )

        with gr.Tab("设置"):
            with gr.Group(elem_classes="card-style"):
                gr.HTML('<div class="section-label">模型 API 配置</div>')
                with gr.Accordion("LLM 配置", open=False):
                    with gr.Row(equal_height=True):
                        llm_base_url_input = gr.Textbox(
                            label="Base URL",
                            value=os.getenv("LLM_API_BASE_URL", "http://localhost:8000/v1"),
                            placeholder="https://xxx/v1",
                        )
                        llm_api_key_input = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=os.getenv("LLM_API_KEY", ""),
                            placeholder="API Key",
                        )
                        llm_model_name_input = gr.Textbox(
                            label="模型名称",
                            value=os.getenv("LLM_MODEL_NAME", ""),
                            placeholder="model-name",
                        )
                
                with gr.Accordion("VLM 配置", open=False):
                    with gr.Row(equal_height=True):
                        vlm_base_url_input = gr.Textbox(
                            label="Base URL",
                            value=os.getenv("VLM_API_BASE_URL", "http://localhost:8012/v1"),  # 修改为正确的端口
                            placeholder="https://xxx/v1",
                        )
                        vlm_api_key_input = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=os.getenv("VLM_API_KEY", ""),
                            placeholder="API Key",
                        )
                        vlm_model_name_input = gr.Textbox(
                            label="模型名称",
                            value=os.getenv("VLM_MODEL_NAME", ""),
                            placeholder="model-name",
                        )
                
                with gr.Accordion("Embedding 配置", open=False):
                    with gr.Row(equal_height=True):
                        embedding_base_url_input = gr.Textbox(
                            label="Base URL",
                            value=os.getenv("EMBEDDING_API_BASE_URL", "http://localhost:8010/v1"),
                            placeholder="https://xxx/v1",
                        )
                        embedding_api_key_input = gr.Textbox(
                            label="API Key",
                            type="password",
                            value=os.getenv("EMBEDDING_API_KEY", ""),
                            placeholder="API Key",
                        )
                        embedding_model_name_input = gr.Textbox(
                            label="模型名称",
                            value=os.getenv("EMBEDDING_MODEL_NAME", ""),
                            placeholder="model-name",
                        )

            with gr.Group(elem_classes="card-style"):
                gr.HTML('<div class="section-label">VideoAgent 参数配置</div>')
                
                # 合并参数配置
                with gr.Row(elem_classes="param-row"):
                    with gr.Column(elem_classes="param-col"):
                        video_segment_length_input = gr.Number(
                            label="视频分段长度 (秒)",
                            value=_rag_runtime_settings["video_segment_length"],
                            minimum=1,
                            info="每个视频片段的持续时间",
                            precision=0,
                        )
                        retrieval_topk_chunks_input = gr.Number(
                            label="文本段检索 Top-K",
                            value=_rag_runtime_settings["retrieval_topk_chunks"],
                            minimum=1,
                            info="检索相关片段的数量",
                            precision=0,
                        )
                    with gr.Column(elem_classes="param-col"):
                        rough_num_frames_per_segment_input = gr.Number(
                            label="每段采样帧数",
                            value=_rag_runtime_settings["rough_num_frames_per_segment"],
                            minimum=1,
                            info="每段视频采样的帧数",
                            precision=0,
                        )
                        segment_retrieval_top_k_input = gr.Number(
                            label="视频段检索 Top-K",
                            value=_rag_runtime_settings["segment_retrieval_top_k"],
                            minimum=1,
                            info="检索相关视频段的数量",
                            precision=0,
                        )
                
                with gr.Row(elem_classes="param-row"):
                    with gr.Column(elem_classes="param-col"):
                        query_better_than_threshold_input = gr.Number(
                            label="查询阈值",
                            value=_rag_runtime_settings["query_better_than_threshold"],
                            minimum=0,
                            maximum=1,
                            info="查询匹配的最小阈值",
                            precision=3,
                        )
                    with gr.Column(elem_classes="param-col"):
                        chunk_token_size_input = gr.Number(
                            label="文本块最大Token数",
                            value=_rag_runtime_settings["chunk_token_size"],
                            minimum=1,
                            info="单个文本块的最大token数量",
                            precision=0,
                        )
                
                apply_settings_btn = gr.Button("保存设置", variant="primary", size="lg", elem_classes="gr-button-primary apply-btn-container")
                settings_status = gr.Textbox(
                    show_label=False,
                    lines=3,
                    max_lines=5,
                    interactive=False,
                    placeholder="等待保存...",
                    elem_classes="console-font"
                )

    # ---------- 事件流绑定 ----------
    index_btn.click(
        fn=index_videos,
        inputs=[video_upload, working_dir_state],
        outputs=[index_log, index_video_preview],
    ).then(
        fn=refresh_video_list,
        inputs=[working_dir_state],
        outputs=[indexed_list],
    )

    refresh_btn.click(
        fn=refresh_video_list,
        inputs=[working_dir_state],
        outputs=[indexed_list],
    )

    query_args = dict(
        fn=query_videos,
        inputs=[query_input, working_dir_state],
        outputs=[response_box, query_clip_gallery],
        stream_every=0.05,
    )
    query_btn.click(**query_args)
    query_input.submit(**query_args)

    clear_query.click(
        lambda: ("", "等待检索...", []),
        None,
        [query_input, response_box, query_clip_gallery],
    )

    apply_settings_btn.click(
        fn=apply_system_settings,
        inputs=[
            llm_base_url_input,
            llm_api_key_input,
            llm_model_name_input,
            vlm_base_url_input,
            vlm_api_key_input,
            vlm_model_name_input,
            embedding_base_url_input,
            embedding_api_key_input,
            embedding_model_name_input,
            video_segment_length_input,
            rough_num_frames_per_segment_input,
            retrieval_topk_chunks_input,
            query_better_than_threshold_input,
            chunk_token_size_input,
            segment_retrieval_top_k_input,  # 新增输入参数
        ],
        outputs=[settings_status],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7869, show_error=True)
