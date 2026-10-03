"""All runtime settings, read once from the environment / .env."""
import os
from dataclasses import dataclass, field

try:  # optional: plain environment variables work too
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _env(key: str, default: str = "") -> str:
    val = os.getenv(key)
    return default if val is None or val.strip() == "" else val.strip().strip('"').strip("'")


def _env_int(key: str, default: int) -> int:
    try:
        return int(float(_env(key, str(default))))
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    try:
        return float(_env(key, str(default)))
    except ValueError:
        return default


@dataclass
class Endpoint:
    base_url: str
    model: str = ""          # empty -> first model reported by /v1/models
    api_key: str = "EMPTY"
    max_prefill: int = 0     # 0 -> auto-detect from the server
    max_context: int = 0     # 0 -> auto-detect from the server


@dataclass
class Settings:
    working_dir: str = field(default_factory=lambda: _env("VIDEOAGENT_WORKING_DIR", "./working_dir"))

    llm: Endpoint = field(default_factory=lambda: Endpoint(
        # single AX650 card: leave empty and the VLM also acts as the LLM (saves ~2.2 GB CMM)
        base_url=_env("LLM_API_BASE_URL") or _env("VLM_API_BASE_URL", "http://127.0.0.1:8011/v1"),
        model=_env("LLM_MODEL_NAME"),
        api_key=_env("LLM_API_KEY", "EMPTY"),
        max_prefill=_env_int("LLM_MAX_PREFILL_TOKENS", 0),
        max_context=_env_int("LLM_MAX_CONTEXT_TOKENS", 0),
    ))
    vlm: Endpoint = field(default_factory=lambda: Endpoint(
        base_url=_env("VLM_API_BASE_URL", "http://127.0.0.1:8011/v1"),
        model=_env("VLM_MODEL_NAME"),
        api_key=_env("VLM_API_KEY", "EMPTY"),
        max_prefill=_env_int("VLM_MAX_PREFILL_TOKENS", 0),
        max_context=_env_int("VLM_MAX_CONTEXT_TOKENS", 0),
    ))
    embedding: Endpoint = field(default_factory=lambda: Endpoint(
        # optional: empty -> no vector retrieval, questions are routed via chapter summaries
        base_url=_env("EMBEDDING_API_BASE_URL"),
        model=_env("EMBEDDING_MODEL_NAME"),
        api_key=_env("EMBEDDING_API_KEY", "EMPTY"),
        max_prefill=_env_int("EMBEDDING_MAX_PREFILL_TOKENS", 0),
        max_context=_env_int("EMBEDDING_MAX_CONTEXT_TOKENS", 0),
    ))
    # send enable_thinking=false to the LLM (Qwen3 hybrid-thinking models); never sent to the VLM
    llm_disable_thinking: bool = field(default_factory=lambda: _env("LLM_DISABLE_THINKING", "1") not in ("0", "false"))
    # servers/perception_server.py: SenseVoice ASR + PP-OCR. Empty -> index without speech / on-screen text.
    perception_url: str = field(default_factory=lambda: _env("PERCEPTION_API_BASE_URL",
                                                             _env("ASR_API_BASE_URL", _env("SHERPA_ASR_URL", ""))))

    # Optional HF tokenizer.json for exact token counting (needs `pip install tokenizers`).
    # Without it a calibrated estimate is used, which is fine thanks to overflow retries.
    tokenizer_json: str = field(default_factory=lambda: _env("TOKENIZER_JSON"))

    # --- indexing ---
    segment_seconds: int = field(default_factory=lambda: _env_int("VIDEOAGENT_SEGMENT_SECONDS", 10))
    max_frames_per_segment: int = field(default_factory=lambda: _env_int("VIDEOAGENT_MAX_FRAMES_PER_SEGMENT", 5))
    # frames are kept large enough for OCR to read shop signs / subtitles (the VLM resizes them itself)
    frame_max_side: int = field(default_factory=lambda: _env_int("VIDEOAGENT_FRAME_MAX_SIDE", 1280))
    ocr_frames_per_segment: int = field(default_factory=lambda: _env_int("VIDEOAGENT_OCR_FRAMES_PER_SEGMENT", 2))
    caption_max_tokens: int = field(default_factory=lambda: _env_int("VIDEOAGENT_CAPTION_MAX_TOKENS", 256))
    # chapter summaries built while indexing (answer "describe the whole video" without a query-time map-reduce)
    chapter_max_segments: int = field(default_factory=lambda: _env_int("VIDEOAGENT_CHAPTER_MAX_SEGMENTS", 6))
    chapter_tokens: int = field(default_factory=lambda: _env_int("VIDEOAGENT_CHAPTER_TOKENS", 200))

    # --- query ---
    top_k: int = field(default_factory=lambda: _env_int("VIDEOAGENT_TOP_K", 4))
    score_threshold: float = field(default_factory=lambda: _env_float("VIDEOAGENT_SCORE_THRESHOLD", 0.2))
    refine_top_n: int = field(default_factory=lambda: _env_int("VIDEOAGENT_REFINE_TOP_N", 1))
    answer_max_tokens: int = field(default_factory=lambda: _env_int("VIDEOAGENT_ANSWER_MAX_TOKENS", 512))

    request_timeout: float = field(default_factory=lambda: _env_float("VIDEOAGENT_REQUEST_TIMEOUT", 600))
