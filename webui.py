import os
import json
import logging
import warnings
import threading
import queue as _queue
import tempfile

import gradio as gr
from dotenv import load_dotenv

# 设置临时目录到项目目录而不是系统 /tmp
_script_dir = os.path.dirname(os.path.abspath(__file__))
_temp_dir = os.path.join(_script_dir, ".gradio_temp")
os.makedirs(_temp_dir, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = _temp_dir
tempfile.gettempdir = lambda: _temp_dir

load_dotenv()
warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

from VideoAgent import VideoRAG, QueryParam


# ---------- Global State ----------
_videorag: VideoRAG | None = None
_rag_lock = threading.Lock()


def _get_rag(working_dir: str) -> VideoRAG:
    global _videorag
    with _rag_lock:
        if _videorag is None or _videorag.working_dir != working_dir:
            _videorag = VideoRAG(working_dir=working_dir)

    print("working_dir:", working_dir)
    return _videorag


# ---------- Helpers ----------

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
        return "（暂无已索引视频）"
    return "\n".join(f"  • {v}" for v in sorted(videos))


# ---------- Log capture ----------

class _LogCapture(logging.Handler):
    """把 logger 输出转发到 queue，供 Gradio 生成器逐行 yield。"""

    def __init__(self):
        super().__init__()
        self.q: _queue.Queue[str] = _queue.Queue()

    def emit(self, record: logging.LogRecord):
        self.q.put(self.format(record))


# ---------- UI Callbacks ----------

def refresh_video_list(working_dir: str) -> str:
    return _fmt_video_list(_read_indexed_videos(working_dir))


def index_videos(video_files, working_dir: str, progress=gr.Progress()):
    """Generator — yields status strings so the textbox updates incrementally."""
    if not working_dir.strip():
        yield "❌ 错误：请填写工作目录路径。"
        return
    if not video_files:
        yield "❌ 错误：请上传至少一个视频文件。"
        return

    os.makedirs(working_dir, exist_ok=True)
    video_paths = [f.name for f in video_files]
    names = [os.path.basename(p) for p in video_paths]
    total_videos = len(names)

    progress(0, desc="初始化中...")
    yield f"📊 正在初始化 VideoRAG…\n📁 工作目录：{working_dir}\n⏳ 即将索引 {total_videos} 个视频\n"

    cap = _LogCapture()
    cap.setFormatter(logging.Formatter("%(levelname)s | %(message)s"))
    root_logger = logging.getLogger()
    root_logger.addHandler(cap)

    log_lines: list[str] = [
        f"📹 即将索引 {total_videos} 个视频：",
        *[f"   • {n}" for n in names],
        "",
    ]
    result: dict = {}

    def run():
        try:
            rag = _get_rag(working_dir)
            rag.insert_video(video_path_list=video_paths)
            result["ok"] = True
        except Exception as e:
            result["error"] = str(e)

    t = threading.Thread(target=run, daemon=True)
    t.start()

    # 线程运行期间持续转发日志
    progress(0.1, desc="处理中...")
    while t.is_alive() or not cap.q.empty():
        try:
            line = cap.q.get(timeout=0.3)
            log_lines.append(line)
            yield "\n".join(log_lines)
            progress(min(0.95, 0.1 + len(log_lines) * 0.02), desc="处理中...")
        except _queue.Empty:
            pass

    # 线程结束后彻底排空队列（防止最后一批日志丢失）
    while not cap.q.empty():
        log_lines.append(cap.q.get_nowait())
    root_logger.removeHandler(cap)

    if "error" in result:
        log_lines.append(f"\n❌ 索引失败：{result['error']}")
        progress(1, desc="失败")
    else:
        indexed = _read_indexed_videos(working_dir)
        log_lines.append(f"\n✅ 索引完成！共 {len(indexed)} 个视频已入库。")
        progress(1, desc="完成")

    yield "\n".join(log_lines)


def query_videos(query_text: str, working_dir: str, progress=gr.Progress()) -> str:
    if not query_text.strip():
        return "❌ 错误：请输入查询内容。"
    if not working_dir.strip():
        return "❌ 错误：请填写工作目录路径。"

    progress(0, desc="检查中...")
    indexed = _read_indexed_videos(working_dir)
    if not indexed:
        return "❌ 错误：当前工作目录中没有已索引的视频，请先在「视频索引」标签页完成索引。"

    progress(0.2, desc="加载模型中...")
    try:
        rag = _get_rag(working_dir)
        progress(0.4, desc="检索中...")
        param = QueryParam()
        progress(0.6, desc="生成回答中...")
        result = rag.query(query=query_text, param=param)
        progress(1, desc="完成")
        return result
    except Exception as e:
        progress(1, desc="失败")
        return f"❌ 查询出错：{e}"


# ---------- Build UI ----------

# 根据当前脚本的位置，设置 working_dir 在同级目录
_script_dir = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WORKING_DIR = os.path.join(_script_dir, "working_dir")

with gr.Blocks(title="VideoRAG WebUI", theme=gr.themes.Soft()) as demo:

    # 页面头部
    gr.Markdown(
        """
# 🎬 VideoAgent WebUI
## 基于 **Qwen3-VL** 的视频检索增强生成系统

---
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            working_dir_box = gr.Textbox(
                label="📁 工作目录 (Working Directory)",
                value=DEFAULT_WORKING_DIR,
                placeholder="用于存储索引和缓存的目录路径",
                info="⚠️ 请确保该目录有足够的磁盘空间用于存储视频索引"
            )

    with gr.Tabs():

        # ── Tab 1: Index ──────────────────────────────────────
        with gr.Tab("🎥 视频索引"):
            gr.Markdown("### 上传并索引视频文件")

            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("#### 📤 视频上传")
                    video_upload = gr.File(
                        label="选择视频文件（支持多选）",
                        file_count="multiple",
                        file_types=["video"],
                    )
                    gr.Markdown("> 💡 支持 MP4、MOV、AVI 等常见视频格式")

                    with gr.Row():
                        index_btn = gr.Button("▶️ 开始索引", variant="primary", scale=2)

                with gr.Column(scale=1):
                    gr.Markdown("#### ✅ 已索引视频")
                    indexed_list = gr.Textbox(
                        label="",
                        value=refresh_video_list(DEFAULT_WORKING_DIR),
                        interactive=False,
                        lines=10,
                    )
                    refresh_btn = gr.Button("🔄 刷新列表", variant="secondary", scale=1)

            gr.Markdown("---")
            gr.Markdown("#### 📋 索引日志")
            index_log = gr.Textbox(
                label="",
                interactive=False,
                lines=15,
                max_lines=200,
                autoscroll=True,
                placeholder="在此处查看实时处理日志...",
            )

            # 索引完成后自动刷新已索引列表
            index_btn.click(
                fn=index_videos,
                inputs=[video_upload, working_dir_box],
                outputs=[index_log],
            ).then(
                fn=refresh_video_list,
                inputs=[working_dir_box],
                outputs=[indexed_list],
            )

            refresh_btn.click(
                fn=refresh_video_list,
                inputs=[working_dir_box],
                outputs=[indexed_list],
            )

        # ── Tab 2: Query ──────────────────────────────────────
        with gr.Tab("🔍 视频查询"):
            gr.Markdown("### 根据自然语言查询视频内容")

            gr.Markdown("---")
            gr.Markdown("#### ❓ 输入查询")
            query_input = gr.Textbox(
                label="您的问题",
                placeholder="请输入您的问题，例如：\n• 视频中什么时候出现了主持人？\n• 哪个片段讲述了核心内容？",
                lines=4,
            )

            with gr.Row():
                query_btn = gr.Button("🚀 提交查询", variant="primary", scale=3)
                clear_btn = gr.ClearButton(components=[query_input], value="🗑️ 清空", scale=1)

            gr.Markdown("---")
            gr.Markdown("#### 💬 模型回答")
            response_box = gr.Textbox(
                label="",
                interactive=False,
                lines=18,
                max_lines=100,
                placeholder="模型的回答将显示在这里...",
            )

            query_fn_args = dict(
                fn=query_videos,
                inputs=[query_input, working_dir_box],
                outputs=[response_box],
            )
            query_btn.click(**query_fn_args)
            query_input.submit(**query_fn_args)  # 支持回车提交


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7869,
        share=False,
        show_error=True,
    )
