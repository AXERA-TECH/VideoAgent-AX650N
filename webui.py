"""VideoAgent web UI — standard-library HTTP server + a static single-page app (web/).

    python3 webui.py [--port 7869] [--working-dir ./working_dir]

No web framework needed (runs on a stock AX650 board image). Progress of indexing and
answering is streamed to the page with Server-Sent Events.

API
    GET    /api/status                    services, detected limits
    GET    /api/videos                    indexed videos
    GET    /api/videos/<name>             segments + chapters of one video
    DELETE /api/videos/<name>
    POST   /api/upload?name=<file name>   raw file body -> {"job"}; indexing starts
    GET    /api/jobs                      ids of running indexing jobs
    GET    /api/jobs/<id>/events          SSE: indexing progress
    GET    /api/ask?q=..&videos=a,b       SSE: answer steps + streamed text
    GET    /media/thumb/<width>/<path>    cached preview of a frame inside the working dir
    GET    /media/video/<name>            source video (HTTP Range for seeking)
"""
import argparse
import json
import logging
import mimetypes
import os
import re
import threading
import time
import urllib.parse
import uuid
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from VideoAgent import VideoAgent

log = logging.getLogger("videoagent.web")
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")
AGENT: VideoAgent = None
JOBS: dict = {}
_index_lock = threading.Lock()  # one indexing job at a time (single NPU)


class Job:
    def __init__(self, path: str):
        self.id = uuid.uuid4().hex[:12]
        self.path = path
        self.events: list[dict] = []
        self.done = False
        self.cond = threading.Condition()

    def push(self, ev: dict):
        with self.cond:
            self.events.append(ev)
            self.cond.notify_all()

    def run(self):
        def progress(frac, msg, data=None, stage=""):
            ev = {"type": stage or "log", "frac": round(frac, 4), "msg": msg}
            if stage == "plan":
                ev.update(name=data["name"], duration=data["duration"], bounds=data["bounds"])
            elif stage in ("watching", "segment"):
                ev["segment"] = seg_json(data)
            elif stage == "chapter":
                ev["chapter"] = {**data, "thumb": frame_url(data.get("thumb", ""))}
            self.push(ev)

        try:
            self.push({"type": "log", "frac": 0, "msg": "排队等待中..." if _index_lock.locked() else "开始处理"})
            with _index_lock:
                name = AGENT.index(self.path, progress=progress)[0]
            self.push({"type": "finished", "frac": 1, "name": name, "msg": "完成"})
        except Exception as e:
            log.exception("indexing failed")
            self.push({"type": "error", "msg": f"索引失败：{e}"})
        finally:
            with self.cond:
                self.done = True
                self.cond.notify_all()


def frame_url(path: str, width: int = 320) -> str:
    """URL of a downscaled preview of a frame (frames are stored large for OCR)."""
    if not path:
        return ""
    rel = os.path.relpath(os.path.abspath(path), AGENT.store.dir)
    return f"/media/thumb/{width}/" + urllib.parse.quote(rel)


def seg_json(seg) -> dict:
    d = asdict(seg)
    d["frames"] = [frame_url(f, 640) for f in seg.frames]
    d["thumb"] = frame_url(seg.frames[len(seg.frames) // 2]) if seg.frames else ""
    return d


_thumb_lock = threading.Lock()


def thumb_file(rel: str, width: int) -> str | None:
    """Cached JPEG preview of a frame inside the working dir."""
    src = os.path.normpath(os.path.join(AGENT.store.dir, rel))
    if not src.startswith(AGENT.store.dir + os.sep) or not os.path.isfile(src):
        return None
    width = max(64, min(width, 1280))
    dst = os.path.join(AGENT.store.dir, ".thumbs", str(width), rel)
    if not os.path.exists(dst):
        from PIL import Image
        with _thumb_lock:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            im = Image.open(src)
            im.thumbnail((width, width))
            im.convert("RGB").save(dst + ".tmp", "JPEG", quality=80)
            os.replace(dst + ".tmp", dst)
    return dst


def video_summary(name: str) -> dict:
    v = AGENT.store.videos[name]
    segs = v["segments"]
    mid = segs[len(segs) // 2] if segs else None
    thumb = ""
    if mid and mid["frames"]:
        thumb = frame_url(os.path.join(AGENT.store.dir, mid["frames"][len(mid["frames"]) // 2]), 480)
    return {"name": name, "duration": v["duration"], "segments": len(segs),
            "chapters": len((v.get("levels") or [[]])[0]), "thumb": thumb,
            "has_ocr": any(s.get("ocr") for s in segs), "has_speech": any(s.get("transcript") for s in segs)}


def with_urls(item: dict) -> dict:
    return {**item, "thumb": frame_url(item["thumb"])} if item.get("thumb") else item


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # ------------------------------------------------------------ helpers
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code: int = 200):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode(), "application/json; charset=utf-8")

    def _sse_start(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

    def _sse(self, ev: dict):
        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
        self.wfile.flush()

    def _file(self, path: str, cache: bool = True):
        if not path or not os.path.isfile(path):
            return self._json({"detail": "not found"}, 404)
        size = os.path.getsize(path)
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        rng = re.match(r"bytes=(\d*)-(\d*)", self.headers.get("Range", ""))
        start, end = 0, size - 1
        partial = bool(rng and (rng.group(1) or rng.group(2)))
        if partial:
            if rng.group(1):
                start = int(rng.group(1))
                end = int(rng.group(2)) if rng.group(2) else size - 1
            else:  # suffix range
                start = max(0, size - int(rng.group(2)))
            end = min(end, size - 1)
        length = max(0, end - start + 1)
        self.send_response(206 if partial else 200)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.send_header("Cache-Control", "max-age=86400" if cache else "no-cache")
        self.end_headers()
        if self.command == "HEAD":
            return
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    def log_message(self, fmt, *args):
        pass

    # ------------------------------------------------------------- routes
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        path, qs = urllib.parse.unquote(url.path), urllib.parse.parse_qs(url.query)
        try:
            if path in ("/", "/index.html"):
                return self._file(os.path.join(WEB_DIR, "index.html"), cache=False)
            if path.startswith("/static/"):
                f = os.path.normpath(os.path.join(WEB_DIR, path[len("/static/"):]))
                return self._file(f, cache=False) if f.startswith(WEB_DIR) else self._json({}, 403)
            if path == "/api/jobs":  # running indexing jobs (lets a reloaded page reattach)
                return self._json([j.id for j in JOBS.values() if not j.done])
            if path == "/api/status":
                return self._json(AGENT.status())
            if path == "/api/videos":
                return self._json([video_summary(n) for n in AGENT.store.names()])
            if path.startswith("/api/videos/"):
                name = path[len("/api/videos/"):]
                if name not in AGENT.store.videos:
                    return self._json({"detail": "not found"}, 404)
                segs = [seg_json(x) for x in AGENT.store.segments([name])]
                levels = [[with_urls(c) for c in lv] for lv in AGENT.store.levels(name)]
                return self._json({**video_summary(name), "segments_list": segs, "levels": levels,
                                   "video_url": "/media/video/" + urllib.parse.quote(name)})
            m = re.match(r"^/api/jobs/([0-9a-f]+)/events$", path)
            if m:
                return self._job_events(m.group(1))
            if path == "/api/ask":
                videos = [v for v in qs.get("videos", [""])[0].split(",") if v]
                return self._ask(qs.get("q", [""])[0], videos or None)
            m = re.match(r"^/media/thumb/(\d+)/(.+)$", path)
            if m:
                return self._file(thumb_file(m.group(2), int(m.group(1))))
            if path.startswith("/media/video/"):
                return self._file(AGENT.store.video_path(path[len("/media/video/"):]))
            self._json({"detail": "not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_DELETE(self):
        path = urllib.parse.unquote(urllib.parse.urlparse(self.path).path)
        if path.startswith("/api/videos/"):
            AGENT.store.remove_video(path[len("/api/videos/"):])
            return self._json({"ok": True})
        self._json({"detail": "not found"}, 404)

    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        if url.path != "/api/upload":
            return self._json({"detail": "not found"}, 404)
        name = os.path.basename(urllib.parse.parse_qs(url.query).get("name", ["video.mp4"])[0]) or "video.mp4"
        name = re.sub(r"[^\w一-鿿.\-]+", "_", name)
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return self._json({"detail": "empty upload"}, 400)
        up_dir = os.path.join(AGENT.store.dir, "uploads")
        os.makedirs(up_dir, exist_ok=True)
        dest = os.path.join(up_dir, name)
        if os.path.exists(dest):
            stem, ext = os.path.splitext(name)
            dest = os.path.join(up_dir, f"{stem}_{int(time.time())}{ext}")
        left = length
        with open(dest, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left:
            os.remove(dest)
            return self._json({"detail": "upload interrupted"}, 400)
        job = Job(dest)
        JOBS[job.id] = job
        threading.Thread(target=job.run, daemon=True).start()
        self._json({"job": job.id})

    # ------------------------------------------------------------ streams
    def _job_events(self, job_id: str):
        job = JOBS.get(job_id)
        if job is None:
            return self._json({"detail": "unknown job"}, 404)
        self._sse_start()
        sent = 0
        try:
            while True:
                with job.cond:
                    if sent >= len(job.events) and not job.done:
                        job.cond.wait(timeout=15)
                    evs, done = job.events[sent:], job.done
                if not evs and not done:
                    self.wfile.write(b": ping\n\n")  # keep proxies / browsers from timing out
                    self.wfile.flush()
                for ev in evs:
                    self._sse(ev)
                sent += len(evs)
                if done and sent >= len(job.events):
                    self._sse({"type": "end"})
                    return
        except (BrokenPipeError, ConnectionResetError):
            return

    def _ask(self, q: str, videos):
        self._sse_start()
        try:
            for ev in AGENT.ask(q, videos):
                t = ev["type"]
                if t in ("hits", "context"):
                    ev = {**ev, "items": [with_urls(it) for it in ev["items"]]}
                elif t in ("refined", "condensed"):
                    ev = {**ev, "item": with_urls(ev["item"])}
                elif t == "done":
                    ev = {**ev, "refs": [{**with_urls(r), "video_url": "/media/video/" + urllib.parse.quote(r["video"])}
                                         for r in ev["refs"]]}
                self._sse(ev)
            self._sse({"type": "end"})
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            log.exception("ask failed")
            try:
                self._sse({"type": "error", "text": f"出错了：{e}"})
                self._sse({"type": "end"})
            except Exception:
                pass


def main():
    global AGENT
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=int(os.getenv("WEBUI_PORT", 7869)))
    ap.add_argument("--working-dir", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    AGENT = VideoAgent(args.working_dir)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"VideoAgent web UI: http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
