"""On-disk index: one JSON with all segment metadata + one .npz of vectors per video.

    working_dir/
      index.json              {"videos": {name: {"path", "duration", "segments": [...]}}}
      vectors/<name>.npz      text (N, D), visual (N, D) — L2-normalised
      videos/<name>/frames/   sampled frames (reused by query-time re-captioning)
"""
import json
import os
import shutil
import threading
from dataclasses import asdict, dataclass, field

import numpy as np


@dataclass
class Segment:
    video: str
    index: int
    start: float
    end: float
    caption: str = ""
    transcript: str = ""
    ocr: str = ""                                   # on-screen text (signs, subtitles) read by OCR
    frames: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.video}#{self.index}"


class Store:
    VERSION = 2

    def __init__(self, working_dir: str):
        self.dir = os.path.abspath(working_dir)
        os.makedirs(os.path.join(self.dir, "vectors"), exist_ok=True)
        self._index_path = os.path.join(self.dir, "index.json")
        self._lock = threading.RLock()
        self.videos: dict[str, dict] = {}
        self._vectors: dict[str, dict[str, np.ndarray]] = {}
        self._load()

    # ------------------------------------------------------------------ io
    def _load(self):
        if os.path.exists(self._index_path):
            with open(self._index_path, encoding="utf-8") as f:
                data = json.load(f)
            if data.get("version") == self.VERSION:
                self.videos = data.get("videos", {})
        for name in list(self.videos):
            vec = os.path.join(self.dir, "vectors", f"{name}.npz")
            if os.path.exists(vec):
                with np.load(vec) as z:
                    self._vectors[name] = {k: z[k] for k in z.files}
            else:  # half-written entry
                self.videos.pop(name)

    def _save_index(self):
        tmp = self._index_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"version": self.VERSION, "videos": self.videos}, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self._index_path)

    def _rel(self, seg: dict) -> dict:
        # frames are stored relative to the working dir so the directory can be moved
        seg["frames"] = [os.path.relpath(f, self.dir) for f in seg["frames"]]
        return seg

    def video_dir(self, name: str) -> str:
        return os.path.join(self.dir, "videos", name)

    # ------------------------------------------------------------- mutation
    def add_video(self, name: str, path: str, duration: float, fingerprint: str,
                  segments: list[Segment], text_vecs: np.ndarray, visual_vecs: np.ndarray,
                  levels: list[list[dict]] | None = None):
        with self._lock:
            np.savez(os.path.join(self.dir, "vectors", f"{name}.npz"),
                     text=text_vecs.astype(np.float32), visual=visual_vecs.astype(np.float32))
            self._vectors[name] = {"text": text_vecs, "visual": visual_vecs}
            self.videos[name] = {
                "path": os.path.abspath(path),
                "duration": duration,
                "fingerprint": fingerprint,
                "segments": [self._rel(asdict(s)) for s in segments],
                "levels": [[{**c, "thumb": os.path.relpath(c["thumb"], self.dir) if c.get("thumb") else ""}
                            for c in lv] for lv in (levels or [])],  # chapter summaries: levels[0] = chapters, levels[1] = parts, ...
            }
            self._save_index()

    def set_levels(self, name: str, levels: list[list[dict]]):
        with self._lock:
            self.videos[name]["levels"] = [
                [{**c, "thumb": os.path.relpath(c["thumb"], self.dir) if c.get("thumb") else ""} for c in lv]
                for lv in levels]
            self._save_index()

    def remove_video(self, name: str):
        with self._lock:
            self.videos.pop(name, None)
            self._vectors.pop(name, None)
            self._save_index()
            vec = os.path.join(self.dir, "vectors", f"{name}.npz")
            if os.path.exists(vec):
                os.remove(vec)
            shutil.rmtree(self.video_dir(name), ignore_errors=True)

    # ---------------------------------------------------------------- query
    def find_by_fingerprint(self, fingerprint: str) -> str | None:
        return next((n for n, v in self.videos.items() if v.get("fingerprint") == fingerprint), None)

    def names(self) -> list[str]:
        return sorted(self.videos)

    def segments(self, names: list[str] | None = None) -> list[Segment]:
        out = []
        for name in names or self.names():
            for s in self.videos.get(name, {}).get("segments", []):
                out.append(Segment(**{**s, "frames": [os.path.join(self.dir, f) for f in s["frames"]]}))
        return out

    def vectors(self, names: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
        text, visual = [], []
        for name in names or self.names():
            if name in self._vectors:
                text.append(self._vectors[name]["text"])
                visual.append(self._vectors[name]["visual"])
        if not text:
            return np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32)
        return np.concatenate(text), np.concatenate(visual)

    def levels(self, name: str) -> list[list[dict]]:
        """Chapter summary levels with absolute thumb paths."""
        return [[{**c, "thumb": os.path.join(self.dir, c["thumb"]) if c.get("thumb") else ""} for c in lv]
                for lv in self.videos.get(name, {}).get("levels", [])]

    def video_path(self, name: str) -> str | None:
        return self.videos.get(name, {}).get("path")
