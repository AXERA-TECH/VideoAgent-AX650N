"""Video indexing: segment -> ASR -> VLM caption -> text/visual embeddings.

The three model services usually sit on different NPUs, so the per-segment
stages run as a pipeline (ASR of segment i+1 overlaps the caption of i, which
overlaps the embedding of i-1). Frames per segment are derived from the VLM /
embedding prefill limits instead of being a fixed number.
"""
import base64
import hashlib
import logging
import math
import os
import re
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import media, prompts
from .clients import ContextOverflow, ModelError
from .store import Segment

log = logging.getLogger("videoagent")

_TRANSCRIPT_TOKENS = 200   # transcript share of the caption prompt
_THINK_RE = re.compile(r"<think>.*?</think>|</?think>", re.S)


_REPEAT_RE = re.compile(r"(.{4,160}?)\1{2,}$", re.S)


def repetition_cut(text: str) -> str | None:
    """If `text` ends with a phrase repeated 3+ times (greedy small models loop), drop the repeats."""
    m = _REPEAT_RE.search(text[-1200:])
    if not m:
        return None
    return text[: len(text) - len(m.group(0)) + len(m.group(1))].rstrip("，、, ") + "……"


def clean_text(text: str) -> str:
    text = _THINK_RE.sub("", text or "").strip()
    cut = repetition_cut(text)
    return cut if cut is not None else text


def segment_text(seg: Segment, caption: str | None = None) -> str:
    """How a segment is presented to the LLM / embedder: caption, on-screen text, speech."""
    parts = [f"画面：{caption if caption is not None else seg.caption or '（无）'}"]
    if seg.ocr:
        parts.append(f"文字：{seg.ocr}")
    parts.append(f"语音：{seg.transcript or '（无）'}")
    return "\n".join(parts)


def fingerprint(path: str) -> str:
    h = hashlib.md5()
    size = os.path.getsize(path)
    h.update(str(size).encode())
    with open(path, "rb") as f:
        h.update(f.read(1 << 20))
        if size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END)
            h.update(f.read())
    return h.hexdigest()


def _b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def _even_pick(items: list, n: int) -> list:
    if len(items) <= n:
        return list(items)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    return [items[i] for i in idx]


class Indexer:
    def __init__(self, agent):
        self.a = agent

    # ------------------------------------------------------------ budgets
    def frames_per_segment(self) -> int:
        """How many frames one caption request can carry within the VLM prefill limit."""
        s, vlm = self.a.settings, self.a.vlm
        budget = vlm.prompt_budget(s.caption_max_tokens, n_messages=1)
        text = vlm.count(prompts.caption_prompt("00:00:00", "00:00:00", 9, "")) + _TRANSCRIPT_TOKENS
        per_image = max(1, vlm.limits.image_tokens)
        n = (budget - text) // per_image
        if n < 1:
            raise ModelError(f"VLM prefill limit {vlm.limits.max_prefill} cannot fit even one image "
                             f"({per_image} tokens) plus the prompt")
        return int(min(n, s.max_frames_per_segment))

    def _segment_bounds(self, duration: float) -> list[tuple[float, float]]:
        L = self.a.settings.segment_seconds
        n = max(1, math.ceil(duration / L))
        bounds = [(i * L, min((i + 1) * L, duration)) for i in range(n)]
        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < L / 2:  # merge a short tail
            last = bounds.pop()
            bounds[-1] = (bounds[-1][0], last[1])
        return bounds

    def _unique_name(self, path: str) -> str:
        stem = os.path.splitext(os.path.basename(path))[0]
        stem = re.sub(r"[^\w一-鿿-]+", "_", stem).strip("_") or "video"
        name, i = stem, 2
        while name in self.a.store.videos:
            name, i = f"{stem}_{i}", i + 1
        return name

    # ------------------------------------------------------------- stages
    def _transcribe(self, audio, seg: Segment, tmp_dir: str) -> str:
        if audio is None or not self.a.perception.asr_enabled:
            return ""
        chunk = audio[int(seg.start * 16000): int(seg.end * 16000)]
        if chunk.size < 1600 or float(np.sqrt(np.mean(chunk ** 2))) < 1e-3:  # <0.1s or silence
            return ""
        wav = os.path.join(tmp_dir, f"{seg.index}.wav")
        media.write_wav(wav, chunk)
        try:
            text = self.a.perception.transcribe(wav)
            return "" if len(re.sub(r"[\W_]+", "", text)) < 2 else text  # "." etc. = no speech
        except Exception as e:
            log.warning("ASR failed for %s: %s", seg.key, e)
            return ""
        finally:
            os.remove(wav)

    def _read_text(self, seg: Segment) -> str:
        """OCR a few frames of the segment; unique lines in reading order."""
        if not seg.frames or not self.a.perception.ocr_enabled:
            return ""
        lines = []
        for f in _even_pick(seg.frames, self.a.settings.ocr_frames_per_segment):
            try:
                for t in self.a.perception.ocr(f):
                    text = re.sub(r"\s+", "", t.get("text", ""))
                    if t.get("score", 0) >= 0.6 and len(text) >= 2 and not any(text in x for x in lines):
                        lines = [x for x in lines if x not in text] + [text]
            except Exception as e:
                log.warning("OCR failed for %s: %s", seg.key, e)
        return self.a.counter.truncate("；".join(lines), 120)

    def _perceive(self, audio, seg: Segment, tmp_dir: str) -> Segment:
        seg.transcript = self._transcribe(audio, seg, tmp_dir)
        seg.ocr = self._read_text(seg)
        return seg

    def caption(self, seg: Segment, frames: list[str], query: str | None = None,
                max_tokens: int | None = None) -> str:
        """VLM description of a segment; drops frames if the prompt overflows."""
        vlm, counter = self.a.vlm, self.a.counter
        max_tokens = max_tokens or self.a.settings.caption_max_tokens
        transcript = counter.truncate(seg.transcript, _TRANSCRIPT_TOKENS)
        start, end = media.fmt_time(seg.start), media.fmt_time(seg.end)
        frames = list(frames)
        while True:
            text = (prompts.refine_prompt(start, end, len(frames), transcript, query, seg.ocr) if query
                    else prompts.caption_prompt(start, end, len(frames), transcript, seg.ocr))
            content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64_file(f)}"}}
                       for f in frames]
            content.append({"type": "text", "text": text})
            try:
                out = clean_text(vlm.generate([{"role": "user", "content": content}], max_tokens=max_tokens,
                                              stop_on_repeat=repetition_cut, frequency_penalty=0.5))
                if out:
                    return out
                raise ModelError("empty caption")
            except ContextOverflow:
                if len(frames) <= 1:
                    if transcript:
                        transcript = ""
                        continue
                    raise
                frames = _even_pick(frames, max(1, len(frames) // 2))
                log.warning("VLM prompt overflow on %s, retrying with %d frames", seg.key, len(frames))
            except ModelError as e:
                # small VLMs occasionally stop at the first token (axllm: "模型运行失败"): simpler prompt once
                if text == prompts.CAPTION_FALLBACK:
                    raise
                log.warning("VLM gave no caption for %s (%s), retrying with a plain prompt", seg.key, e)
                content[-1] = {"type": "text", "text": prompts.CAPTION_FALLBACK}
                out = clean_text(vlm.generate([{"role": "user", "content": content}], max_tokens=max_tokens,
                                              stop_on_repeat=repetition_cut, frequency_penalty=0.5))
                if not out:
                    raise ModelError(f"VLM returned no caption for {seg.key}")
                return out

    def _embed(self, seg: Segment) -> tuple[np.ndarray, np.ndarray]:
        emb = self.a.embedder
        if emb is None:
            z = np.zeros(1, np.float32)
            return z, z
        text = f"{seg.caption}\n{seg.ocr}\n{seg.transcript}".strip() or "(empty)"
        t = emb.embed_text(text, self.a.counter)
        frames = _even_pick(seg.frames, emb.max_images(len(seg.frames))) if seg.frames else []
        v = emb.embed_images([_b64_file(f) for f in frames]) if frames else t
        return t, v

    # ------------------------------------------------------------ chapters
    def _upper_levels(self, levels: list, chapter):
        """Add summary levels until the top one fits the answer window (long videos only)."""
        engine = self.a.engine
        budget = engine.answer_budget("")
        gb = engine.group_budget(self.a.settings.chapter_tokens)
        while engine._tokens(levels[-1]) > budget and len(levels[-1]) > 1 and len(levels) < 5:
            groups = engine.group(levels[-1], gb)
            if len(groups) >= len(levels[-1]):
                break
            levels.append([chapter(g, len(levels) + 1) for g in groups])

    def rebuild_chapters(self, name: str, progress=None) -> list[list[dict]]:
        """Recompute the chapter summaries of an indexed video (e.g. after changing prompts)."""
        from .query import Item

        s, engine, store = self.a.settings, self.a.engine, self.a.store
        items = [Item(x.video, x.start, x.end, segment_text(x),
                      thumb=x.frames[len(x.frames) // 2] if x.frames else "") for x in store.segments([name])]

        def chapter(group, level):
            it = engine.summarize(group, s.chapter_tokens)
            if progress:
                progress(level, f"L{level} {media.fmt_time(it.start)}-{media.fmt_time(it.end)}：{it.text}")
            return it

        levels = [[chapter(g, 1) for g in engine.group(items, engine.group_budget(s.chapter_tokens),
                                                       s.chapter_max_segments)]]
        self._upper_levels(levels, chapter)
        store.set_levels(name, [[{"start": c.start, "end": c.end, "text": c.text, "thumb": c.thumb} for c in lv]
                                for lv in levels])
        return store.levels(name)

    # --------------------------------------------------------------- main
    def index(self, path: str, progress=None) -> tuple[str, bool]:
        """Index one video. Returns (name, newly_indexed).

        progress(frac, msg, data=None, stage="") is called with stage
          "plan"      data = {"name", "duration", "bounds": [[start, end], ...]}
          "watching"  data = Segment about to be captioned (frames known)
          "segment"   data = finished Segment (caption + transcript)
          "chapter"   data = {"level", "start", "end", "text", "thumb"}
        """
        def report(frac, msg, data=None, stage=""):
            log.info("%s", msg)
            if progress:
                progress(frac, msg, data, stage)

        path = os.path.abspath(path)
        fp = fingerprint(path)
        existing = self.a.store.find_by_fingerprint(fp)
        if existing:
            report(1.0, f"「{existing}」已索引，跳过")
            return existing, False

        name = self._unique_name(path)
        info = media.probe(path)
        if info["duration"] <= 0:
            raise RuntimeError(f"cannot read video duration: {path}")
        bounds = self._segment_bounds(info["duration"])
        n_frames = self.frames_per_segment()
        fps = n_frames / self.a.settings.segment_seconds
        report(0.02, f"「{name}」时长 {media.fmt_time(info['duration'])}，切成 {len(bounds)} 段，"
                     f"每段 {n_frames} 帧（VLM prefill 上限 {self.a.vlm.limits.max_prefill}，"
                     f"每帧 {self.a.vlm.limits.image_tokens} token）",
               {"name": name, "duration": info["duration"], "bounds": bounds}, "plan")

        vdir = self.a.store.video_dir(name)
        try:
            return self._index(name, path, fp, info, bounds, n_frames, fps, vdir, report)
        except BaseException:
            shutil.rmtree(vdir, ignore_errors=True)
            raise

    def _index(self, name, path, fp, info, bounds, n_frames, fps, vdir, report):
        from .query import Item  # circular at import time

        s = self.a.settings
        engine = self.a.engine
        frames = media.extract_frames(path, os.path.join(vdir, "frames"), fps, s.frame_max_side)
        segments = [Segment(name, i, float(a), float(b)) for i, (a, b) in enumerate(bounds)]
        for ts, f in frames:
            i = min(int(ts // s.segment_seconds), len(segments) - 1)
            segments[i].frames.append(f)
        audio = None
        if self.a.perception.asr_enabled and info["has_audio"]:
            audio = media.extract_audio(path, os.path.join(vdir, "audio.pcm"))
        report(0.05, f"抽帧 {len(frames)} 张；语音识别{'开启' if audio is not None else '关闭/无音轨'}；"
                     f"文字识别（OCR）{'开启' if self.a.perception.ocr_enabled else '关闭'}")

        n = len(segments)
        text_vecs, vis_vecs = [None] * n, [None] * n
        chapter_budget = engine.group_budget(s.chapter_tokens)

        def item_of(seg):
            return Item(seg.video, seg.start, seg.end, segment_text(seg),
                        thumb=seg.frames[len(seg.frames) // 2] if seg.frames else "")

        def chapter(group, level, frac=0.98):
            it = engine.summarize(group, s.chapter_tokens)
            data = {"level": level, "start": it.start, "end": it.end, "text": it.text, "thumb": it.thumb}
            report(frac, f"章节摘要 L{level} {media.fmt_time(it.start)}-{media.fmt_time(it.end)}：{it.text[:40]}",
                   data, "chapter")
            return it

        with tempfile.TemporaryDirectory() as tmp, \
                ThreadPoolExecutor(1) as asr_pool, ThreadPoolExecutor(1) as vlm_pool, \
                ThreadPoolExecutor(1) as emb_pool, ThreadPoolExecutor(1) as llm_pool:
            def do_caption(seg, perceive_future):
                perceive_future.result()
                report(0.05 + 0.95 * seg.index / n,
                       f"正在观看 {media.fmt_time(seg.start)}-{media.fmt_time(seg.end)}", seg, "watching")
                seg.caption = self.caption(seg, seg.frames) if seg.frames else ""
                return seg

            asr_f = [asr_pool.submit(self._perceive, audio, x, tmp) for x in segments]
            cap_f = [vlm_pool.submit(do_caption, x, f) for x, f in zip(segments, asr_f)]
            emb_f = [emb_pool.submit(lambda f: self._embed(f.result()), f) for f in cap_f]
            chap_f, group, used = [], [], 0
            try:
                for i, f in enumerate(emb_f):
                    text_vecs[i], vis_vecs[i] = f.result()
                    seg = segments[i]
                    frac = 0.05 + 0.9 * (i + 1) / n
                    report(frac, f"[{i + 1}/{n}] {media.fmt_time(seg.start)}-{media.fmt_time(seg.end)} "
                                 f"{seg.caption[:60]}", seg, "segment")
                    # level-1 chapters are summarised on the LLM while the VLM keeps captioning
                    it = item_of(seg)
                    cost = engine.item_cost(it)
                    if group and (used + cost > chapter_budget or len(group) >= s.chapter_max_segments):
                        chap_f.append(llm_pool.submit(chapter, group, 1, frac))
                        group, used = [], 0
                    group.append(it)
                    used += cost
                if group:
                    chap_f.append(llm_pool.submit(chapter, group, 1, 0.96))
                levels = [[f.result() for f in chap_f]]
            except BaseException:
                for f in asr_f + cap_f + emb_f + chap_f:
                    f.cancel()
                raise

        self._upper_levels(levels, chapter)
        self.a.store.add_video(name, path, info["duration"], fp, segments,
                               np.stack(text_vecs), np.stack(vis_vecs),
                               levels=[[{"start": c.start, "end": c.end, "text": c.text, "thumb": c.thumb}
                                        for c in lv] for lv in levels])
        report(1.0, f"「{name}」索引完成：{n} 个片段，{len(levels[0])} 个章节", None, "done")
        return name, True
