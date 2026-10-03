"""Question answering over indexed videos, sized to the LLM's prefill window.

Strategy, decided per query from the measured token budget:
  full      – every segment fits: give the LLM the whole (chronological) index.
  retrieve  – specific question: hybrid text/visual retrieval, optionally re-watch
              the best segments with the VLM, pack as many hits as the budget allows.
  summarize – global question ("describe this video") or nothing retrieved, and the
              index is too large: map-reduce – condense consecutive segments into
              partial summaries that each fit the window, recursively, then answer.
Any context overflow reported by the server shrinks the budget and retries.
"""
import logging
import re
from dataclasses import dataclass

import numpy as np

from . import media, prompts
from .clients import ContextOverflow
from .indexer import clean_text, repetition_cut as _repetition_cut, segment_text as _segment_text
from .store import Segment

log = logging.getLogger("videoagent")

_CITE_RE = re.compile(r"\[(\d{1,3})\]")
_GLOBAL_RE = re.compile(
    r"描述|概括|总结|总述|介绍|讲(了|的)?(什么|啥)|说(了|的)?(什么|啥)|发生(了)?(什么|啥)|主要内容|大意|梗概|"
    r"整体|整个|全部|全片|summar|overview|describe|what happens", re.I)
_FILLER_RE = re.compile(
    r"用?(简体|繁体)?(中文|英文|汉语)|请|帮我|给我|一下|这段|这个|这部|该|视频|画面|影片|片段|内容|场景|镜头|"
    r"里面?|中|的|了|吗|呢|吧|一|下|[，。？！?!,.\s]|"
    r"描述|概括|总结|总述|介绍|讲|说|什么|啥|发生|主要|大意|梗概|整体|整个|全部|全片", re.I)


_QFILL_RE = re.compile(
    r"请问|请|帮我|给我|一下|[哪那这]一?段|[哪那这]个|哪些|哪里|哪儿|什么时候|什么|何时|有没有|是否|是不是|出现|"
    r"视频|画面|片段|镜头|场景|情况|内容|里面?|的|了|吗|呢|吧|有|在|是|[，。？！?!,.\s“”\"'：:、（）()]")


def query_terms(query: str) -> list[str]:
    """Content terms of a question: CJK bigrams of what remains after question words, plus latin words."""
    terms = []
    for chunk in _QFILL_RE.sub(" ", query).split():
        for run in re.findall(r"[\u4e00-\u9fff]+", chunk):
            terms += [run] if len(run) <= 2 else [run[i:i + 2] for i in range(len(run) - 1)]
        terms += [w.lower() for w in re.findall(r"[A-Za-z0-9]{2,}", chunk)]
    return list(dict.fromkeys(t for t in terms if len(t) >= 2))


def lexical_scores(query: str, segs: list[Segment]) -> list[float]:
    """Share of the question's terms found in each segment (on-screen text counts most)."""
    terms = query_terms(query)
    if not terms:
        return [0.0] * len(segs)
    out = []
    for x in segs:
        ocr, other = x.ocr.lower(), f"{x.caption}\n{x.transcript}".lower()
        hit = sum(1.0 if t in ocr or t in other else 0.0 for t in terms)
        bonus = 0.1 if any(t in ocr for t in terms) else 0.0
        out.append(min(1.0, hit / len(terms) + bonus))
    return out


def is_global_question(query: str) -> bool:
    """'描述这段画面' / '视频讲了什么' -> True; '张飞出场时穿什么' -> False."""
    return bool(_GLOBAL_RE.search(query)) and len(_FILLER_RE.sub("", query)) <= 3


@dataclass
class Item:
    """One numbered entry of the answer context (a segment or a condensed span)."""
    video: str
    start: float
    end: float
    text: str
    score: float = 0.0
    thumb: str = ""   # a frame path, for the UI

    def as_dict(self, n: int | None = None) -> dict:
        return {"n": n, "video": self.video, "start": self.start, "end": self.end,
                "text": self.text, "score": round(self.score, 3), "thumb": self.thumb}

    def render(self, n: int) -> str:
        return f"[{n}] 视频「{self.video}」{media.fmt_time(self.start)}-{media.fmt_time(self.end)}\n{self.text}"


def _thumb(seg: Segment) -> str:
    return seg.frames[len(seg.frames) // 2] if seg.frames else ""


class QueryEngine:
    def __init__(self, agent):
        self.a = agent

    # ------------------------------------------------------------ helpers
    def _tokens(self, items: list[Item]) -> int:
        return sum(self.a.llm.count(it.render(99)) + 2 for it in items)

    def answer_budget(self, query: str) -> int:
        llm = self.a.llm
        fixed = llm.count(prompts.ANSWER_SYSTEM.format(context="")) + llm.count(query) + 120
        return llm.prompt_budget(self.a.settings.answer_max_tokens) - fixed

    def _route(self, query: str, names: list[str], segs: list[Segment]):
        """No embedding model: let the LLM pick the relevant chapters, score their segments."""
        chapters = []
        for n in names:
            lv = self.a.store.levels(n)
            if lv:
                chapters += [(n, c) for c in lv[0]]
            else:  # index without chapters: every segment is its own "chapter"
                chapters += [(n, {"start": x.start, "end": x.end, "text": x.caption})
                             for x in segs if x.video == n]
        llm = self.a.llm
        lines, budget = [], llm.prompt_budget(16) - llm.count(prompts.ROUTE_SYSTEM + query) - 40
        for i, (n, c) in enumerate(chapters, 1):
            line = f"[{i}] 「{n}」{media.fmt_time(c['start'])}-{media.fmt_time(c['end'])}：{c['text']}"
            budget -= llm.count(line)
            if budget < 0:
                break
            lines.append(line)
        msgs = [{"role": "system", "content": prompts.ROUTE_SYSTEM},
                {"role": "user", "content": "\n".join(lines) + f"\n\n问题：{query}"}]
        picked = []
        try:
            for m in re.findall(r"\d+", llm.chat(msgs, max_tokens=16)):
                if 1 <= int(m) <= len(lines) and int(m) - 1 not in picked:
                    picked.append(int(m) - 1)
        except Exception as e:
            log.warning("chapter routing failed: %s", e)
        score = {}
        for rank, ci in enumerate(picked[:3]):
            n, c = chapters[ci]
            for x in segs:
                if x.video == n and c["start"] <= x.start < c["end"]:
                    score.setdefault(x.key, 1.0 - 0.1 * rank)
        ranked = sorted(((x, score.get(x.key, 0.0)) for x in segs), key=lambda t: (-t[1], t[0].video, t[0].start))
        return ranked

    def _rank(self, query: str, names: list[str], segs: list[Segment]):
        """Hybrid ranking: reciprocal-rank fusion of text and visual cosine similarity."""
        if self.a.embedder is None:
            return self._route(query, names, segs)
        q = self.a.embedder.embed_text(query, self.a.counter)
        T, V = self.a.store.vectors(names)
        t, v = T @ q, V @ q
        k = 10.0
        rrf = 1 / (k + np.argsort(np.argsort(-t))) + 1 / (k + np.argsort(np.argsort(-v)))
        best = np.maximum(t, v)
        order = np.argsort(-rrf)
        return [(segs[i], float(best[i])) for i in order]

    def _pack(self, items: list[Item], budget: int) -> list[Item]:
        """Keep the highest-scored items that fit, trimming a lone oversize item."""
        out, used = [], 0
        for it in sorted(items, key=lambda x: -x.score):
            cost = self.a.llm.count(it.render(99)) + 2
            if used + cost > budget:
                if not out:  # the best item alone is too big: truncate it
                    it = Item(it.video, it.start, it.end,
                              self.a.counter.truncate(it.text, budget - 40), it.score)
                    out.append(it)
                continue
            out.append(it)
            used += cost
        return sorted(out, key=lambda x: (x.video, x.start))

    # --------------------------------------------------- map-reduce pieces
    def group_budget(self, out_tokens: int, query: str | None = None) -> int:
        llm = self.a.llm
        return llm.prompt_budget(out_tokens) - llm.count(prompts.map_system(query, out_tokens))

    def item_cost(self, it: Item) -> int:
        return self.a.llm.count(it.render(99)) + 2

    def group(self, items: list[Item], budget: int, max_items: int = 0) -> list[list[Item]]:
        """Split consecutive items (never across videos) into groups that fit `budget`."""
        groups, cur, used = [], [], 0
        for it in items:
            cost = self.item_cost(it)
            if cur and (used + cost > budget or it.video != cur[0].video or (max_items and len(cur) >= max_items)):
                groups.append(cur)
                cur, used = [], 0
            if cost > budget:
                it = Item(it.video, it.start, it.end, self.a.counter.truncate(it.text, budget - 40), thumb=it.thumb)
                cost = budget
            cur.append(it)
            used += cost
        if cur:
            groups.append(cur)
        return groups

    def summarize(self, group: list[Item], out_tokens: int, query: str | None = None) -> Item:
        """One map step: condense consecutive items into one item spanning their time range."""
        llm = self.a.llm
        sys_prompt = prompts.map_system(query, int(out_tokens * 0.8))
        ctx = "\n\n".join(it.text for it in group)  # already chronological; time tags get echoed back
        msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": ctx}]
        kw = dict(max_tokens=out_tokens, stop_on_repeat=_repetition_cut, frequency_penalty=0.5)
        try:
            text = llm.generate(msgs, **kw)
        except ContextOverflow:
            msgs[1]["content"] = self.a.counter.truncate(ctx, int(self.group_budget(out_tokens, query) * 0.75))
            text = llm.generate(msgs, **kw)
        return Item(group[0].video, group[0].start, group[-1].end, clean_text(text), thumb=group[0].thumb)

    def _condense(self, query: str | None, items: list[Item], budget: int):
        """Map-reduce until the items fit `budget`. Generator: yields status events, returns items."""
        level = 0
        while self._tokens(items) > budget:
            level += 1
            if level > 4:  # summaries refuse to shrink: keep what fits
                return self._pack(items, budget)
            n_groups = max(2, -(-self._tokens(items) // max(1, int(budget * 0.9))))
            out_tokens = int(max(80, min(300, budget // n_groups - 30)))
            groups = self.group(items, self.group_budget(out_tokens, query))
            new_items = []
            for gi, g in enumerate(groups):
                yield {"type": "status", "text": f"内容超过模型单次上限（prefill {self.a.llm.limits.max_prefill} "
                                                 f"token），分段整合 第 {level} 轮 {gi + 1}/{len(groups)}"}
                new_items.append(self.summarize(g, out_tokens, query))
                yield {"type": "condensed", "level": level, "index": gi, "total": len(groups),
                       "item": new_items[-1].as_dict()}
            items = new_items
        return items

    def _levels_for(self, names: list[str], budget: int) -> tuple[int, list[Item]] | None:
        """Lowest precomputed summary level (chapters) of the selected videos that fits the budget."""
        store = self.a.store
        depth = max((len(store.levels(n)) for n in names), default=0)
        for lv in range(1, depth + 1):
            items = []
            for n in names:
                levels = store.levels(n)
                if not levels:
                    return None
                items += [Item(n, c["start"], c["end"], c["text"], thumb=c.get("thumb", ""))
                          for c in levels[min(lv, len(levels)) - 1]]
            if self._tokens(items) <= budget:
                return lv, items
            # slightly too big: trim every chapter evenly rather than falling back to a coarser level
            per = budget // len(items) - 30
            if per >= 80:
                items = [Item(it.video, it.start, it.end, self.a.counter.truncate(it.text, per), thumb=it.thumb)
                         for it in items]
                if self._tokens(items) <= budget:
                    return lv, items
        return None

    # --------------------------------------------------------------- main
    def ask(self, query: str, videos: list[str] | None = None):
        """Yield events: {"type": "status"|"delta"|"done"|"error", ...}."""
        query = (query or "").strip()
        if not query:
            yield {"type": "error", "text": "请输入问题"}
            return
        store, s = self.a.store, self.a.settings
        names = [v for v in (videos or store.names()) if v in store.videos]
        segs = store.segments(names)
        if not segs:
            yield {"type": "error", "text": prompts.NO_VIDEO}
            return

        budget = self.answer_budget(query)
        all_items = [Item(x.video, x.start, x.end, _segment_text(x), thumb=_thumb(x)) for x in segs]
        total = self._tokens(all_items)
        lim = self.a.llm.limits
        plan = {"type": "plan", "segments": len(segs), "videos": len(names), "total_tokens": total,
                "budget": budget, "max_prefill": lim.max_prefill, "max_context": lim.max_context,
                "answer_tokens": s.answer_max_tokens, "global": is_global_question(query)}

        if total <= budget:
            mode, items = "full", all_items
            yield {**plan, "mode": mode}
        else:
            routed = self.a.embedder is None
            method, ranked, hits = "", [], []
            if not plan["global"]:
                # 1) literal match of the question's key words (shop signs, names, subtitles, speech)
                lex = lexical_scores(query, segs)
                ranked = sorted(zip(segs, lex), key=lambda t: (-t[1], t[0].video, t[0].start))
                hits = [(x, sc) for x, sc in ranked if sc >= 0.5][: s.top_k]
                method = "keyword"
                if not hits:
                    # 2) vector retrieval, or without an embedding model: the LLM reads the chapter summaries
                    method = "chapter" if routed else "vector"
                    yield {"type": "status", "text": "关键词没有直接命中，让模型根据章节摘要判断相关片段..." if routed
                           else "关键词没有直接命中，按语义相似度检索..."}
                    ranked = self._rank(query, names, segs)
                    hits = [(x, sc) for x, sc in ranked if sc >= s.score_threshold]
                    hits = hits if routed else hits[: s.top_k]  # routed: whole chapters, the budget trims them
            mode = "summarize" if (plan["global"] or not hits) else "retrieve"
            yield {**plan, "mode": mode}
            if ranked:
                yield {"type": "hits", "method": method, "threshold": 0.5 if method == "keyword" else s.score_threshold,
                       "used": mode == "retrieve", "routed": method == "chapter", "terms": query_terms(query),
                       "items": [{**Item(x.video, x.start, x.end, x.caption, sc, _thumb(x)).as_dict(), "ocr": x.ocr,
                                  "transcript": x.transcript} for x, sc in ranked[: max(s.top_k, 6)]]}
            if mode == "summarize":
                pre = self._levels_for(names, budget)
                if pre:
                    lv, items = pre
                    yield {"type": "status", "text": f"使用索引时预先整合的第 {lv} 层章节摘要（{len(items)} 条）"}
                    for i, it in enumerate(items):
                        yield {"type": "condensed", "level": lv, "index": i, "total": len(items),
                               "item": it.as_dict(), "precomputed": True}
                else:
                    top = None
                    for n in names:  # start from the highest precomputed level, if any
                        lv = self.a.store.levels(n)
                        top = (top or []) + ([Item(n, c["start"], c["end"], c["text"], thumb=c.get("thumb", ""))
                                              for c in lv[-1]] if lv else
                                             [it for it in all_items if it.video == n])
                    items = yield from self._condense(query, top or all_items, budget)
            else:
                items = []
                for rank, (seg, sc) in enumerate(hits):
                    caption = None
                    if rank < s.refine_top_n and seg.frames and method == "vector":
                        yield {"type": "status", "text": f"VLM 重新观察片段 {media.fmt_time(seg.start)}-"
                                                         f"{media.fmt_time(seg.end)}..."}
                        try:
                            caption = self.a.indexer.caption(seg, seg.frames, query=query, max_tokens=256)
                            yield {"type": "refined", "item": Item(seg.video, seg.start, seg.end, caption, sc,
                                                                   _thumb(seg)).as_dict()}
                        except Exception as e:
                            log.warning("refine caption failed: %s", e)
                    items.append(Item(seg.video, seg.start, seg.end, _segment_text(seg, caption), sc, _thumb(seg)))
                items = self._pack(items, budget)

        # ---- answer, shrinking the context if the server still says it is too long
        # global question over chapter summaries: the model only writes the overview, the chapter
        # list is appended verbatim (small models otherwise copy it slowly and run out of tokens)
        overview_only = mode == "summarize" and plan["global"] and len(items) > 1
        for attempt in range(4):
            context = "\n\n".join(it.render(i + 1) for i, it in enumerate(items))
            if overview_only:
                user, max_new = query + prompts.OVERVIEW_HINT, 200
            else:
                user = query + (prompts.global_hint([(it.start, it.end) for it in items]) if plan["global"] else "")
                max_new = s.answer_max_tokens
            msgs = [{"role": "system", "content": prompts.ANSWER_SYSTEM.format(context=context)},
                    {"role": "user", "content": user}]
            answer = ""
            try:
                yield {"type": "context", "tokens": self._tokens(items), "budget": budget,
                       "items": [it.as_dict(i + 1) for i, it in enumerate(items)]}
                yield {"type": "status", "text": f"LLM 生成回答（{len(items)} 条资料）..."}
                for chunk in self.a.llm.stream(msgs, max_tokens=max_new):
                    answer += chunk
                    cut = _repetition_cut(answer)
                    if cut is not None:  # greedy small models loop: stop and drop the repeats
                        log.warning("repetition detected, stopping generation")
                        answer = cut
                        yield {"type": "delta", "text": "", "answer": clean_text(answer)}
                        break
                    yield {"type": "delta", "text": chunk, "answer": clean_text(answer)}
                break
            except ContextOverflow:
                if answer or attempt == 3:
                    raise
                budget = int(budget * 0.75)
                log.warning("answer context overflow, shrinking budget to %d", budget)
                items = (self._pack(items, budget) if mode == "retrieve"
                         else (yield from self._condense(query, items, budget)))

        if overview_only:
            answer = clean_text(answer)
            if not answer.startswith("总述"):
                answer = "总述：" + answer
            tail = "".join(f"\n\n[{i}] {media.fmt_time(it.start)}-{media.fmt_time(it.end)}：{it.text}"
                           for i, it in enumerate(items, 1))
            answer += tail
            yield {"type": "delta", "text": tail, "answer": answer}

        answer = clean_text(answer)
        cited = []
        for m in _CITE_RE.findall(answer):
            n = int(m)
            if 1 <= n <= len(items) and n not in cited:
                cited.append(n)
        if not cited and (mode == "retrieve" or overview_only):
            cited = list(range(1, len(items) + 1))
        refs = [items[n - 1].as_dict(n) for n in cited]
        if not answer:
            yield {"type": "error", "text": "模型没有返回内容"}
            return
        yield {"type": "done", "answer": answer, "refs": refs, "mode": mode}
