"""VideoAgent facade: wires settings, model clients, index store, indexer and query engine."""
import logging
import os
from dataclasses import asdict

from .clients import ChatModel, EmbeddingModel, PerceptionClient, _LimitCache
from .config import Settings
from .indexer import Indexer
from .query import QueryEngine
from .store import Store
from .tokens import TokenCounter

log = logging.getLogger("videoagent")


class VideoAgent:
    def __init__(self, working_dir: str | None = None, settings: Settings | None = None):
        self.settings = settings or Settings()
        if working_dir:
            self.settings.working_dir = working_dir
        s = self.settings
        self.store = Store(s.working_dir)
        # model limits depend only on the served model, so share the cache across working dirs
        cache = _LimitCache(os.path.join(os.path.expanduser(os.getenv("VIDEOAGENT_CACHE_DIR", "~/.cache/videoagent")),
                                         "model_limits.json"))
        self.counter = TokenCounter(s.tokenizer_json)
        self.llm = ChatModel(s.llm, cache, self.counter, "llm", vision=False, timeout=s.request_timeout,
                             # never when the VLM doubles as the LLM (Instruct VLMs break on enable_thinking)
                             thinking_switch=s.llm_disable_thinking and
                             s.llm.base_url.rstrip("/") != s.vlm.base_url.rstrip("/"))
        self.vlm = ChatModel(s.vlm, cache, self.counter, "vlm", vision=True, timeout=s.request_timeout)
        self.embedder = (EmbeddingModel(s.embedding, cache, timeout=s.request_timeout)
                         if s.embedding.base_url else None)
        self.perception = PerceptionClient(s.perception_url)
        self.indexer = Indexer(self)
        self.engine = QueryEngine(self)

    # ------------------------------------------------------------- public
    def index(self, paths: list[str] | str, progress=None) -> list[str]:
        """Index videos (already indexed files are skipped). Returns their names."""
        if isinstance(paths, str):
            paths = [paths]
        return [self.indexer.index(p, progress)[0] for p in paths]

    def ask(self, query: str, videos: list[str] | None = None):
        """Stream answer events, see QueryEngine.ask."""
        return self.engine.ask(query, videos)

    def query(self, query: str, videos: list[str] | None = None) -> dict:
        """Blocking helper: {"answer", "refs", "mode"}; raises on error."""
        for ev in self.ask(query, videos):
            if ev["type"] == "error":
                raise RuntimeError(ev["text"])
            if ev["type"] == "done":
                return ev
        raise RuntimeError("no answer")

    def status(self) -> dict:
        out = {}
        for name, m in (("llm", self.llm), ("vlm", self.vlm), ("embedding", self.embedder)):
            if m is None:
                out[name] = "(disabled: questions are routed via chapter summaries)"
                continue
            if name == "llm" and m.base == self.vlm.base:
                out[name] = f"(same service as the VLM: {m.base})"
                continue
            try:
                out[name] = {"url": m.base, **asdict(m.limits)}
            except Exception as e:
                out[name] = {"url": m.base, "error": str(e)}
        out["perception"] = {"url": self.perception.base or "(disabled)", **self.perception.capabilities()}
        out["token_counter"] = "exact (tokenizer.json)" if self.counter.exact else "estimate + auto-calibration"
        out["videos"] = self.store.names()
        return out
