"""Token counting for context budgeting.

Exact counting needs the model's HF tokenizer.json; without it we estimate and
self-calibrate against the `usage.prompt_tokens` the server reports back, so the
estimate converges to the real tokenizer after a few requests. Budgets always
keep a safety margin and callers retry with a smaller budget on overflow, so an
estimate that is slightly off never breaks a request.
"""
import re
import threading

_CJK_RE = re.compile(r"[　-〿㐀-䶿一-鿿＀-￯]")
_DIGIT_RE = re.compile(r"[0-9]")
_SPACE_RE = re.compile(r"\s+")


class TokenCounter:
    def __init__(self, tokenizer_json: str = ""):
        self._tok = None
        if tokenizer_json:
            try:
                from tokenizers import Tokenizer
                self._tok = Tokenizer.from_file(tokenizer_json)
            except Exception as e:  # optional dependency / bad path: fall back to estimate
                print(f"[tokens] cannot load {tokenizer_json} ({e}); using estimate")
        self._scale = 1.0
        self._lock = threading.Lock()

    @property
    def exact(self) -> bool:
        return self._tok is not None

    @staticmethod
    def _raw_estimate(text: str) -> float:
        if not text:
            return 0.0
        cjk = len(_CJK_RE.findall(text))
        digits = len(_DIGIT_RE.findall(text))  # Qwen splits numbers into single digits
        rest = len(_SPACE_RE.sub("", text)) - cjk - digits
        return cjk * 1.0 + digits + rest / 3.2 + 1

    def count(self, text: str) -> int:
        if self._tok is not None:
            return len(self._tok.encode(text, add_special_tokens=False).ids)
        return int(self._raw_estimate(text) * self._scale) + 1

    def calibrate(self, text: str, actual_tokens: int, overhead: int = 0):
        """Feed back the real prompt token count reported by the server."""
        if self._tok is not None or actual_tokens <= 0:
            return
        est = self._raw_estimate(text)
        if est < 64:  # too short to say anything about the ratio
            return
        ratio = max(0.5, min(2.5, (actual_tokens - overhead) / est))
        with self._lock:
            # never scale below 1.0: under-estimating is the dangerous direction
            self._scale = max(1.0, 0.7 * self._scale + 0.3 * ratio * 1.05)

    def truncate(self, text: str, max_tokens: int) -> str:
        if max_tokens <= 0:
            return ""
        if self.count(text) <= max_tokens:
            return text
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.count(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + "…"
