"""The one place that talks to a model.

Responsibilities kept here so no other module has to think about them: retries, JSON-mode
parsing, cost accounting, and an on-disk response cache so a re-run costs nothing and the
pipeline stays reproducible.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from profiler.audit import Audit

# USD per 1M tokens. Approximate and easy to update; used only for the audit trail.
PRICING: Dict[str, tuple] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
}
DEFAULT_PRICE = (1.00, 4.00)

# Tokens-per-minute headroom per model. A single shared budget throttles the cheap model down
# to the frontier model's limit for no reason, which is most of what made a run take hours.
# Conservative defaults; raise them if your account tier allows.
TPM_BUDGETS: Dict[str, int] = {
    "gpt-4.1": 28_000,
    "gpt-4o": 28_000,
    "gpt-4.1-mini": 180_000,
    "gpt-4o-mini": 180_000,
}
DEFAULT_TPM = 28_000


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    prompt_price, completion_price = PRICING.get(model, DEFAULT_PRICE)
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


class RateLimiter:
    """Pace requests under a tokens-per-minute budget.

    Reacting to a 429 costs a 20-70s sleep across every worker; staying under the limit costs
    nothing. Tokens are estimated before sending (chars/4) and charged to a rolling 60s window.
    """

    def __init__(self, tokens_per_minute: int) -> None:
        self.budget = max(tokens_per_minute, 1000)
        self._window: deque = deque()          # (timestamp, tokens)
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> float:
        """Block until this request fits in the budget. Returns seconds waited."""
        waited = 0.0
        estimated_tokens = min(estimated_tokens, self.budget)
        while True:
            with self._lock:
                now = time.time()
                while self._window and now - self._window[0][0] > 60.0:
                    self._window.popleft()
                used = sum(t for _, t in self._window)
                if used + estimated_tokens <= self.budget:
                    self._window.append((now, estimated_tokens))
                    return waited
                oldest = self._window[0][0] if self._window else now
                sleep_for = max(0.25, 60.0 - (now - oldest))
            time.sleep(min(sleep_for, 5.0))
            waited += min(sleep_for, 5.0)


@dataclass
class LLMResult:
    ok: bool
    data: Any = None
    error: Optional[str] = None
    raw: str = ""


class LLMClient:
    """A small, explicit wrapper. Returns structured data or an error - never raises upward."""

    def __init__(self, api_key: Optional[str], audit: Audit, cache_dir: str,
                 enabled: bool = True, timeout: float = 45.0, max_retries: int = 4,
                 tokens_per_minute: int = 28000) -> None:
        self.audit = audit
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.max_retries = max_retries
        self.enabled = bool(enabled and api_key)
        self.default_tpm = tokens_per_minute
        self._limiters: Dict[str, RateLimiter] = {}
        self._limiter_lock = threading.Lock()
        self._client = None
        os.makedirs(cache_dir, exist_ok=True)
        if self.enabled:
            from openai import OpenAI
            self._client = OpenAI(api_key=api_key, timeout=timeout, max_retries=0)

    def limiter_for(self, model: str) -> "RateLimiter":
        with self._limiter_lock:
            if model not in self._limiters:
                self._limiters[model] = RateLimiter(
                    TPM_BUDGETS.get(model, self.default_tpm))
            return self._limiters[model]

    # -- cache -----------------------------------------------------------------------
    def _key(self, model: str, system: str, user: str, schema_name: str) -> str:
        blob = json.dumps([model, system, user, schema_name], sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _cached(self, key: str) -> Optional[Any]:
        path = os.path.join(self.cache_dir, key + ".json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (ValueError, OSError):
            return None

    def _store(self, key: str, payload: Any) -> None:
        try:
            with open(os.path.join(self.cache_dir, key + ".json"), "w",
                      encoding="utf-8") as handle:
                json.dump(payload, handle)
        except OSError:
            pass

    # -- the call ---------------------------------------------------------------------
    def json_call(self, purpose: str, model: str, system: str, user: str,
                  schema_hint: str = "", max_tokens: int = 1600,
                  temperature: float = 0.0, **audit_extra: Any) -> LLMResult:
        """Ask for a JSON object back. `schema_hint` is appended to the system prompt."""
        if not self.enabled:
            return LLMResult(ok=False, error="llm disabled")

        full_system = system if not schema_hint else f"{system}\n\n{schema_hint}"
        key = self._key(model, full_system, user, purpose)
        prompt_hash = key[:12]

        cached = self._cached(key)
        if cached is not None:
            self.audit.llm(purpose=purpose, model=model, prompt_tokens=0, completion_tokens=0,
                           cost_usd=0.0, cached=True, prompt_hash=prompt_hash, **audit_extra)
            return LLMResult(ok=True, data=cached, raw=json.dumps(cached))

        # Stay under the budget rather than discovering it with a 429.
        # Reserve a realistic completion, not the ceiling: max_tokens is an upper bound that
        # real replies rarely approach, and charging it in full throttled the run to a crawl.
        estimated = (len(full_system) + len(user)) // 4 + min(max_tokens, 600)
        self.limiter_for(model).acquire(estimated)

        last_error = ""
        for attempt in range(self.max_retries):
            try:
                response = self._client.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": full_system},
                              {"role": "user", "content": user}],
                    response_format={"type": "json_object"},
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                text = (response.choices[0].message.content or "").strip()
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                cost = estimate_cost(model, prompt_tokens, completion_tokens)

                try:
                    data = json.loads(text)
                except ValueError as exc:
                    last_error = f"invalid JSON: {exc}"
                    self.audit.event("llm_bad_json", purpose=purpose, attempt=attempt,
                                     prompt_hash=prompt_hash)
                    continue

                self.audit.llm(purpose=purpose, model=model, prompt_tokens=prompt_tokens,
                               completion_tokens=completion_tokens, cost_usd=cost,
                               cached=False, prompt_hash=prompt_hash, **audit_extra)
                self._store(key, data)
                return LLMResult(ok=True, data=data, raw=text)

            except Exception as exc:                      # network, rate limit, server error
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
                self.audit.event("llm_error", purpose=purpose, attempt=attempt,
                                 error=last_error)
                if attempt + 1 < self.max_retries:
                    time.sleep(_backoff(exc, attempt))

        return LLMResult(ok=False, error=last_error)


_RETRY_AFTER_RE = re.compile(r"try again in ([\d.]+)(m?s)", re.I)


def _backoff(exc: Exception, attempt: int) -> float:
    """How long to wait before retrying.

    A rate limit measured per MINUTE cannot be waited out in the two seconds an exponential
    backoff starting at 1 would give it, so 429s get their own, much longer schedule. The
    provider usually states the wait in the message; honour it when it is there.
    """
    text = str(exc)
    is_rate_limit = "429" in text or "RateLimit" in type(exc).__name__
    if not is_rate_limit:
        return min(2 ** attempt, 8) + random.random()

    match = _RETRY_AFTER_RE.search(text)
    if match:
        value = float(match.group(1))
        seconds = value / 1000.0 if match.group(2).lower() == "ms" else value
        return min(max(seconds + 1.0, 2.0), 70.0)
    # no hint: wait out a good part of the per-minute window, with jitter so workers desync
    return min(20.0 * (attempt + 1), 70.0) + random.random() * 3


def clip(text: str, limit: int) -> str:
    """Bound what we send. Truncation is marked so the model knows it is seeing a fragment."""
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n...[{len(text) - limit} chars omitted]...\n" + text[-tail:]
