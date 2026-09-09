"""Per-call analysis: everything derivable from ONE row, with no grouping.

This is the surface that survives a tiny export. It works at N=1, so it still says something
useful when there are too few rows per callsite for any group-level conclusion.

Deliberately NOT here, because none of it is knowable from a single call: static vs dynamic
regions, cacheable *potential*, callsite identity, run membership.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from profiler.load import Call

CHARS_PER_TOKEN = 4
_STRUCTURE_RE = re.compile(r"^\s*([-*+]|\d+[.)]|#{1,6}\s|<[a-zA-Z_/]+>|\|)")


def tokens(chars: int) -> int:
    return chars // CHARS_PER_TOKEN


def text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False, default=str)


def tool_schema_chars(tools: List[Dict[str, Any]]) -> int:
    return sum(len(json.dumps(t, ensure_ascii=False, default=str)) for t in tools)


@dataclass
class CallFacts:
    """One call, measured. Every field is directly observable."""

    request_id: str
    app_id: str
    model: str

    # size, split by where the tokens actually go
    system_tokens: int = 0
    tool_schema_tokens: int = 0
    user_tokens: int = 0
    history_tokens: int = 0      # assistant + tool result messages
    estimated_prompt_tokens: int = 0
    reported_prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # cache reality, straight from the provider
    cached_tokens: Optional[int] = None
    cache_hit_ratio: Optional[float] = None

    # tools
    tools_declared: int = 0
    tools_called: int = 0
    tool_names: List[str] = field(default_factory=list)

    # shape
    role_seq: str = ""
    n_messages: int = 0
    history_depth: int = 0
    prose_share: float = 0.0
    duplicate_line_tokens: int = 0     # repeated lines *within* this one prompt
    longest_line_chars: int = 0

    # response
    finish_reason: Optional[str] = None
    response_format: Optional[str] = None

    flags: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()}
        return data


ROLE_CHARS = {"system": "S", "developer": "S", "user": "U", "assistant": "A", "tool": "T"}


def analyse_call(call: Call) -> CallFacts:
    """Measure one call. No comparisons, no group, no LLM."""
    facts = CallFacts(request_id=call.request_id, app_id=call.app_id, model=call.model)

    system_chars = user_chars = history_chars = 0
    all_lines: List[str] = []
    for message in call.messages:
        role = message.get("role")
        body = text_of(message.get("content"))
        if role in ("system", "developer"):
            system_chars += len(body)
            all_lines.extend(ln.strip() for ln in body.split("\n") if ln.strip())
        elif role == "user":
            user_chars += len(body)
            all_lines.extend(ln.strip() for ln in body.split("\n") if ln.strip())
        else:
            history_chars += len(body)
            # tool calls carry weight even when content is empty
            history_chars += len(json.dumps(message.get("tool_calls") or [], default=str))

    schema_chars = tool_schema_chars(call.tools)
    facts.system_tokens = tokens(system_chars)
    facts.tool_schema_tokens = tokens(schema_chars)
    facts.user_tokens = tokens(user_chars)
    facts.history_tokens = tokens(history_chars)
    facts.estimated_prompt_tokens = tokens(
        system_chars + schema_chars + user_chars + history_chars
    )
    facts.reported_prompt_tokens = call.prompt_tokens
    facts.completion_tokens = call.completion_tokens

    facts.cached_tokens = call.cached_tokens
    if call.cached_tokens is not None and call.prompt_tokens:
        facts.cache_hit_ratio = round(call.cached_tokens / call.prompt_tokens, 4)

    facts.tools_declared = len(call.tools)
    facts.tool_names = sorted(
        (t.get("function") or {}).get("name", "") for t in call.tools
    )
    facts.tools_called = len(call.response_message.get("tool_calls") or [])

    facts.role_seq = "".join(
        ROLE_CHARS.get(m.get("role") or "", "?") for m in call.messages
    )
    facts.n_messages = len(call.messages)
    facts.history_depth = sum(
        1 for m in call.messages if m.get("role") in ("assistant", "tool")
    )

    structural = sum(1 for ln in all_lines if _STRUCTURE_RE.match(ln))
    facts.prose_share = round(1.0 - structural / len(all_lines), 3) if all_lines else 0.0
    facts.longest_line_chars = max((len(ln) for ln in all_lines), default=0)

    repeats = Counter(all_lines)
    facts.duplicate_line_tokens = tokens(
        sum(len(ln) * (count - 1) for ln, count in repeats.items() if count > 1)
    )

    facts.finish_reason = call.finish_reason
    facts.response_format = (call.response_format or {}).get("type")

    facts.flags = _flags(facts)
    return facts


def _flags(f: CallFacts) -> List[str]:
    """Things visible in a single call that are worth a human's attention."""
    out: List[str] = []
    if f.cache_hit_ratio is not None and f.cache_hit_ratio == 0 \
            and (f.reported_prompt_tokens or 0) >= 1024:
        out.append("no cache hit on a prompt large enough to cache")
    if f.tools_declared and f.tool_schema_tokens > max(f.system_tokens, 1) * 1.5:
        out.append("tool schemas outweigh the system prompt")
    if f.tools_declared >= 15:
        out.append("15 or more tools declared in one call")
    if f.duplicate_line_tokens > 50:
        out.append("duplicated lines within a single prompt")
    if f.history_tokens > 4 * max(f.system_tokens + f.user_tokens, 1):
        out.append("history dominates the payload")
    if f.finish_reason == "length":
        out.append("response truncated by max_tokens")
    return out


def summarise(facts: List[CallFacts]) -> Dict[str, Any]:
    """App-level roll-up of the single-call surface. Still no grouping involved."""
    if not facts:
        return {}
    total_prompt = sum(f.reported_prompt_tokens or f.estimated_prompt_tokens for f in facts)
    cached = [f.cached_tokens for f in facts if f.cached_tokens is not None]
    ratios = [f.cache_hit_ratio for f in facts if f.cache_hit_ratio is not None]
    flag_counts = Counter(flag for f in facts for flag in f.flags)
    return {
        "calls": len(facts),
        "total_prompt_tokens": total_prompt,
        "total_completion_tokens": sum(f.completion_tokens or 0 for f in facts),
        "total_cached_tokens": sum(cached) if cached else None,
        "mean_cache_hit_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "calls_with_zero_cache": sum(1 for r in ratios if r == 0),
        "mean_tools_declared": round(
            sum(f.tools_declared for f in facts) / len(facts), 2),
        "mean_history_depth": round(
            sum(f.history_depth for f in facts) / len(facts), 2),
        "token_split": {
            "system": sum(f.system_tokens for f in facts),
            "tool_schemas": sum(f.tool_schema_tokens for f in facts),
            "user": sum(f.user_tokens for f in facts),
            "history": sum(f.history_tokens for f in facts),
        },
        "flags": dict(flag_counts.most_common()),
    }
