"""Node-level metrics: what is cacheable, what is wasted, what shape the callsite has.

These are measurements, not predictions, so they need no ground truth and are valid on real
traffic immediately. They are also where the money is: the cacheable-prefix numbers convert
directly into a token count you can act on.

Validated in the spike by a controlled A/B - two callsites with byte-identical content differing
only in where a dynamic header sat scored a cacheable prefix of 3 tokens versus 773.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profiler.load import Call
from profiler.segment import Segmentation, common_prefix, lines_of, static_dynamic
from profiler.single import CHARS_PER_TOKEN, text_of, tokens

# Providers only cache a prefix beyond a minimum length; below it, reordering buys nothing.
# Model-dependent (commonly ~1024 tokens), so it is a setting rather than a constant.
DEFAULT_CACHE_MINIMUM = 1024


def request_document(call: Call) -> str:
    """The request serialised in wire order: tools, then messages.

    The cacheable prefix is a property of the whole serialised request, not of the system
    prompt. Measured: reading only the system prompt reported 2,886 cacheable tokens for a real
    coding agent whose true figure is 5,544 - the tool schemas are nearly as large as the prompt.
    """
    parts: List[str] = []
    for tool in call.tools:
        fn = tool.get("function") or {}
        parts.append(f"TOOL {fn.get('name', '')}: {fn.get('description', '')}")
        parameters = fn.get("parameters")
        if parameters:
            parts.append(json.dumps(parameters, sort_keys=True, ensure_ascii=False,
                                    default=str))
    for message in call.messages:
        parts.append(f"[{message.get('role')}] {text_of(message.get('content'))}")
    return "\n".join(parts)


def prose_document(call: Call) -> str:
    """Only the natural-language surfaces, for line-level frequency analysis."""
    return "\n".join(
        text_of(m.get("content")) for m in call.messages
        if m.get("role") in ("system", "developer", "user")
    )


@dataclass
class NodeMetrics:
    calls: int
    avg_request_tokens: int
    cacheable_now: int              # shared prefix across the node's calls, in tokens
    recoverable: int                # additional tokens cacheable if dynamic regions moved last
    static_share: float
    wasted_tokens: int              # recoverable x calls, over this export
    reported_cached_mean: Optional[float] = None
    cache_minimum: int = DEFAULT_CACHE_MINIMUM
    tools_declared: int = 0
    distinct_tool_sets: int = 1
    tools_called: List[str] = field(default_factory=list)
    tools_never_called: List[str] = field(default_factory=list)
    tool_schema_tokens: int = 0
    avg_messages: float = 0.0
    prose_share: float = 0.0
    anatomy: Optional["NodeAnatomy"] = None
    findings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["anatomy"] = self.anatomy.as_dict() if self.anatomy else None
        return data


def compute(calls: Sequence[Call], seg: Segmentation,
            cache_minimum: int = DEFAULT_CACHE_MINIMUM,
            facts: Optional[Sequence[Any]] = None) -> NodeMetrics:
    documents = [request_document(c) for c in calls]

    prefix = common_prefix(documents)
    avg_chars = sum(len(d) for d in documents) // max(len(documents), 1)

    declared: Dict[str, Dict[str, Any]] = {}
    tool_sets = set()
    for call in calls:
        names = []
        for tool in call.tools:
            fn = tool.get("function") or {}
            name = fn.get("name")
            if name:
                declared[name] = tool
                names.append(name)
        tool_sets.add(tuple(sorted(names)))
    called = sorted({
        (tc.get("function") or {}).get("name")
        for c in calls for tc in (c.response_message.get("tool_calls") or [])
        if (tc.get("function") or {}).get("name")
    })

    cached_values = [c.cached_tokens for c in calls if c.cached_tokens is not None]

    tool_schema_tokens = tokens(sum(
        len(json.dumps(t, ensure_ascii=False, default=str)) for t in declared.values()))
    cacheable_now = tokens(len(prefix))

    # What the prefix COULD be if every dynamic region moved after the stable content: all the
    # static content of the whole request.
    #
    # The bug this replaces subtracted a prose-derived number from a whole-request-derived one:
    # `tokens(static prose) - tokens(whole-request prefix)`. Large stable tool schemas inflate
    # the prefix, so the difference clamped to zero and a genuine reordering win stayed hidden.
    # Measured on the fixture: 49 of 56 callsites declared tools, and exactly 1 ever reported a
    # non-zero figure. Both sides are now whole-request quantities.
    #
    # Tool schemas only count as static when the tool list never varies; when it does, the
    # schemas are themselves a source of prefix breakage and cannot be assumed stable.
    static_prose_tokens = tokens(len("\n".join(seg.static_lines)))
    static_tool_tokens = tool_schema_tokens if len(tool_sets) == 1 else 0
    recoverable = max(0, static_prose_tokens + static_tool_tokens - cacheable_now)

    metrics = NodeMetrics(
        calls=len(calls),
        avg_request_tokens=tokens(avg_chars),
        cacheable_now=cacheable_now,
        recoverable=recoverable,
        static_share=round(
            seg.static_chars / max(seg.static_chars + seg.dynamic_chars, 1), 3),
        wasted_tokens=recoverable * len(calls),
        reported_cached_mean=(round(sum(cached_values) / len(cached_values), 1)
                              if cached_values else None),
        cache_minimum=cache_minimum,
        tools_declared=len(declared),
        distinct_tool_sets=len(tool_sets),
        tools_called=called,
        tools_never_called=sorted(set(declared) - set(called)),
        tool_schema_tokens=tool_schema_tokens,
        avg_messages=round(sum(len(c.messages) for c in calls) / max(len(calls), 1), 1),
        prose_share=round(1.0 - (len([1 for ln in lines_of("\n".join(seg.static_lines))
                                      if ln.startswith(("-", "*", "#", "<", "|"))])
                                 / max(len(seg.static_lines), 1)), 2),
    )
    if facts:
        metrics.anatomy = anatomy(facts, calls)
    metrics.findings = _findings(metrics, seg)
    return metrics


def _findings(m: NodeMetrics, seg: Segmentation) -> List[str]:
    """Deterministic, defensible statements - the profile's own conclusions."""
    out: List[str] = []

    if m.recoverable > 50 and m.cacheable_now < m.avg_request_tokens * 0.25:
        out.append(
            f"cache-prefix: a dynamic region sits ahead of the static content; moving it after "
            f"the stable text would make ~{m.recoverable} more tokens cacheable "
            f"({m.wasted_tokens:,} tokens across this export)")
    if m.cacheable_now >= m.cache_minimum:
        out.append(f"cache-prefix: {m.cacheable_now} token stable prefix already exceeds the "
                   f"~{m.cache_minimum} token minimum; caching is viable as-is")
    elif m.cacheable_now + m.recoverable >= m.cache_minimum > m.cacheable_now:
        out.append(f"cache-prefix: reordering would lift the prefix from {m.cacheable_now} past "
                   f"the ~{m.cache_minimum} token cache minimum")

    if m.reported_cached_mean is not None and m.reported_cached_mean == 0 \
            and m.cacheable_now >= m.cache_minimum:
        out.append("cache-prefix: the provider reports zero cached tokens despite a stable "
                   "prefix long enough to cache - check that caching is enabled for this route")

    if m.tools_never_called:
        out.append(
            f"tools: {len(m.tools_never_called)} of {m.tools_declared} declared tools are never "
            f"called in this export ({', '.join(m.tools_never_called[:5])}"
            f"{'...' if len(m.tools_never_called) > 5 else ''}); their schemas cost "
            f"~{m.tool_schema_tokens} tokens on every call")
    if m.distinct_tool_sets > 1:
        out.append(f"tools: the tool list varies across calls ({m.distinct_tool_sets} distinct "
                   f"sets), which breaks the cached prefix on every change")

    if m.anatomy and m.anatomy.truncated:
        share = m.anatomy.truncated / max(m.calls, 1)
        out.append(f"responses: {m.anatomy.truncated} of {m.calls} replies "
                   f"({share:.0%}) stopped at the token limit rather than "
                   f"finishing - output is being cut off")
    if m.anatomy and len(m.anatomy.models) > 1:
        out.append(f"models: this callsite runs on {len(m.anatomy.models)} "
                   f"different models ({', '.join(list(m.anatomy.models)[:4])}), "
                   f"so cost and behaviour are not comparable across its calls")
    if m.avg_messages > 20:
        out.append(f"history: {m.avg_messages:.0f} messages per call on average; check the "
                   f"truncation policy")
    # Compare per call, not totals: dynamic_chars sums every DISTINCT dynamic line across the
    # node, so with 300 calls carrying a unique user line it is always large and the finding
    # would fire on nearly every node.
    dynamic_per_call = seg.dynamic_chars / max(m.calls, 1)
    if seg.static_chars and dynamic_per_call > seg.static_chars * 1.5:
        out.append(
            f"compression: per call the varying text (~{tokens(int(dynamic_per_call))} tok) "
            f"outweighs the template (~{tokens(seg.static_chars)} tok); the payload, not the "
            f"instructions, is what costs here")
    return out


# ---------------------------------------------------------------------------------------
# Anatomy: what a callsite is actually made of.
#
# `role_shape` ("SU") is a CLUSTERING feature - it helps decide whether two requests belong
# together. It was also being shown to the reader, where it says almost nothing: "SU x60" means
# sixty calls had one system message and one user message, which is true of most of an estate.
#
# What earns confidence is the anatomy: how the request divides into regions, which of those
# regions hold still and which move, and what comes back. All of it is already measured per call
# by `single.py`; it was simply never assembled per callsite.
# ---------------------------------------------------------------------------------------

@dataclass
class Region:
    """One part of the request - instructions, tool schemas, user turn, history."""

    name: str
    mean_tokens: int
    share: float                  # of the mean request, measured against the regions' own sum
    p10: int
    p90: int
    static_share: float = 1.0     # fraction of this region's characters identical across calls
    # The split itself, kept rather than recomputed. The store and the HTML page both need it,
    # and on a 200k-token callsite recomputing a document frequency three times is the run.
    static_lines: List[str] = field(default_factory=list)
    dynamic_lines: List[str] = field(default_factory=list)

    @property
    def varies(self) -> bool:
        """Does the CONTENT of this region change between calls?

        Size was the first test and it was wrong. A retrieval payload is very often the same
        length every call and entirely different text: `app_rag` sent 9,546 tokens of retrieved
        passages that the profile then reported as "the same every call". Content stability is
        measured by line-level document frequency within the region, exactly as the static /
        dynamic segmentation does for the prompt as a whole. Size is kept as a secondary signal
        for regions that grow, such as a conversation history.
        """
        return self.static_share < 0.9 or self.p90 > max(self.p10 * 1.15, self.p10 + 20)

    def as_dict(self) -> Dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items()
                if k not in ("static_lines", "dynamic_lines")}
        return {**data, "varies": self.varies,
                "static_line_count": len(self.static_lines),
                "dynamic_line_count": len(self.dynamic_lines)}


@dataclass
class NodeAnatomy:
    regions: List[Region] = field(default_factory=list)
    models: Dict[str, int] = field(default_factory=dict)
    finish_reasons: Dict[str, int] = field(default_factory=dict)
    truncated: int = 0
    completion_p50: int = 0
    completion_p95: int = 0
    tool_call_rate: float = 0.0
    distinct_ratio: float = 0.0
    span_hours: float = 0.0
    calls_per_hour: float = 0.0
    description: str = ""

    def as_dict(self) -> Dict[str, Any]:
        data = dict(self.__dict__)
        data["regions"] = [r.as_dict() for r in self.regions]
        return data


def _pct(values: Sequence[float], q: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    i = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return int(ordered[i])


def _describe(anatomy: NodeAnatomy, calls: int) -> str:
    """One plain sentence. Deterministic - assembled from the numbers, not written by a model."""
    held, moves = [], []
    for r in anatomy.regions:
        if r.mean_tokens < 5:
            continue
        if r.varies:
            pct = f" ({1 - r.static_share:.0%} of it new each time)" if r.static_share < 1 else ""
            moves.append(f"{r.mean_tokens:,} tokens of {r.name}{pct}")
        else:
            held.append(f"{r.mean_tokens:,} tokens of {r.name}")

    parts = []
    if held:
        parts.append("sends " + ", ".join(held) + " that stay the same every call")
    if moves:
        parts.append(("plus " if held else "sends ") + ", ".join(moves) + " that change")
    if not parts:
        parts.append("sends a small, uniform request")

    reply = f"replies in ~{anatomy.completion_p50:,} tokens"
    if anatomy.completion_p95 > anatomy.completion_p50 * 2:
        reply += f" (p95 {anatomy.completion_p95:,})"
    parts.append(reply)

    if anatomy.tool_call_rate > 0:
        parts.append(f"calls a tool on {anatomy.tool_call_rate:.0%} of calls")
    if anatomy.truncated:
        parts.append(f"**{anatomy.truncated} of {calls} responses hit the token limit**")
    if len(anatomy.models) > 1:
        parts.append(f"spread across {len(anatomy.models)} models")
    return "This callsite " + "; ".join(parts) + "."


def region_text(call: Call, region: str) -> str:
    """The text of one region of one request. The unit the static/dynamic split works on."""
    if region == "instructions":
        return "\n".join(text_of(m.get("content")) for m in call.messages
                         if m.get("role") in ("system", "developer"))
    if region == "the user turn":
        return "\n".join(text_of(m.get("content")) for m in call.messages
                         if m.get("role") == "user")
    if region == "conversation history":
        return "\n".join(text_of(m.get("content")) for m in call.messages
                         if m.get("role") in ("assistant", "tool"))
    if region == "tool schemas":
        return "\n".join(json.dumps(t, sort_keys=True, ensure_ascii=False, default=str)
                         for t in call.tools)
    return ""


def region_split(calls: Sequence[Call], region: str) -> Tuple[List[str], List[str], float]:
    """(static lines, dynamic lines, static char share) for one region across a node's calls.

    Same document-frequency method as `segment`, scoped to a single region, so "which part of
    the user turn is template and which is payload" can be answered rather than inferred from a
    whole-prompt average.
    """
    texts = [region_text(c, region) for c in calls]
    if not any(texts):
        return [], [], 1.0
    static, dynamic, df = static_dynamic(texts)
    modal = max(texts, key=len)
    ordered_static = [ln for ln in lines_of(modal) if ln in static]

    # Per CALL, not summed across the node. Summing every distinct dynamic line means a node of
    # 300 calls each carrying one unique line has 300x the dynamic mass of a node of one call,
    # so the share would fall as the export grew rather than describing the request.
    static_chars = sum(len(ln) for ln in static)
    dynamic_chars = sum(
        sum(len(ln) for ln in lines_of(t) if ln in dynamic) for t in texts) / len(texts)
    share = static_chars / max(static_chars + dynamic_chars, 1)
    return (ordered_static,
            sorted(dynamic, key=lambda ln: -df[ln])[:200],
            share)


def per_call_split(call: Call, static_by_region: Dict[str, set]) -> Tuple[int, int]:
    """(static tokens, dynamic tokens) for ONE request, given the node's static line sets.

    Region-level p10/p90 says how much the group's payload varies; this says how much THIS
    request carries. Without it a callsite reads as uniform when a handful of its requests are
    fifty times the size of the rest - and those few are usually where the money is.
    """
    static_chars = dynamic_chars = 0
    for region in ("instructions", "tool schemas", "the user turn", "conversation history"):
        known = static_by_region.get(region) or set()
        for line in lines_of(region_text(call, region)):
            if line in known:
                static_chars += len(line)
            else:
                dynamic_chars += len(line)
    return tokens(static_chars), tokens(dynamic_chars)


def anatomy(facts: Sequence[Any], calls: Sequence[Call]) -> NodeAnatomy:
    """Assemble one node's calls into something a person can judge.

    `facts` are `single.CallFacts` for this node's calls only. Content stability is measured per
    region rather than borrowed from a whole-prompt average, because the two disagree exactly
    where it matters: a RAG callsite's user turn is 100% of the request and 0% template, and a
    whole-prompt figure hides that behind the system prompt's stability.
    """
    result = NodeAnatomy()
    if not facts:
        return result

    measured = []
    for name, attr in (("instructions", "system_tokens"),
                       ("tool schemas", "tool_schema_tokens"),
                       ("the user turn", "user_tokens"),
                       ("conversation history", "history_tokens")):
        values = [getattr(f, attr) for f in facts]
        mean = int(sum(values) / len(values))
        if mean >= 1:
            measured.append((name, mean, values))

    # Share is taken against the regions' own sum, not `avg_request_tokens`. The two come from
    # different estimators - regions from per-message text, the average from the serialised
    # request - so mixing them made one callsite's shares total 114%.
    total = max(sum(mean for _, mean, _ in measured), 1)
    for name, mean, values in measured:
        static_lines, dynamic_lines, static_share = region_split(calls, name)
        result.regions.append(Region(
            name=name, mean_tokens=mean, share=round(mean / total, 3),
            p10=_pct(values, 0.10), p90=_pct(values, 0.90),
            static_share=round(static_share, 3),
            static_lines=static_lines, dynamic_lines=dynamic_lines))

    completions = [f.completion_tokens for f in facts if f.completion_tokens is not None]
    result.completion_p50 = _pct(completions, 0.50)
    result.completion_p95 = _pct(completions, 0.95)

    from collections import Counter
    result.models = dict(Counter(f.model for f in facts).most_common())
    reasons = Counter(f.finish_reason or "unknown" for f in facts)
    result.finish_reasons = dict(reasons.most_common())
    result.truncated = reasons.get("length", 0)
    result.tool_call_rate = round(
        sum(1 for f in facts if f.tools_called) / len(facts), 3)
    result.description = _describe(result, len(facts))
    return result
