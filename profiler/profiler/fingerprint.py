"""What makes two requests "the same callsite".

The previous answer was: the system prompt, and nothing else. That is one signal wide, and it
fails in exactly the shapes a dynamic production estate is full of:

  - a short generic system prompt with the real work in the user turn  -> everything merges
  - no system prompt at all                                            -> fragments into singletons
  - one base template with different task sections                     -> merges (proven: two
    callsites in the fixture collide on an identical hash)
  - the same prompt with different tool sets (router vs executor)      -> merges

Tools, response format and message shape were all computed and then thrown away. They were only
ever tested as a *hard bucket key*, which shattered a callsite under per-call tool injection;
that failure was allowed to rule out using them as a *soft signal*, which is a different thing.

So identity is now a weighted blend of four independent views of the request. Each is scored
0..1 and each can be missing without dragging the result to zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import xxhash

from profiler.load import Call
from profiler.single import text_of

# How much each view counts. Prompt text dominates because when it is distinctive it is the
# strongest evidence; the others carry the cases where it is not. Weights need not sum to 1 -
# `similarity` renormalises over whichever views apply to the pair in hand.
W_PROMPT = 0.60
W_USER = 0.20
W_TOOLS = 0.25
W_SHAPE = 0.10
W_FORMAT = 0.05

# A prompt this short cannot carry identity on its own ("You are a helpful assistant.").
# Counted AFTER noise filtering - see `effective_weak`.
WEAK_PROMPT_LINES = 3

# When a DISCRIMINATING view - the tool set, or the stable part of the user turn - is present
# on both sides and they share nothing at all, that is affirmative evidence of two different
# jobs, not merely a missing signal. Weighted averaging cannot express this: a matching prompt
# outvotes a total disagreement elsewhere and the pair merges anyway.
#
# Both cases were measured:
#   router vs executor, same base prompt, disjoint tools
#       0.60(prompt) + 0.00(tools) + 0.10(shape) + 0.05(format) = 0.75
#   summarise vs translate, same generic prompt, disjoint user templates
#       (0.60 + 0.00(user) + 0.10 + 0.05) / 0.95                = 0.79
# Both sail past tau=0.6 and merge. Putting tools in the node id does not help either: the
# merge happens first, so only one id is ever minted.
#
# The cap lands inside the adjudication band (0.45-0.75) on purpose. An agent that injects a
# different tool per call produces disjoint sets within ONE genuine callsite - the failure that
# ruled tools out as a hard bucket key. So the deterministic layer says "probably different"
# and leaves the LLM adjudicator room to overrule it on the evidence.
DISJOINT_VIEW_CEILING = 0.50
DISJOINT_TOOLS_CEILING = DISJOINT_VIEW_CEILING       # kept: callers outside this module
MAX_IDENTITY_CHARS = 200_000     # scale guard: bound the text any one call contributes


def h(text: str) -> int:
    return xxhash.xxh64(text.encode()).intdigest()


def lines_of(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").split("\n") if ln.strip()]


def role_shape(call: Call) -> str:
    """The message pattern, collapsed so run length does not matter.

    'SUATATAT' and 'SUATAT' are the same shape of callsite at different depths, so repeated
    groups collapse: both become 'SU(AT)'.
    """
    chars = {"system": "S", "developer": "S", "user": "U", "assistant": "A", "tool": "T"}
    seq = "".join(chars.get(m.get("role") or "", "?") for m in call.messages)
    seq = re.sub(r"(AT)+", "(AT)", seq)
    seq = re.sub(r"(UA)+", "(UA)", seq)
    seq = re.sub(r"A{2,}", "A", seq)
    seq = re.sub(r"T{2,}", "T", seq)
    return seq


def first_user_text(call: Call) -> str:
    """The opening user turn, which in most frameworks is itself a template."""
    return next((text_of(m.get("content")) for m in call.messages
                 if m.get("role") == "user"), "")[:MAX_IDENTITY_CHARS]


def prompt_text(call: Call) -> str:
    """The instruction surface: system messages, or the opening user turn when there is none."""
    system = "\n".join(
        text_of(m.get("content")) for m in call.messages
        if m.get("role") in ("system", "developer")
    ).strip()
    if system:
        return system[:MAX_IDENTITY_CHARS]
    return first_user_text(call)


def user_template_text(call: Call) -> str:
    """The user turn treated as its own template surface - empty when it IS `prompt_text`.

    A user turn is nearly always assembled from a template ("Context: {docs}\\n\\nQuestion:
    {q}"), so its stable lines identify a callsite just as the system prompt's do. Previously
    it was consulted only as a fallback when no system message existed, which threw the signal
    away for every agent that has both. Returning "" when there is no system message avoids
    counting the same text under two views.
    """
    has_system = any(m.get("role") in ("system", "developer") for m in call.messages)
    return first_user_text(call) if has_system else ""


def tool_names(call: Call) -> FrozenSet[str]:
    return frozenset(
        (t.get("function") or {}).get("name", "") for t in call.tools
        if isinstance(t, dict) and (t.get("function") or {}).get("name")
    )


@dataclass
class Fingerprint:
    """Every view of one request that identity is allowed to use."""

    request_id: str
    prompt_lines: FrozenSet[int]
    ordered_lines: List[int]
    tools: FrozenSet[str]
    shape: str
    response_format: Optional[str]
    has_system: bool
    prompt_line_count: int
    user_lines: FrozenSet[int] = frozenset()
    ordered_user_lines: List[int] = field(default_factory=list)

    @property
    def weak_prompt(self) -> bool:
        """Raw thinness, before noise filtering. Prefer `effective_weak` when df is known."""
        return self.prompt_line_count <= WEAK_PROMPT_LINES


def effective_weak(f: Fingerprint, filtered: Optional[Dict[str, FrozenSet[int]]]) -> bool:
    """Is this prompt too thin to identify anything, AFTER noise filtering?

    The bug this replaces: weakness was judged on the raw line count, so a prompt of one static
    line plus forty per-call lines was never flagged weak - even though noise filtering reduces
    its comparison key to that single line, where Jaccard can only return 1.0 or 0.0 and every
    callsite sharing the line merges. Weakness is a property of the key we actually compare, so
    it has to be measured there.
    """
    if filtered is None:
        return f.weak_prompt
    return len(filtered.get(f.request_id, f.prompt_lines)) <= WEAK_PROMPT_LINES


def build(call: Call) -> Fingerprint:
    ordered = [h(ln) for ln in lines_of(prompt_text(call))]
    ordered_user = [h(ln) for ln in lines_of(user_template_text(call))]
    return Fingerprint(
        request_id=call.request_id,
        prompt_lines=frozenset(ordered),
        ordered_lines=ordered,
        tools=tool_names(call),
        shape=role_shape(call),
        response_format=(call.response_format or {}).get("type"),
        has_system=any(m.get("role") in ("system", "developer") for m in call.messages),
        prompt_line_count=len(set(ordered)),
        user_lines=frozenset(ordered_user),
        ordered_user_lines=ordered_user,
    )


def _jaccard(a: FrozenSet, b: FrozenSet) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


@dataclass
class Similarity:
    """A score plus the reason for it, so a human can see why two calls did or did not join."""

    score: float
    prompt: float
    tools: float
    shape: float
    fmt: float
    user: float = 0.0
    weights: Dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        parts = [f"prompt {self.prompt:.2f}(w{self.weights.get('prompt', 0):.2f})"]
        if "user" in self.weights:
            parts.append(f"user {self.user:.2f}(w{self.weights['user']:.2f})")
        parts += [f"tools {self.tools:.2f}(w{self.weights.get('tools', 0):.2f})",
                  f"shape {self.shape:.2f}(w{self.weights.get('shape', 0):.2f})",
                  f"format {self.fmt:.2f}(w{self.weights.get('format', 0):.2f})"]
        return " ".join(parts)


def similarity(a: Fingerprint, b: Fingerprint,
               filtered: Optional[Dict[str, FrozenSet[int]]] = None,
               filtered_user: Optional[Dict[str, FrozenSet[int]]] = None) -> Similarity:
    """Weighted agreement across every view that applies to this pair.

    `filtered` and `filtered_user` supply noise-filtered line sets when the caller has them
    (each surface gets its own document frequency); without them the raw sets are used. Weights
    are renormalised over the views that actually apply, so a callsite with no tools is not
    penalised for having none.
    """
    left = filtered.get(a.request_id, a.prompt_lines) if filtered else a.prompt_lines
    right = filtered.get(b.request_id, b.prompt_lines) if filtered else b.prompt_lines

    prompt = _jaccard(left, right)
    shape = 1.0 if a.shape == b.shape else 0.0
    fmt = 1.0 if a.response_format == b.response_format else 0.0

    weights = {"prompt": W_PROMPT, "shape": W_SHAPE, "format": W_FORMAT}

    # Tools only carry evidence when at least one side declares some. Two callsites that both
    # declare none agree on nothing informative, so the view is dropped rather than scored 1.
    tools = 0.0
    if a.tools or b.tools:
        tools = _jaccard(a.tools, b.tools)
        weights["tools"] = W_TOOLS

    # The user turn is its own template surface. Applicability is decided on the FILTERED sets,
    # not the raw ones: a user turn that is entirely dynamic has no stable lines to compare, so
    # scoring it 0.0 at full weight would penalise a genuine match for carrying a payload.
    # (Same class of error as judging prompt thinness before filtering.)
    u_left = (filtered_user.get(a.request_id, a.user_lines) if filtered_user else a.user_lines)
    u_right = (filtered_user.get(b.request_id, b.user_lines) if filtered_user else b.user_lines)
    user = 0.0
    if u_left or u_right:
        user = _jaccard(u_left, u_right)
        weights["user"] = W_USER

    # When the instruction surface is too thin to identify anything - measured on the filtered
    # key, not the raw one - lean on the other views instead of letting a single shared line
    # merge unrelated callsites.
    if effective_weak(a, filtered) or effective_weak(b, filtered):
        weights["prompt"] = W_PROMPT * 0.4
        if "tools" in weights:
            weights["tools"] = W_TOOLS * 1.8
        if "user" in weights:
            weights["user"] = W_USER * 2.0
        weights["shape"] = W_SHAPE * 2.0
        weights["format"] = W_FORMAT * 2.0

    total = sum(weights.values()) or 1.0
    score = (
        prompt * weights.get("prompt", 0.0)
        + user * weights.get("user", 0.0)
        + tools * weights.get("tools", 0.0)
        + shape * weights.get("shape", 0.0)
        + fmt * weights.get("format", 0.0)
    ) / total
    if (a.tools and b.tools and tools == 0.0) or (u_left and u_right and user == 0.0):
        score = min(score, DISJOINT_VIEW_CEILING)
    return Similarity(score=score, prompt=prompt, tools=tools, shape=shape, fmt=fmt,
                      user=user, weights={k: v / total for k, v in weights.items()})


def identity_signature(fingerprints: Sequence[Fingerprint]) -> Dict[str, Any]:
    """What identity had to work with for this group - reported so the reader can judge it."""
    if not fingerprints:
        return {}
    from collections import Counter

    shapes = Counter(f.shape for f in fingerprints)
    return {
        "with_system_prompt": sum(1 for f in fingerprints if f.has_system),
        "with_user_template": sum(1 for f in fingerprints if f.user_lines),
        "weak_prompts": sum(1 for f in fingerprints if f.weak_prompt),
        "median_prompt_lines": sorted(f.prompt_line_count for f in fingerprints)[
            len(fingerprints) // 2],
        "distinct_tool_sets": len({f.tools for f in fingerprints}),
        "shape": shapes.most_common(1)[0][0],
        "shapes": [{"shape": s, "calls": n} for s, n in shapes.most_common(6)],
        "distinct_shapes": len(shapes),
        "distinct_formats": len({f.response_format for f in fingerprints}),
    }
