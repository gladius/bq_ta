"""Per-node verification: is the profile we produced actually true of these requests?

Two checks per node, both grounded in real sampled requests rather than a leading question.

  audit_profile  We state our claims - this is the template, this is what varies, this is the
                 purpose, these calls are one callsite - and ask the model to check each one
                 against several real requests. Falsifiable: it can point at the line we got
                 wrong.

  intruder       medoid + a genuine member + one request from the NEAREST competing node,
                 shuffled and unlabelled. If the model cannot pick the outsider, the boundary
                 between those two nodes is not real. Run only where a neighbour exists.

The earlier version sampled nine nodes out of fifty-six and asked "do these belong together?",
which invites agreement and left most nodes unexamined. Every node is checked here.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profiler.llm.client import LLMClient, clip

MAX_PROMPT_EXCERPT = 1400
MAX_SAMPLES = 3


def request_excerpt(call: Any, label: str, prompt_chars: int = MAX_PROMPT_EXCERPT) -> str:
    """The whole request condensed - not the system prompt alone."""
    from profiler.fingerprint import prompt_text, role_shape, tool_names
    from profiler.single import text_of

    user = next((text_of(m.get("content")) for m in call.messages
                 if m.get("role") == "user"), "")
    tools = sorted(tool_names(call))
    return (f"--- {label} ---\n"
            f"tools: {tools if tools else 'none'}\n"
            f"message shape: {role_shape(call)}\n"
            f"instructions:\n{clip(prompt_text(call), prompt_chars)}\n"
            f"first user turn:\n{clip(user, 400)}\n")


@dataclass
class NodeVerdict:
    node_id: str
    profile_accurate: Optional[bool] = None
    template_errors: List[str] = field(default_factory=list)
    missed_dynamic: List[str] = field(default_factory=list)
    purpose_matches: Optional[bool] = None
    one_callsite: Optional[bool] = None
    intruder: Optional[str] = None          # correct | wrong | missed | n/a
    verdict: str = "unverified"             # confirmed | profile_wrong | impure | boundary_weak
    confidence: float = 0.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


AUDIT_SYSTEM = """You check whether an automated profile of an LLM callsite is TRUE of the real
requests it came from.

You are given a set of CLAIMS the tool made, then several REAL REQUESTS it grouped together.
Check each claim against the evidence. Be specific and be willing to say the tool got it wrong -
that is the point of the exercise.

Definitions:
- "static" means the LINE appears, unchanged, in every one of the requests shown. A line such
  as "Document:" or "Transcript:" that introduces varying text below it IS static - the label is
  unchanged even though what follows it is not. Judge the line, not the section it heads.
- claims about the system prompt must be checked against the system prompt, and claims about the
  user turn against the user turn. They are shown separately for that reason.
- "dynamic" means it changes between requests, or is present in some and absent in others.
- message shape (SU, SU(AT)T, ...) VARIES between steps of one agent loop. Different
  shapes do NOT mean different callsites; ignore shape differences.
- one callsite means the same place in a program doing the same job. Different injected values
  (names, dates, retrieved text, tool arguments, optional sections) are still ONE callsite.
  A different job, or different tools for a different purpose, means different callsites."""

AUDIT_SCHEMA = """Reply with JSON:
{"template_accurate": bool,
 "wrongly_static": ["lines claimed static that actually vary"],
 "missed_dynamic": ["things that vary which the tool did not list"],
 "purpose_matches": bool,
 "one_callsite": bool,
 "confidence": 0.0-1.0,
 "summary": "one or two sentences on what, if anything, is wrong"}"""


def audit_profile(client: LLMClient, model: str, node: Any, label: Optional[Dict[str, Any]],
                  segmentation: Any, metrics: Any, samples: Sequence[Any]) -> NodeVerdict:
    """State the profile's claims; check them against real requests."""
    verdict = NodeVerdict(node_id=node.node_id)
    if not samples:
        return verdict

    dynamic = "\n".join(f"    - {f.template}" for f in segmentation.fields[:8]) or "    (none)"
    # Show the FULL template. Truncating it made the judge report 'you omitted these
    # lines', which was true of the excerpt and false of the profile - the harness was on
    # trial, not the profiling.
    template = "\n".join(f"    {ln}" for ln in node.template) or "    (empty)"
    if len(template) > 5000:
        template = template[:5000] + f"\n    ...[{len(node.template)} lines total]"
    purpose = (label or {}).get("purpose") or (label or {}).get("name") or "(not named)"

    claims = (
        f"CLAIMS MADE BY THE TOOL\n"
        f"  purpose: {purpose}\n"
        f"  these {metrics.calls} calls are ONE callsite\n"
        f"  cacheable prefix: {metrics.cacheable_now} tokens of {metrics.avg_request_tokens}\n"
        f"  TEMPLATE - claimed identical on every call:\n{template}\n"
        f"  CLAIMED TO VARY PER CALL:\n{dynamic}\n"
    )
    evidence = "\n".join(
        request_excerpt(c, f"REQUEST {i + 1}") for i, c in enumerate(samples[:MAX_SAMPLES]))

    result = client.json_call("judge_profile", model, AUDIT_SYSTEM,
                              f"{claims}\nREAL REQUESTS FROM THIS CALLSITE:\n{evidence}",
                              AUDIT_SCHEMA, max_tokens=1200, node_id=node.node_id)
    if not result.ok or not isinstance(result.data, dict):
        return verdict

    data = result.data
    verdict.profile_accurate = bool(data.get("template_accurate"))
    verdict.template_errors = [str(x)[:160] for x in (data.get("wrongly_static") or [])][:6]
    verdict.missed_dynamic = [str(x)[:160] for x in (data.get("missed_dynamic") or [])][:6]
    verdict.purpose_matches = bool(data.get("purpose_matches"))
    verdict.one_callsite = bool(data.get("one_callsite"))
    try:
        verdict.confidence = round(float(data.get("confidence", 0.0)), 2)
    except (TypeError, ValueError):
        verdict.confidence = 0.0
    summary = str(data.get("summary", ""))[:300]
    if summary:
        verdict.notes.append(summary)
    return verdict


INTRUDER_SYSTEM = """You are shown three LLM requests. Exactly one of them MAY come from a
different callsite - a different place in a program doing a different job - than the other two.

Different injected values (names, dates, retrieved documents, tool arguments, optional sections)
do NOT make a different callsite. A different job, or a different tool set serving a different
purpose, does.

If one request is doing a different job, name it. If all three look like the same callsite,
answer "none". Do not guess."""

INTRUDER_SCHEMA = """Reply with JSON:
{"odd_one": "A"|"B"|"C"|"none", "confidence": 0.0-1.0, "reason": "one sentence"}"""


def intruder_test(client: LLMClient, model: str, node: Any, medoid: Any, member: Any,
                  intruder: Any, seed: int = 0) -> Tuple[Optional[str], str]:
    """Can the model separate this node from its nearest neighbour? Returns (result, reason)."""
    entries = [(medoid, "member"), (member, "member"), (intruder, "intruder")]
    random.Random(seed).shuffle(entries)
    planted = chr(65 + next(i for i, (_, kind) in enumerate(entries) if kind == "intruder"))
    body = "\n".join(request_excerpt(c, chr(65 + i), 1500) for i, (c, _) in enumerate(entries))

    result = client.json_call("judge_intruder", model, INTRUDER_SYSTEM, body, INTRUDER_SCHEMA,
                              max_tokens=350, node_id=node.node_id)
    if not result.ok or not isinstance(result.data, dict):
        return None, ""
    answer = str(result.data.get("odd_one", "none")).strip().upper()
    reason = str(result.data.get("reason", ""))[:200]
    if answer == planted:
        return "correct", reason
    if answer in ("NONE", ""):
        return "missed", reason
    return "wrong", reason


def conclude(verdict: NodeVerdict) -> NodeVerdict:
    """One word from the checks. Profile errors outrank boundary weakness."""
    if verdict.one_callsite is False:
        verdict.verdict = "impure"
    elif verdict.intruder == "wrong":
        verdict.verdict = "impure"
    elif verdict.profile_accurate is False or verdict.template_errors:
        verdict.verdict = "profile_wrong"
    elif verdict.intruder == "missed":
        verdict.verdict = "boundary_weak"
    elif verdict.profile_accurate:
        verdict.verdict = "confirmed"
    else:
        verdict.verdict = "unverified"
    return verdict


def summarise(verdicts: Sequence[NodeVerdict]) -> Dict[str, Any]:
    from collections import Counter

    counts = Counter(v.verdict for v in verdicts)
    checked = [v for v in verdicts if v.verdict != "unverified"]
    intruders = [v.intruder for v in verdicts if v.intruder in ("correct", "wrong", "missed")]
    return {
        "nodes_judged": len(verdicts),
        "by_verdict": dict(counts),
        "confirmed_rate": (round(counts.get("confirmed", 0) / len(checked), 3)
                           if checked else None),
        "boundary_tests": len(intruders),
        "boundary_correct": (round(sum(1 for i in intruders if i == "correct") / len(intruders),
                                   3) if intruders else None),
        "problems": [
            {"node_id": v.node_id, "verdict": v.verdict,
             "template_errors": v.template_errors, "missed_dynamic": v.missed_dynamic,
             "notes": v.notes}
            for v in verdicts if v.verdict not in ("confirmed", "unverified")
        ],
    }
