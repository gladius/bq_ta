"""Conditional LLM judgment at the two points where the deterministic method is weakest.

Both fire only when there is genuine ambiguity, so on clean data they cost nothing:

  adjudicate_pairs  - pairs whose similarity lands in the band where the threshold is unreliable.
                      Measured: tau 0.6 was perfect on the spike corpus but 0.7 silently split a
                      callsite into 2 and 0.8 into 5. This replaces a brittle constant with
                      judgment on a handful of pairs.
  rescue_residual   - clusters below the size floor. A fixed floor of 20 hid 38 of 56 callsites
                      in the spike corpus; the adaptive floor helps, but small exports still
                      leave a tail worth asking about.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from profiler.audit import Audit
from profiler.identity import AmbiguousPair, GroupingResult, Node
from profiler.llm.client import LLMClient, clip
from profiler.llm.judge import request_excerpt

PAIR_SYSTEM = """You decide whether two LLM prompts come from the SAME callsite - the same place
in the same program - or from different ones.

Same callsite: the instructions, role and task are the same; only injected values differ
(user names, dates, retrieved documents, tool arguments, optional sections).
Different callsites: the model is being asked to do a different job.

A prompt that has been edited between deployments (a few lines reworded, one rule added) is
still the SAME callsite. Judge the job, not the wording."""

PAIR_SCHEMA = """Reply with JSON:
{"same_callsite": bool, "confidence": 0.0-1.0, "reason": "one sentence"}"""

RESCUE_SYSTEM = """You decide whether a small group of LLM requests is a real, distinct callsite
or just noise that should stay unassigned.

A real callsite has coherent, purposeful instructions that describe a job the program does
repeatedly - even if it appears only a handful of times in this export.
Noise is a one-off, a malformed prompt, or a fragment of another callsite.

Low call volume is NOT evidence against being a real callsite. Judge the content."""

RESCUE_SCHEMA = """Reply with JSON:
{"is_real_callsite": bool, "confidence": 0.0-1.0, "purpose": "what this callsite does",
 "reason": "one sentence"}"""


def adjudicate_pairs(
    client: LLMClient, model: str, audit: Audit, result: GroupingResult,
    call_of: Dict[str, Any], max_pairs: int = 30,
) -> List[Tuple[str, str]]:
    """Ask about ambiguous pairs. Returns node-id pairs to merge.

    Only pairs the deterministic step could not settle are sent, so the cost is proportional to
    real ambiguity rather than to row count.

    The pair is shown the WHOLE request - instructions, user template, tools, message shape -
    not the system prompt alone. Measured on the hard corpus: shown only the prompts, the model
    merged an executor and a router that share a base prompt and differ entirely in their tools,
    reporting "the prompts are identical in role and rules". They are; the tools are the whole
    difference, and it had not been shown them. Asking a judge to overrule a four-view decision
    on one view is not adjudication, it is a coin toss with a rationale attached.
    """
    if not result.ambiguous:
        return []

    node_of: Dict[str, str] = {}
    for node in result.nodes:
        for request_id in node.request_ids:
            node_of[request_id] = node.node_id

    merges: List[Tuple[str, str]] = []
    seen: set = set()
    for pair in result.ambiguous[:max_pairs]:
        left_node = node_of.get(pair.left_request_id)
        right_node = node_of.get(pair.right_request_id)
        if not left_node or not right_node or left_node == right_node:
            continue
        key = tuple(sorted((left_node, right_node)))
        if key in seen:
            continue
        seen.add(key)

        left_call = call_of.get(pair.left_request_id)
        right_call = call_of.get(pair.right_request_id)
        if left_call is None or right_call is None:
            continue
        user = (f"Similarity between these two requests: {pair.score:.3f} "
                f"(the automatic threshold is inconclusive in this range).\n"
                f"View breakdown: {pair.explain}\n\n"
                f"{request_excerpt(left_call, 'REQUEST A', 2500)}\n"
                f"{request_excerpt(right_call, 'REQUEST B', 2500)}")
        response = client.json_call("adjudicate_pair", model, PAIR_SYSTEM, user, PAIR_SCHEMA,
                                    max_tokens=400, left=left_node, right=right_node)
        if not response.ok or not isinstance(response.data, dict):
            continue

        same = bool(response.data.get("same_callsite"))
        reason = str(response.data.get("reason", ""))[:240]
        audit.decision(
            "merge_pair", subject=f"{left_node} + {right_node}",
            outcome="merged" if same else "kept separate", reason=reason,
            similarity=pair.score, confidence=response.data.get("confidence"),
        )
        if same:
            merges.append(key)
    return merges


def rescue_residual(
    client: LLMClient, model: str, audit: Audit, result: GroupingResult,
    call_of: Dict[str, Any], max_clusters: int = 20, min_members: int = 2,
) -> List[str]:
    """Ask whether below-floor clusters are real callsites. Returns node ids to promote."""
    candidates = [n for n in result.residual if n.size >= min_members]
    if not candidates:
        return []

    promoted: List[str] = []
    for node in sorted(candidates, key=lambda n: -n.size)[:max_clusters]:
        sample = call_of.get(node.medoid_request_id or node.request_ids[0])
        if sample is None:
            continue
        user = (f"This group holds {node.size} calls out of {result.min_size} needed to clear "
                f"the automatic size floor, so it was left unassigned.\n\n"
                f"INDUCED TEMPLATE:\n-----\n"
                f"{clip(chr(10).join(node.template), 2500)}\n-----\n\n"
                f"{request_excerpt(sample, 'A REPRESENTATIVE CALL', 2500)}")
        response = client.json_call("rescue_residual", model, RESCUE_SYSTEM, user,
                                    RESCUE_SCHEMA, max_tokens=500, node_id=node.node_id)
        if not response.ok or not isinstance(response.data, dict):
            continue

        real = bool(response.data.get("is_real_callsite"))
        audit.decision(
            "rescue_cluster", subject=node.node_id,
            outcome="promoted" if real else "left unassigned",
            reason=str(response.data.get("reason", ""))[:240],
            size=node.size, purpose=str(response.data.get("purpose", ""))[:160],
            confidence=response.data.get("confidence"),
        )
        if real:
            promoted.append(node.node_id)
    return promoted


def label_node(client: LLMClient, model: str, node_id: str, template: Sequence[str],
               tools: Sequence[str], sample: str) -> Optional[Dict[str, Any]]:
    """Name what a node does, so the report reads as something other than hashes."""
    system = ("You name what an LLM callsite does, from its prompt. Be specific and short. "
              "Answer as JSON: {\"name\": \"3-6 words\", \"purpose\": \"one sentence\", "
              "\"agent_role\": \"the role this call plays in a larger agent, if any\"}")
    user = (f"TOOLS AVAILABLE: {list(tools)[:20]}\n\n"
            f"STABLE TEMPLATE:\n-----\n{clip(chr(10).join(template), 5000)}\n-----\n\n"
            f"A REPRESENTATIVE CALL:\n-----\n{clip(sample, 2500)}\n-----")
    response = client.json_call("label_node", model, system, user, "", max_tokens=300,
                                node_id=node_id)
    if not response.ok or not isinstance(response.data, dict):
        return None
    return {
        "name": str(response.data.get("name", ""))[:80],
        "purpose": str(response.data.get("purpose", ""))[:300],
        "agent_role": str(response.data.get("agent_role", ""))[:160],
    }
