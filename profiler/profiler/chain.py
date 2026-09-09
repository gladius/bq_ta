"""Run reconstruction: which calls belong to the same agent run, and in what order.

The mechanism is the only one available from a gateway log. An agent loop resends its history,
so step N's message list contains step N-1's as a strict prefix. Index every request by its
message hashes, then link each request to the most recent earlier request whose messages are a
prefix of its own. Walking those links to a root yields one run.

Measured in the spike over 116,252 real pairs: precision/recall 1.000 where history is replayed.

**And it is unrecoverable where it is not.** An agent that compacts its history (0.344) or
slides a window over it (0.054) destroys the shared prefix *before the gateway sees the
request*. No algorithm gets that back - the evidence is not in the log. So this module also
reports whether the method could have worked on this app, and says so plainly rather than
handing back a confident tree built from nothing.

Two linking modes:
  strict   the whole message list must match as a prefix
  relaxed  index 0 is ignored when both sides carry a system message, so a callsite that
           rebuilds its system prompt every step can still chain

Deterministic. Reads calls only; never consults the LLM layer.
"""

from __future__ import annotations

import bisect
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import xxhash

from profiler.load import Call
from profiler.single import text_of

LINK_WINDOW_S = 30 * 60          # a parent older than this is a coincidence, not a step
RETRY_WINDOW_S = 5               # an identical request this soon after is a retry, not a step
CONTINUATION_MESSAGES = 3        # more messages than this means it cannot be a first step
ORPHAN_ALARM = 0.30              # unlinked continuations above this: history is not replayed


def _hash_message(message: Dict[str, Any]) -> int:
    """Role plus content plus tool-call identity - enough to tell two turns apart."""
    calls = message.get("tool_calls") or []
    names = ",".join(str((c.get("function") or {}).get("name", "")) for c in calls)
    material = f"{message.get('role')}|{text_of(message.get('content'))}|{names}"
    return xxhash.xxh64(material.encode()).intdigest()


@dataclass
class Req:
    """One request reduced to what linking needs."""

    request_id: str
    ts: Any
    msg_hashes: Tuple[int, ...]
    has_system: bool
    position: int = -1

    def key(self, relaxed: bool) -> Tuple[int, ...]:
        if relaxed and self.has_system:
            return self.msg_hashes[1:]
        return self.msg_hashes

    @property
    def is_continuation(self) -> bool:
        """Too many messages to be the opening call of a run."""
        return len(self.msg_hashes) >= CONTINUATION_MESSAGES


class _Index:
    """Message tuple -> the requests carrying it, ordered by position."""

    def __init__(self) -> None:
        self._by_key: Dict[Tuple[int, ...], List[Req]] = {}

    def add(self, req: Req, relaxed: bool) -> None:
        self._by_key.setdefault(req.key(relaxed), []).append(req)

    def latest_before(self, key: Tuple[int, ...], position: int) -> Optional[Req]:
        candidates = self._by_key.get(key)
        if not candidates:
            return None
        i = bisect.bisect_left([c.position for c in candidates], position)
        return candidates[i - 1] if i > 0 else None


def link(requests: Sequence[Req], relaxed: bool) -> Dict[str, Tuple[Optional[str], bool]]:
    """request_id -> (parent_request_id, is_retry). Requests must be position-ordered."""
    index = _Index()
    for req in requests:
        index.add(req, relaxed)

    parents: Dict[str, Tuple[Optional[str], bool]] = {}
    for req in reversed(requests):           # newest first
        own = req.key(relaxed)

        # an identical message list moments earlier is a retry, not a new step
        identical = index.latest_before(own, req.position)
        if identical is not None and (req.ts - identical.ts).total_seconds() <= RETRY_WINDOW_S:
            parents[req.request_id] = (identical.request_id, True)
            continue

        best: Optional[Req] = None
        for length in range(len(own) - 1, 0, -1):
            candidate = index.latest_before(own[:length], req.position)
            if candidate is None:
                continue
            if (req.ts - candidate.ts).total_seconds() > LINK_WINDOW_S:
                continue
            if best is None or candidate.position > best.position:
                best = candidate
        parents[req.request_id] = (best.request_id if best else None, False)
    return parents


def _walk(requests: Sequence[Req],
          parents: Dict[str, Tuple[Optional[str], bool]]) -> Dict[str, Tuple[str, int]]:
    """request_id -> (run_id, step).

    A parent always sits at a strictly smaller position than its child, so one forward pass in
    position order resolves everything and a cycle is impossible.
    """
    known = {r.request_id for r in requests}
    resolved: Dict[str, Tuple[str, int]] = {}
    for req in requests:
        parent = parents.get(req.request_id, (None, False))[0]
        if parent is None or parent not in known:
            resolved[req.request_id] = (req.request_id, 0)
        else:
            root, depth = resolved[parent]
            resolved[req.request_id] = (root, depth + 1)
    return resolved


@dataclass
class Run:
    """One reconstructed agent run."""

    run_id: str
    steps: int
    retries: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    node_path: List[str] = field(default_factory=list)   # callsites in the order they fired
    duration_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class ChainReport:
    """Runs for one app, plus an honest statement of whether linking could work here."""

    mode: str = "strict"
    runs: int = 0
    linked_calls: int = 0
    unlinked_calls: int = 0
    orphan_continuations: int = 0
    continuations: int = 0
    verdict: str = "reliable"          # reliable | partial | unusable | single_step
    note: str = ""
    mean_steps: float = 0.0
    max_steps: int = 0
    retries: int = 0
    mean_run_tokens: int = 0
    top_paths: List[Dict[str, Any]] = field(default_factory=list)
    longest: List[Dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _report_for(app_calls: Sequence[Call], reqs: List[Req], mode: str,
                parents: Dict[str, Tuple[Optional[str], bool]],
                node_of: Dict[str, str]) -> ChainReport:
    resolved = _walk(reqs, parents)
    by_id = {c.request_id: c for c in app_calls}

    grouped: Dict[str, List[Req]] = {}
    for req in reqs:
        grouped.setdefault(resolved[req.request_id][0], []).append(req)

    runs: List[Run] = []
    for run_id, members in grouped.items():
        members.sort(key=lambda r: resolved[r.request_id][1])
        calls = [by_id[r.request_id] for r in members if r.request_id in by_id]
        span = ((members[-1].ts - members[0].ts).total_seconds() if len(members) > 1 else 0.0)
        runs.append(Run(
            run_id=run_id,
            steps=len(members),
            retries=sum(1 for r in members if parents.get(r.request_id, (None, False))[1]),
            prompt_tokens=sum(c.prompt_tokens or 0 for c in calls),
            completion_tokens=sum(c.completion_tokens or 0 for c in calls),
            cached_tokens=sum(c.cached_tokens or 0 for c in calls),
            node_path=[node_of.get(r.request_id, "?") for r in members],
            duration_s=round(span, 1),
        ))

    linked = sum(1 for r in reqs if parents.get(r.request_id, (None, False))[0])
    continuations = [r for r in reqs if r.is_continuation]
    orphans = [r for r in continuations if not parents.get(r.request_id, (None, False))[0]]

    report = ChainReport(
        mode=mode, runs=len(runs), linked_calls=linked,
        unlinked_calls=len(reqs) - linked,
        continuations=len(continuations), orphan_continuations=len(orphans),
        mean_steps=round(sum(r.steps for r in runs) / max(len(runs), 1), 2),
        max_steps=max((r.steps for r in runs), default=0),
        retries=sum(r.retries for r in runs),
        mean_run_tokens=int(sum(r.prompt_tokens + r.completion_tokens for r in runs)
                            / max(len(runs), 1)),
    )

    # The honest part: say when the log cannot support this.
    if not continuations:
        report.verdict = "single_step"
        report.note = ("every call is a first call - this app does not replay history, so "
                       "there are no multi-step runs to reconstruct")
    else:
        orphan_rate = len(orphans) / len(continuations)
        if orphan_rate > ORPHAN_ALARM:
            report.verdict = "unusable"
            report.note = (
                f"{len(orphans)} of {len(continuations)} mid-run calls ({orphan_rate:.0%}) have "
                f"no linkable parent. The agent almost certainly compacts or windows its "
                f"history, which destroys the shared prefix before the gateway sees it. This "
                f"cannot be recovered from logs - ask the client to send a run id header.")
        elif orphan_rate > 0.05:
            report.verdict = "partial"
            report.note = (f"{orphan_rate:.0%} of mid-run calls could not be linked; treat run "
                           f"counts as a lower bound")
        else:
            report.verdict = "reliable"

    paths = Counter(" -> ".join(r.node_path[:6]) for r in runs if r.steps > 1)
    report.top_paths = [{"path": p, "runs": n} for p, n in paths.most_common(5)]
    report.longest = [r.as_dict() for r in sorted(runs, key=lambda r: -r.steps)[:3]]
    return report


def reconstruct(app_calls: Sequence[Call],
                node_of: Optional[Dict[str, str]] = None) -> ChainReport:
    """Link one app's calls into runs, choosing whichever mode links more.

    `node_of` maps request id -> callsite id, so a run can be described as the sequence of
    callsites it visited rather than as opaque request ids.
    """
    node_of = node_of or {}
    ordered = sorted(app_calls, key=lambda c: (c.ts, c.request_id))
    reqs = [
        Req(request_id=c.request_id, ts=c.ts,
            msg_hashes=tuple(_hash_message(m) for m in c.messages),
            has_system=bool(c.messages) and c.messages[0].get("role") in ("system",
                                                                          "developer"),
            position=i)
        for i, c in enumerate(ordered)
    ]
    if not reqs:
        return ChainReport(verdict="single_step", note="no calls")

    # Strict first; relaxed only if it genuinely links more, so the simpler explanation wins.
    strict = link(reqs, relaxed=False)
    relaxed = link(reqs, relaxed=True)
    n_strict = sum(1 for r in reqs if strict.get(r.request_id, (None, False))[0])
    n_relaxed = sum(1 for r in reqs if relaxed.get(r.request_id, (None, False))[0])
    if n_relaxed > n_strict:
        return _report_for(ordered, reqs, "relaxed", relaxed, node_of)
    return _report_for(ordered, reqs, "strict", strict, node_of)
