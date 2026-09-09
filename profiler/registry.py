"""Memory across runs: have we seen this callsite before, and what did we already say?

`stable_node_id` is content-derived, so an unchanged prompt gets the same id next month. That
handles the easy half. The hard half is that the id hashes the *template*, and the template is
derived from the observed calls - so editing one line of a prompt changes the template, changes
the hash, and the callsite arrives looking brand new. In a live estate where prompts change
weekly that means re-analysing everything forever and never accumulating anything.

So exact id match is tried first, and anything left over is matched **fuzzily**, using the same
`similarity()` the grouper uses. Above tau it is the same logical callsite carrying a new
version; below it, genuinely new.

Deterministic and offline: a SQLite file, no LLM, no network. Nothing here feeds back into
grouping or metrics - it only annotates a finished profile with what was already known.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profiler.fingerprint import (DISJOINT_TOOLS_CEILING, W_FORMAT, W_PROMPT,
                                  W_SHAPE, W_TOOLS, _jaccard, h)

# A remembered callsite must look at least this similar to be treated as the same one across
# a prompt edit. Deliberately above the grouping tau: re-identification across time should be
# more conservative than clustering within one export.
REIDENTIFY_TAU = 0.70

SCHEMA = """
CREATE TABLE IF NOT EXISTS callsites (
    node_id       TEXT PRIMARY KEY,
    app_id        TEXT NOT NULL,
    name          TEXT,
    purpose       TEXT,
    template      TEXT NOT NULL,          -- JSON list of lines
    tools         TEXT NOT NULL,          -- JSON list of names
    shape         TEXT,
    first_seen    TEXT NOT NULL,
    last_seen     TEXT NOT NULL,
    runs_seen     INTEGER NOT NULL DEFAULT 1,
    version       INTEGER NOT NULL DEFAULT 1,
    calls_total   INTEGER NOT NULL DEFAULT 0,
    last_verdict  TEXT
);
CREATE TABLE IF NOT EXISTS observations (
    node_id       TEXT NOT NULL,
    run_id        TEXT NOT NULL,
    seen_at       TEXT NOT NULL,
    calls         INTEGER,
    avg_tokens    INTEGER,
    cacheable_now INTEGER,
    recoverable   INTEGER,
    cached_mean   REAL,
    verdict       TEXT,
    PRIMARY KEY (node_id, run_id)
);
CREATE TABLE IF NOT EXISTS versions (
    node_id       TEXT NOT NULL,
    version       INTEGER NOT NULL,
    changed_at    TEXT NOT NULL,
    prior_node_id TEXT,
    similarity    REAL,
    lines_added   INTEGER,
    lines_removed INTEGER,
    PRIMARY KEY (node_id, version)
);
CREATE INDEX IF NOT EXISTS callsites_app ON callsites(app_id);
"""


@dataclass
class Known:
    """One remembered callsite, as loaded from disk."""

    node_id: str
    app_id: str
    name: Optional[str]
    purpose: Optional[str]
    template: List[str]
    tools: List[str]
    shape: Optional[str]
    first_seen: str
    last_seen: str
    runs_seen: int
    version: int
    calls_total: int
    last_verdict: Optional[str]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def template_similarity(a_template: Sequence[str], a_tools: Sequence[str], a_shape: str,
                        b_template: Sequence[str], b_tools: Sequence[str],
                        b_shape: str) -> float:
    """The same weighted blend the grouper uses, over stored templates rather than raw calls.

    Weights are renormalised over the applicable views, so a callsite with no tools on either
    side is not dragged down for having none.
    """
    left, right = frozenset(h(l) for l in a_template), frozenset(h(l) for l in b_template)
    prompt = _jaccard(left, right)
    shape = 1.0 if a_shape == b_shape else 0.0

    weights = {"prompt": W_PROMPT, "shape": W_SHAPE + W_FORMAT}
    tools = 0.0
    if a_tools or b_tools:
        tools = _jaccard(frozenset(a_tools), frozenset(b_tools))
        weights["tools"] = W_TOOLS

    total = sum(weights.values()) or 1.0
    score = (prompt * weights["prompt"] + tools * weights.get("tools", 0.0)
             + shape * weights["shape"]) / total
    # Same veto as the grouper: a shared template with no shared tools is two callsites, so
    # re-identification must not quietly fold one into the other's history.
    if a_tools and b_tools and tools == 0.0:
        score = min(score, DISJOINT_TOOLS_CEILING)
    return score


def line_delta(old: Sequence[str], new: Sequence[str]) -> Tuple[int, int]:
    old_set, new_set = set(old), set(new)
    return len(new_set - old_set), len(old_set - new_set)


class Registry:
    """A small SQLite file remembering every callsite we have profiled."""

    def __init__(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.path = path
        self._con = sqlite3.connect(path)
        self._con.executescript(SCHEMA)
        self._con.commit()

    def close(self) -> None:
        self._con.commit()
        self._con.close()

    # -- reading -----------------------------------------------------------------------

    def known_for_app(self, app_id: str) -> List[Known]:
        rows = self._con.execute(
            "SELECT node_id, app_id, name, purpose, template, tools, shape, first_seen, "
            "last_seen, runs_seen, version, calls_total, last_verdict "
            "FROM callsites WHERE app_id = ?", (app_id,)).fetchall()
        return [
            Known(node_id=r[0], app_id=r[1], name=r[2], purpose=r[3],
                  template=json.loads(r[4]), tools=json.loads(r[5]), shape=r[6],
                  first_seen=r[7], last_seen=r[8], runs_seen=r[9], version=r[10],
                  calls_total=r[11], last_verdict=r[12])
            for r in rows
        ]

    def history_of(self, node_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        rows = self._con.execute(
            "SELECT run_id, seen_at, calls, avg_tokens, cacheable_now, recoverable, "
            "cached_mean, verdict FROM observations WHERE node_id = ? "
            "ORDER BY seen_at DESC LIMIT ?", (node_id, limit)).fetchall()
        keys = ("run_id", "seen_at", "calls", "avg_tokens", "cacheable_now", "recoverable",
                "cached_mean", "verdict")
        return [dict(zip(keys, r)) for r in rows]

    # -- writing -----------------------------------------------------------------------

    def reconcile(self, app_results: Sequence[Any], audit: Any = None) -> Dict[str, int]:
        """Match this run's nodes against memory, then record what we saw.

        Returns counts for the audit trail. Every node comes back annotated via
        `NodeResult.history` with one of:

          new       never seen before
          seen      same id as a previous run - the prompt has not changed
          changed   matched an older callsite fuzzily - the prompt was edited
        """
        counts = {"new": 0, "seen": 0, "changed": 0, "retired": 0}
        run_id = getattr(audit, "run_id", "") if audit is not None else ""

        for app in app_results:
            remembered = {k.node_id: k for k in self.known_for_app(app.app_id)}
            claimed: set = set()

            for node in app.nodes:
                node_id = node.node.node_id
                tools = sorted(node.metrics.tools_called)
                shape = node.node.signature.get("shape") or ""
                exact = remembered.get(node_id)

                if exact is not None:
                    node.history = self._seen_again(exact, node, run_id)
                    counts["seen"] += 1
                    claimed.add(node_id)
                    continue

                prior, score = self._closest(
                    [k for k in remembered.values() if k.node_id not in claimed],
                    node.node.template, tools, shape)

                if prior is not None and score >= REIDENTIFY_TAU:
                    node.history = self._version_bump(prior, node, run_id, score)
                    counts["changed"] += 1
                    claimed.add(prior.node_id)
                    if audit is not None:
                        audit.decision(
                            "callsite_changed", subject=node_id,
                            outcome=f"v{node.history['version']}",
                            reason=(f"matched {prior.node_id} at {score:.2f}; "
                                    f"+{node.history['lines_added']} "
                                    f"-{node.history['lines_removed']} lines"))
                else:
                    node.history = self._insert(node, run_id, tools, shape)
                    counts["new"] += 1

            counts["retired"] += len(set(remembered) - claimed)
        self._con.commit()
        return counts

    # -- internals ---------------------------------------------------------------------

    @staticmethod
    def _closest(candidates: Sequence[Known], template: Sequence[str], tools: Sequence[str],
                 shape: str) -> Tuple[Optional[Known], float]:
        best, best_score = None, 0.0
        for known in candidates:
            score = template_similarity(template, tools, shape,
                                        known.template, known.tools, known.shape or "")
            if score > best_score:
                best, best_score = known, score
        return best, best_score

    def _observe(self, node_id: str, node: Any, run_id: str) -> None:
        m = node.metrics
        self._con.execute(
            "INSERT OR REPLACE INTO observations (node_id, run_id, seen_at, calls, "
            "avg_tokens, cacheable_now, recoverable, cached_mean, verdict) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (node_id, run_id, _now(), m.calls, m.avg_request_tokens, m.cacheable_now,
             m.recoverable, m.reported_cached_mean,
             node.verdict.verdict if node.verdict else None))

    def _insert(self, node: Any, run_id: str, tools: List[str], shape: str) -> Dict[str, Any]:
        node_id = node.node.node_id
        label = node.label or {}
        self._con.execute(
            "INSERT OR REPLACE INTO callsites (node_id, app_id, name, purpose, template, "
            "tools, shape, first_seen, last_seen, runs_seen, version, calls_total, "
            "last_verdict) VALUES (?,?,?,?,?,?,?,?,?,1,1,?,?)",
            (node_id, node.node.app_id, label.get("name"), label.get("purpose"),
             json.dumps(node.node.template), json.dumps(tools), shape, _now(), _now(),
             node.metrics.calls, node.verdict.verdict if node.verdict else None))
        self._observe(node_id, node, run_id)
        return {"status": "new", "version": 1, "runs_seen": 1,
                "first_seen": _now(), "previous": []}

    def _seen_again(self, known: Known, node: Any, run_id: str) -> Dict[str, Any]:
        self._con.execute(
            "UPDATE callsites SET last_seen = ?, runs_seen = runs_seen + 1, "
            "calls_total = calls_total + ?, last_verdict = ? WHERE node_id = ?",
            (_now(), node.metrics.calls,
             node.verdict.verdict if node.verdict else None, known.node_id))
        self._observe(known.node_id, node, run_id)
        return {"status": "seen", "version": known.version,
                "runs_seen": known.runs_seen + 1, "first_seen": known.first_seen,
                "name": known.name, "purpose": known.purpose,
                "previous": self.history_of(known.node_id, 5)}

    def _version_bump(self, prior: Known, node: Any, run_id: str,
                      score: float) -> Dict[str, Any]:
        """Same logical callsite, edited prompt: carry the history to the new id."""
        node_id = node.node.node_id
        version = prior.version + 1
        added, removed = line_delta(prior.template, node.node.template)
        label = node.label or {}

        self._con.execute(
            "INSERT OR REPLACE INTO callsites (node_id, app_id, name, purpose, template, "
            "tools, shape, first_seen, last_seen, runs_seen, version, calls_total, "
            "last_verdict) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (node_id, node.node.app_id, label.get("name") or prior.name,
             label.get("purpose") or prior.purpose, json.dumps(node.node.template),
             json.dumps(sorted(node.metrics.tools_called)),
             node.node.signature.get("shape") or prior.shape, prior.first_seen, _now(),
             prior.runs_seen + 1, version, prior.calls_total + node.metrics.calls,
             node.verdict.verdict if node.verdict else None))
        self._con.execute(
            "INSERT OR REPLACE INTO versions (node_id, version, changed_at, prior_node_id, "
            "similarity, lines_added, lines_removed) VALUES (?,?,?,?,?,?,?)",
            (node_id, version, _now(), prior.node_id, round(score, 4), added, removed))
        # carry the older id's observations forward under the new id so the series is unbroken
        self._con.execute(
            "UPDATE OR REPLACE observations SET node_id = ? WHERE node_id = ?", (node_id, prior.node_id))
        self._con.execute("DELETE FROM callsites WHERE node_id = ?", (prior.node_id,))
        self._observe(node_id, node, run_id)

        return {"status": "changed", "version": version, "runs_seen": prior.runs_seen + 1,
                "first_seen": prior.first_seen, "prior_node_id": prior.node_id,
                "similarity": round(score, 3), "lines_added": added,
                "lines_removed": removed, "name": prior.name, "purpose": prior.purpose,
                "previous": self.history_of(node_id, 5)}
