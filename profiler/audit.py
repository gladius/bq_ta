"""Run audit trail: every step, every LLM call, every decision.

A generic tracer records *calls*; what matters for debugging this pipeline is *decisions* -
which pairs were adjudicated and how, which clusters were rescued, where the verifier disagreed
with the deterministic result. So the log is domain-specific and self-contained: no service
dependency, and the run is replayable from it without paying for the LLM twice.
"""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_hits: int = 0
    cost_usd: float = 0.0
    by_purpose: Dict[str, int] = field(default_factory=dict)
    by_model: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "calls": self.calls, "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens, "cache_hits": self.cached_hits,
            "cost_usd": round(self.cost_usd, 4),
            "by_purpose": dict(sorted(self.by_purpose.items(), key=lambda kv: -kv[1])),
            "by_model": self.by_model,
        }


class Audit:
    """Append-only event log for one run."""

    def __init__(self, audit_dir: str, run_id: Optional[str] = None) -> None:
        self.run_id = run_id or f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
        os.makedirs(audit_dir, exist_ok=True)
        self.path = os.path.join(audit_dir, f"run-{self.run_id}.jsonl")
        self.usage = LLMUsage()
        self.decisions: List[Dict[str, Any]] = []
        self._started = time.time()
        self._steps: List[Dict[str, Any]] = []
        self._lock = threading.Lock()   # the log is written from worker threads
        self._handle = open(self.path, "w", encoding="utf-8")
        self.event("run_started", run_id=self.run_id)

    # -- writing ---------------------------------------------------------------------
    def event(self, kind: str, **payload: Any) -> None:
        record = {"t": round(time.time() - self._started, 3), "kind": kind}
        record.update(payload)
        with self._lock:
            if self._handle.closed:
                return
            self._handle.write(json.dumps(record, default=str) + "\n")
            self._handle.flush()

    def step(self, name: str, **payload: Any) -> None:
        entry = {"step": name, **payload}
        self._steps.append(entry)
        self.event("step", **entry)

    def llm(self, purpose: str, model: str, prompt_tokens: int, completion_tokens: int,
            cost_usd: float, cached: bool, prompt_hash: str,
            decision: Optional[Any] = None, **extra: Any) -> None:
        with self._lock:
            self._accumulate(purpose, model, prompt_tokens, completion_tokens, cost_usd, cached)
        self.event("llm_call", purpose=purpose, model=model, prompt_tokens=prompt_tokens,
                   completion_tokens=completion_tokens, cost_usd=round(cost_usd, 6),
                   from_cache=cached, prompt_hash=prompt_hash, decision=decision, **extra)

    def _accumulate(self, purpose: str, model: str, prompt_tokens: int,
                    completion_tokens: int, cost_usd: float, cached: bool) -> None:
        self.usage.calls += 1
        self.usage.prompt_tokens += prompt_tokens
        self.usage.completion_tokens += completion_tokens
        self.usage.cost_usd += cost_usd
        self.usage.cached_hits += int(cached)
        self.usage.by_purpose[purpose] = self.usage.by_purpose.get(purpose, 0) + 1
        self.usage.by_model[model] = self.usage.by_model.get(model, 0) + 1

    def decision(self, kind: str, subject: str, outcome: str, reason: str = "",
                 **extra: Any) -> None:
        """A point where something changed as a result of judgment rather than measurement."""
        record = {"kind": kind, "subject": subject, "outcome": outcome, "reason": reason}
        record.update(extra)
        with self._lock:
            self.decisions.append(record)
        # `record` carries its own "kind"; rename it so it does not collide with the event kind
        payload = {("decision_kind" if k == "kind" else k): v for k, v in record.items()}
        self.event("decision", **payload)

    def close(self) -> None:
        self.event("run_finished", elapsed_s=round(time.time() - self._started, 2),
                   llm=self.usage.as_dict())
        self._handle.close()

    # -- rendering -------------------------------------------------------------------
    def write_markdown(self, path: str, load_report: Optional[Dict[str, Any]] = None) -> None:
        lines = [
            f"# Audit — run `{self.run_id}`", "",
            f"- elapsed: {round(time.time() - self._started, 1)} s",
            f"- event log: `{os.path.basename(self.path)}`", "",
            "## LLM usage", "",
        ]
        usage = self.usage.as_dict()
        if not usage["calls"]:
            lines.append("**No LLM calls were made in this run** (no API key, or --no-llm).")
        else:
            billed = usage["calls"] - usage["cache_hits"]
            if billed == 0:
                lines.append(
                    f"- **{usage['calls']} LLM calls**, all {usage['cache_hits']} answered from "
                    f"the local response cache — the analysis ran in full, nothing was re-billed")
            else:
                lines.append(
                    f"- **{usage['calls']} LLM calls**: {billed} sent to the model, "
                    f"{usage['cache_hits']} answered from the local cache")
            lines += [
                f"- {usage['prompt_tokens']:,} prompt + {usage['completion_tokens']:,} "
                f"completion tokens billed this run",
                f"- cost this run **${usage['cost_usd']}**", "",
                "| purpose | calls |", "|---|---|",
            ]
            lines += [f"| {k} | {v} |" for k, v in usage["by_purpose"].items()]
        lines.append("")

        if load_report:
            lines += ["## Input", "", "```json",
                      json.dumps(load_report, indent=1), "```", ""]

        lines += ["## Steps", "", "| step | detail |", "|---|---|"]
        for entry in self._steps:
            detail = ", ".join(f"{k}={v}" for k, v in entry.items() if k != "step")
            lines.append(f"| {entry['step']} | {detail} |")
        lines.append("")

        if self.decisions:
            lines += ["## Decisions taken by judgment, not measurement", "",
                      "| kind | subject | outcome | reason |", "|---|---|---|---|"]
            for d in self.decisions:
                reason = str(d.get("reason", ""))[:160].replace("|", "/")
                lines.append(f"| {d['kind']} | `{d['subject']}` | {d['outcome']} | {reason} |")
            lines.append("")
        else:
            lines += ["## Decisions taken by judgment, not measurement", "",
                      "None — every result in this run is deterministic.", ""]

        with open(path, "w", encoding="utf-8") as handle:
            handle.write("\n".join(lines))
