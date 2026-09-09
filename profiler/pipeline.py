"""Orchestration: CSV in, profile out.

Scope note. This pipeline **profiles**; it does not optimise. The optimisation layer
(efficiency/security auditors, cache rewriting, prompt compression) was removed deliberately:
findings attached to a grouping we could not yet trust are worse than no findings, because they
are confidently wrong. Optimisation returns once the profile half is solid.

Two stages remain that use an LLM, and each earns it:

  label_node   names what a callsite is for. No deterministic method can say "this is the
               planner".
  judge        checks whether the profile we produced is TRUE of its own requests. Production
               has no answer key, so something has to read the evidence.

The rule the design rests on: **LLM output never silently influences a deterministic
measurement.** Judgments are applied through explicit calls (`merge_nodes`, `promote_residual`)
and recorded in the audit trail, so a reader can always tell what was measured and what was
decided. Enforced by `tests/test_separation.py`.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from profiler.audit import Audit
from profiler.chain import ChainReport, reconstruct
from profiler.config import Settings
from profiler.identity import (GroupingResult, Node, group_app, identity_text, merge_nodes,
                               promote_residual)
from profiler.llm import adjudicate, judge
from profiler.llm.client import LLMClient
from profiler.load import Call, LoadReport, group_by_app, load_calls
from profiler.metrics import NodeMetrics, compute, prose_document
from profiler.registry import Registry
from profiler.segment import Segmentation, segment
from profiler.single import analyse_call, summarise


@dataclass
class NodeResult:
    node: Node
    metrics: NodeMetrics
    segmentation: Segmentation
    label: Optional[Dict[str, Any]] = None
    verdict: Optional[judge.NodeVerdict] = None
    history: Optional[Dict[str, Any]] = None      # what the registry knew about this callsite

    def as_dict(self) -> Dict[str, Any]:
        return {
            **self.node.as_dict(),
            "label": self.label,
            "metrics": self.metrics.as_dict(),
            "segmentation": self.segmentation.as_dict(),
            "template": self.node.template[:80],
            "deterministic_findings": self.metrics.findings,
            "judge": self.verdict.as_dict() if self.verdict else None,
            "history": self.history,
        }


@dataclass
class AppResult:
    app_id: str
    calls: int
    single_call: Dict[str, Any]
    nodes: List[NodeResult]
    residual_calls: int
    min_size: int
    df_histogram: Dict[str, int]
    ambiguous_pairs: int
    chain: Optional[ChainReport] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "app_id": self.app_id, "calls": self.calls, "min_node_size": self.min_size,
            "residual_calls": self.residual_calls,
            "ambiguous_pairs": self.ambiguous_pairs,
            "line_frequency_histogram": self.df_histogram,
            "single_call": self.single_call,
            "runs": self.chain.as_dict() if self.chain else None,
            "nodes": [n.as_dict() for n in self.nodes],
        }


@dataclass
class RunResult:
    run_id: str
    load_report: Dict[str, Any]
    apps: List[AppResult]
    judge_summary: Dict[str, Any]
    gate_passed: bool
    gate_reason: str
    llm_usage: Dict[str, Any]
    elapsed_s: float
    # the loaded calls, kept so `store.py` can write the requests themselves. A profile whose
    # groups cannot be opened and read has to be taken on trust.
    calls_by_app: Dict[str, List[Call]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id, "elapsed_s": self.elapsed_s,
            "input": self.load_report, "llm_usage": self.llm_usage,
            "judge": self.judge_summary,
            "gate": {"passed": self.gate_passed, "reason": self.gate_reason},
            "apps": [a.as_dict() for a in self.apps],
        }        # calls_by_app is deliberately absent: it belongs in the store, not the JSON


def run_pipeline(csv_path: str, settings: Settings, audit: Audit,
                 verbose: bool = True) -> RunResult:
    started = time.time()

    # ---- [1] load -----------------------------------------------------------------
    calls, load_report = load_calls(csv_path, settings.mapping)
    audit.step("load", **load_report.as_dict()["skipped"], loaded=load_report.loaded,
               apps=len(load_report.per_app_counts))
    if verbose:
        print(f"  loaded {load_report.loaded} calls across "
              f"{len(load_report.per_app_counts)} apps")

    client = LLMClient(settings.api_key, audit, settings.cache_dir,
                       enabled=settings.llm_enabled)
    registry = Registry(settings.registry_path) if settings.registry_path else None
    by_app = group_by_app(calls)
    app_results: List[AppResult] = []

    for app_id in sorted(by_app):
        app_calls = by_app[app_id]

        # ---- [2] single-call ------------------------------------------------------
        facts = [analyse_call(c) for c in app_calls]
        facts_by_id = {f.request_id: f for f in facts}
        single = summarise(facts)

        # ---- [3][4][5] identity ---------------------------------------------------
        grouping = group_app(
            app_id, app_calls, settings.tau, settings.min_doc_freq,
            settings.min_node_fraction, settings.min_node_floor,
            settings.ambiguous_low, settings.ambiguous_high,
        )
        by_id = {c.request_id: c for c in app_calls}
        audit.step("group", app=app_id, calls=len(app_calls), nodes=len(grouping.nodes),
                   residual=len(grouping.residual), ambiguous=len(grouping.ambiguous),
                   min_size=grouping.min_size)

        # conditional judgment: only where the deterministic step was inconclusive
        if client.enabled and grouping.ambiguous:
            merges = adjudicate.adjudicate_pairs(client, settings.model_bulk, audit,
                                                 grouping, by_id)
            if merges:
                merge_nodes(grouping, merges)
                audit.step("adjudicate_merges", app=app_id, merged=len(merges))
        if client.enabled and grouping.residual:
            promote = adjudicate.rescue_residual(client, settings.model_bulk, audit,
                                                 grouping, by_id)
            if promote:
                promote_residual(grouping, promote)
                audit.step("rescue_residual", app=app_id, promoted=len(promote))

        node_results: List[NodeResult] = []
        pending: List[tuple] = []

        for node in sorted(grouping.nodes, key=lambda n: -n.size):
            members = [by_id[r] for r in node.request_ids if r in by_id]
            if not members:
                continue

            # ---- [6] segment -------------------------------------------------------
            seg = segment([prose_document(c) for c in members])
            # ---- [7] profile -------------------------------------------------------
            metrics = compute(members, seg, facts=[facts_by_id[c.request_id]
                                                   for c in members
                                                   if c.request_id in facts_by_id])
            medoid = by_id.get(node.medoid_request_id or node.request_ids[0])
            medoid_prompt = identity_text(medoid) if medoid else ""

            result = NodeResult(node=node, metrics=metrics, segmentation=seg)
            node_results.append(result)
            if client.enabled:
                pending.append((result, metrics, medoid_prompt))

        # Naming is independent per node, so it runs in a pool. Each task writes only into
        # the NodeResult it owns; nothing is shared between workers.
        if pending:
            with ThreadPoolExecutor(max_workers=settings.llm_workers) as pool:
                list(pool.map(lambda item: _label_node(client, settings, *item), pending))

        # ---- [8] run reconstruction -------------------------------------------------
        node_of = {r: n.node.node_id for n in node_results for r in n.node.request_ids}
        chain = reconstruct(app_calls, node_of)
        audit.step("chain", app=app_id, runs=chain.runs, linked=chain.linked_calls,
                   unlinked=chain.unlinked_calls, method=chain.verdict)

        app_results.append(AppResult(
            app_id=app_id, calls=len(app_calls), single_call=single, nodes=node_results,
            residual_calls=sum(n.size for n in grouping.residual),
            min_size=grouping.min_size, df_histogram=grouping.df_histogram,
            ambiguous_pairs=len(grouping.ambiguous), chain=chain,
        ))
        if verbose:
            print(f"  {app_id:<24} {len(app_calls):>5} calls -> {len(node_results)} nodes, "
                  f"{chain.runs} runs")

    # ---- [9] per-node verification: is each profile true of its own requests? --------
    node_verdicts: List[judge.NodeVerdict] = []
    if client.enabled:
        node_verdicts = _judge_nodes(client, settings, audit, app_results, by_app)
        if node_verdicts:
            audit.step("judge_nodes", **{k: v for k, v in judge.summarise(
                node_verdicts).items() if k not in ("problems", "by_verdict")})

    # ---- [10] registry: have we seen these callsites before? -------------------------
    if registry is not None:
        seen = registry.reconcile(app_results, audit)
        audit.step("registry", **seen)
        registry.close()

    passed, reason = _gate(node_verdicts)
    audit.close()

    return RunResult(
        run_id=audit.run_id, load_report=load_report.as_dict(), apps=app_results,
        judge_summary=judge.summarise(node_verdicts),
        gate_passed=passed, gate_reason=reason,
        llm_usage=audit.usage.as_dict(), elapsed_s=round(time.time() - started, 1),
        calls_by_app=by_app,
    )


def _label_node(client: LLMClient, settings: Settings, result: NodeResult,
                metrics: NodeMetrics, medoid_prompt: str) -> None:
    """Name one callsite. Writes only into the NodeResult it was given."""
    result.label = adjudicate.label_node(
        client, settings.model_bulk, result.node.node_id, result.node.template,
        metrics.tools_called, medoid_prompt)


def _judge_nodes(client: LLMClient, settings: Settings, audit: Audit,
                 app_results: Sequence[AppResult],
                 by_app: Dict[str, List[Call]]) -> List[judge.NodeVerdict]:
    """Check every node's profile against its own sampled requests.

    Also runs an intruder test wherever the node has a nearest neighbour, so the boundary is
    tested and not only the contents.
    """
    tasks = []
    for app in app_results:
        calls_by_id = {c.request_id: c for c in by_app[app.app_id]}
        for index, node in enumerate(app.nodes):
            neighbour = next((c for c in app.nodes
                              if c is not node and c.node.request_ids), None)
            tasks.append((node, calls_by_id, neighbour, index))

    def run_one(task):
        node, calls_by_id, neighbour, index = task
        members = [calls_by_id[r] for r in node.node.request_ids if r in calls_by_id]
        if not members:
            return None
        step = max(1, len(members) // judge.MAX_SAMPLES)
        samples = members[::step][:judge.MAX_SAMPLES]

        # the bulk model is enough to check stated claims against evidence, and the bulk
        # tier has far more rate headroom than the frontier one
        verdict = judge.audit_profile(client, settings.model_bulk, node.node, node.label,
                                      node.segmentation, node.metrics, samples)
        if neighbour is not None and len(members) >= 2:
            intruder_id = neighbour.node.medoid_request_id or neighbour.node.request_ids[0]
            intruder = calls_by_id.get(intruder_id)
            medoid = calls_by_id.get(node.node.medoid_request_id or node.node.request_ids[0])
            if intruder is not None and medoid is not None:
                other = next((m for m in members if m.request_id != medoid.request_id), None)
                if other is not None:
                    verdict.intruder, reason = judge.intruder_test(
                        client, settings.model_bulk, node.node, medoid, other, intruder,
                        seed=index)
                    if verdict.intruder in ("wrong", "missed") and reason:
                        verdict.notes.append(f"boundary: {reason}")
        judge.conclude(verdict)
        node.verdict = verdict
        if verdict.verdict not in ("confirmed", "unverified"):
            audit.decision("node_verdict", subject=node.node.node_id,
                           outcome=verdict.verdict,
                           reason="; ".join(verdict.notes)[:200])
        return verdict

    with ThreadPoolExecutor(max_workers=settings.llm_workers) as pool:
        results = list(pool.map(run_one, tasks))
    return [v for v in results if v is not None]


def _gate(node_verdicts: Sequence[judge.NodeVerdict]) -> Tuple[bool, str]:
    """The run passes when most nodes were confirmed against their own requests."""
    if not node_verdicts:
        return True, "no LLM key: profile is unverified"
    stats = judge.summarise(node_verdicts)
    rate = stats["confirmed_rate"]
    bad = len(stats["problems"])
    confirmed = stats["by_verdict"].get("confirmed", 0)
    if rate is None:
        return True, "no nodes could be judged"
    if rate >= 0.8:
        return True, (f"{confirmed}/{stats['nodes_judged']} node profiles confirmed against "
                      f"their own requests" + (f"; {bad} flagged" if bad else ""))
    return False, (f"only {confirmed}/{stats['nodes_judged']} node profiles were confirmed "
                   f"({bad} flagged) - treat this report as unverified")
