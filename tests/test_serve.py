"""The live browser must render every view without a running server.

The views are plain functions over a store, so they are testable directly - which is the point
of keeping the HTTP layer down to routing. What is checked here is that each view renders, that
paging is bounded at both ends, and that the two things a reader relies on are actually present:
the static/dynamic marking, and a request count that is the whole group rather than a sample.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest


from profiler.load import Call
from profiler.metrics import compute, prose_document
from profiler.pipeline import AppResult, NodeResult, RunResult
from profiler.segment import segment
from profiler.single import analyse_call
from profiler.identity import group_app
from profiler.store import write_store
from serve import Store, view_agent, view_call, view_index, view_node

PROMPT = "\n".join(["## Role", "You triage support tickets.", "Never invent an order id.",
                    "Reply with JSON only."])


def call(i: int) -> Call:
    return Call(
        request_id=f"r{i}", app_id="app",
        ts=datetime(2026, 6, 1) + timedelta(seconds=i * 10), model="gpt-4.1",
        messages=[{"role": "system", "content": f"Request {i} at 10:0{i % 10}\n{PROMPT}"},
                  {"role": "user", "content": f"Ticket {i}: refund request"}],
        tools=[], response_message={"role": "assistant", "content": "{}"},
        finish_reason="stop",
        usage={"prompt_tokens": 200, "completion_tokens": 5,
               "prompt_tokens_details": {"cached_tokens": 0}})


@pytest.fixture
def store(tmp_path):
    calls = [call(i) for i in range(30)]
    grouping = group_app("app", calls, 0.6, 2, 0.02, 3, 0.45, 0.75)
    by_id = {c.request_id: c for c in calls}
    nodes = []
    for node in grouping.nodes:
        members = [by_id[r] for r in node.request_ids]
        seg = segment([prose_document(c) for c in members])
        nodes.append(NodeResult(
            node=node, segmentation=seg,
            metrics=compute(members, seg, facts=[analyse_call(c) for c in members]),
            label={"name": "triage", "purpose": "Triage tickets"}))
    app = AppResult(app_id="app", calls=len(calls), single_call={}, nodes=nodes,
                    residual_calls=0, min_size=grouping.min_size, df_histogram={},
                    ambiguous_pairs=0)
    run = RunResult(run_id="t", load_report={"loaded": 30, "total_rows": 30}, apps=[app],
                    judge_summary={}, gate_passed=True, gate_reason="test",
                    llm_usage={"calls": 0, "cost_usd": 0.0}, elapsed_s=0.1,
                    calls_by_app={"app": calls})
    path = str(tmp_path / "calls.duckdb")
    write_store(run, path, {"app": calls}, "test.csv")
    return Store(path), nodes[0].node.node_id


def test_the_index_lists_the_agent(store):
    body = view_index(store[0]).decode()
    assert "app" in body and "/agent/app" in body


def test_the_agent_page_links_to_its_callsites(store):
    st, node_id = store
    assert f"/node/{node_id}" in view_agent(st, "app").decode()


def test_the_callsite_page_shows_the_split_and_the_whole_group(store):
    st, node_id = store
    body = view_node(st, node_id).decode()
    assert "Static vs varying" in body
    assert "Never invent an order id." in body        # a static line
    assert "30" in body                                # the whole group, not a sample


def test_a_request_renders_with_every_line_marked(store):
    st, node_id = store
    body = view_call(st, node_id, 0).decode()
    assert "shared by every request" in body
    assert "Never invent an order id." in body
    assert "Ticket 0: refund request" in body


def test_paging_is_clamped_at_both_ends(store):
    """Out-of-range must land on a real request, never a stack trace."""
    st, node_id = store
    assert b"1 / 30" in view_call(st, node_id, -5)
    assert b"30 / 30" in view_call(st, node_id, 9999)


def test_a_missing_callsite_does_not_raise(store):
    assert b"No such callsite" in view_node(store[0], "app:doesnotexist")
