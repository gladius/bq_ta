"""Cache metrics: the numbers the optimisation advice is built on."""

from __future__ import annotations

from datetime import datetime


from profiler.load import Call
from profiler.metrics import compute, request_document
from profiler.segment import common_prefix, segment

# A realistically sized prompt. The cache finding deliberately stays silent below ~50
# tokens of recoverable text, so a toy prompt would never exercise it.
STABLE = "\n".join(
    ["## Role", "You reconcile financial ledgers end to end.",
     "The task can take many steps; keep going until it is finished.", "## Rules"]
    + [f"Rule {i}: never drop an unreconciled row, and flag any variance above two "
       f"percent when reporting totals in minor units for account class {i}."
       for i in range(12)]
)


def call(i, system, tools=None):
    return Call(request_id=f"r{i}", app_id="app", ts=datetime(2026, 3, 2), model="m",
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": f"task {i}"}],
                tools=tools or [], response_message={"content": "ok"},
                finish_reason="stop", usage={})


def test_header_at_top_destroys_the_prefix_and_is_recoverable():
    """The controlled A/B from the spike: same content, different placement."""
    top = [call(i, f"User: name{i} | Session: {i:06d}\n{STABLE}") for i in range(20)]
    bottom = [call(i, f"{STABLE}\nUser: name{i} | Session: {i:06d}") for i in range(20)]

    m_top = compute(top, segment([c.messages[0]["content"] for c in top]))
    m_bottom = compute(bottom, segment([c.messages[0]["content"] for c in bottom]))

    assert m_top.cacheable_now < m_bottom.cacheable_now
    assert m_top.recoverable > 0
    assert m_bottom.recoverable == 0
    assert any("cache-prefix" in f for f in m_top.findings)


def test_tools_never_called_are_reported():
    tools = [{"type": "function", "function": {"name": "search", "description": "Search."}},
             {"type": "function", "function": {"name": "unused", "description": "Never used."}}]
    calls = [call(i, STABLE, tools) for i in range(10)]
    calls[0].response_message = {"tool_calls": [{"function": {"name": "search"}}]}
    metrics = compute(calls, segment([c.messages[0]["content"] for c in calls]))
    assert metrics.tools_never_called == ["unused"]
    assert any("never called" in f for f in metrics.findings)


def test_request_document_includes_tool_schemas():
    """Reading only the system prompt understates the cacheable prefix."""
    tools = [{"type": "function",
              "function": {"name": "t", "description": "d" * 500,
                           "parameters": {"type": "object"}}}]
    document = request_document(call(0, STABLE, tools))
    assert "TOOL t" in document
    assert len(document) > len(STABLE)


def test_common_prefix_edges():
    assert common_prefix([]) == ""
    assert common_prefix(["abc"]) == "abc"
    assert common_prefix(["abcdef", "abcxyz"]) == "abc"
    assert common_prefix(["abc", "xyz"]) == ""
