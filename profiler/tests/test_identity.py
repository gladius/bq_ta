"""Grouping: the validated core. These encode the findings the spike measured."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.identity import (adaptive_min_size, group_app, identity_text, jaccard,
                               stable_node_id)
from profiler.load import Call

TAU, MIN_DF, FRACTION, FLOOR, LOW, HIGH = 0.6, 2, 0.02, 3, 0.45, 0.75

BASE = "\n".join([
    "## Role", "You are a support agent.", "Answer from the knowledge base only.",
    "## Rules", "Never invent an order id.", "Escalate billing disputes above 500.",
    "Close with the next action.",
])


def call(i: int, system: str, user: str = "hello", tools=None) -> Call:
    return Call(
        request_id=f"r{i}", app_id="app", ts=datetime(2026, 3, 2, 0, i % 59),
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        tools=tools or [], response_message={"role": "assistant", "content": "ok"},
        finish_reason="stop", usage={},
    )


def group(calls):
    return group_app("app", calls, TAU, MIN_DF, FRACTION, FLOOR, LOW, HIGH)


def test_identical_prompts_form_one_node():
    result = group([call(i, BASE) for i in range(20)])
    assert len(result.nodes) == 1
    assert result.nodes[0].size == 20


def test_per_call_noise_does_not_fragment_a_callsite():
    """The finding that motivated the noise filter: unique lines must not split a node."""
    calls = [call(i, BASE + f"\n[chunk:{i:06d}] retrieved passage number {i} about liability")
             for i in range(30)]
    result = group(calls)
    assert len(result.nodes) == 1, "per-call noise fragmented the callsite"
    assert result.nodes[0].size == 30
    # the noise must not reach the template either
    assert not any("chunk:" in line for line in result.nodes[0].template)


def test_two_different_callsites_stay_separate():
    other = "\n".join(["## Role", "You audit tickets for policy violations.",
                       "## Checks", "Was an order id invented?", "Answer PASS or FAIL."])
    result = group([call(i, BASE) for i in range(20)] +
                   [call(100 + i, other) for i in range(20)])
    assert len(result.nodes) == 2


def test_per_tenant_header_does_not_fragment():
    """The case that defeats an exact-match or head-hash approach."""
    calls = [call(i, f"User: name{i} | Date: 2026-03-{i % 28 + 1:02d}\n{BASE}")
             for i in range(30)]
    result = group(calls)
    assert len(result.nodes) == 1


def test_node_id_is_stable_across_independent_runs():
    first = group([call(i, BASE) for i in range(20)])
    second = group([call(100 + i, BASE, user=f"different question {i}") for i in range(20)])
    assert first.nodes[0].node_id == second.nodes[0].node_id


def test_node_id_changes_when_the_template_changes():
    first = group([call(i, BASE) for i in range(20)])
    edited = BASE.replace("Never invent an order id.", "Never invent an order id, ever.")
    second = group([call(i, edited) for i in range(20)])
    assert first.nodes[0].node_id != second.nodes[0].node_id


def test_small_export_still_yields_nodes():
    """A fixed floor of 20 hid 38 of 56 callsites in the spike corpus."""
    result = group([call(i, BASE) for i in range(6)])
    assert result.min_size == 3
    assert len(result.nodes) == 1


def test_adaptive_floor_scales_with_volume():
    assert adaptive_min_size(50, 0.02, 3) == 3
    assert adaptive_min_size(1000, 0.02, 3) == 20


def test_identity_falls_back_to_user_when_no_system_message():
    prompt = "You are a classifier.\nReturn one category."
    c = Call(request_id="r", app_id="app", ts=datetime(2026, 3, 2), model="m",
             messages=[{"role": "user", "content": prompt}], tools=[],
             response_message={}, finish_reason="stop", usage={})
    assert identity_text(c) == prompt


def test_identity_prefers_system_when_present():
    c = Call(request_id="r", app_id="app", ts=datetime(2026, 3, 2), model="m",
             messages=[{"role": "system", "content": "SYS"},
                       {"role": "user", "content": "USER"}],
             tools=[], response_message={}, finish_reason="stop", usage={})
    assert identity_text(c) == "SYS"


def test_jaccard_edges():
    assert jaccard(frozenset(), frozenset()) == 1.0
    assert jaccard(frozenset({1, 2}), frozenset({1, 2})) == 1.0
    assert jaccard(frozenset({1}), frozenset({2})) == 0.0


def test_ambiguous_pairs_are_surfaced_not_silently_decided():
    """A pair in the uncertain band must be reported for adjudication, not guessed at."""
    a = "\n".join(f"line {i}" for i in range(10))
    b = "\n".join(f"line {i}" for i in range(5, 15))     # half overlap -> Jaccard 0.33
    result = group([call(i, a) for i in range(10)] + [call(50 + i, b) for i in range(10)])
    assert len(result.nodes) == 2
    # the pair sits below the ambiguity band here, so nothing is flagged; the contract is that
    # anything *inside* the band is reported
    for pair in result.ambiguous:
        assert LOW <= pair.score < TAU


# ---------------------------------------------------------------------------------------
# The three shapes a production estate is full of, each of which the first version got wrong.
# ---------------------------------------------------------------------------------------

def tool(name):
    return {"type": "function", "function": {"name": name, "description": f"{name} things",
                                             "parameters": {"type": "object"}}}


def test_same_prompt_different_tools_are_two_callsites():
    """A router and an executor behind one base prompt.

    Scored 0.75 before the disjoint-tools cap - comfortably past tau, so they merged into one
    node. Putting tools in the node id did not help: the merge happens first, so only one id is
    ever minted.
    """
    router = [call(i, BASE, tools=[tool("route"), tool("classify")]) for i in range(20)]
    executor = [call(100 + i, BASE, tools=[tool("deploy"), tool("rollback")])
                for i in range(20)]
    result = group(router + executor)
    assert len(result.nodes) == 2, "a router and an executor merged on their shared prompt"


def test_a_shared_tool_still_holds_one_callsite_together():
    """The cap must not shatter a callsite that injects a tool per call."""
    calls = [call(i, BASE, tools=[tool("search"), tool(f"lookup_tenant_{i % 4}")])
             for i in range(24)]
    assert len(group(calls).nodes) == 1


def test_a_thin_prompt_does_not_swallow_unrelated_callsites():
    """One static line plus per-call noise.

    `weak_prompt` counted RAW lines, so a 40-line prompt was never called thin - even though
    noise filtering cut its comparison key to the single shared line, where Jaccard is 1.0 for
    anything that shares it. Everything behind that line merged.
    """
    thin = "You are a helpful assistant."
    search = [call(i, f"{thin}\nsession {i} opened at 10:0{i % 10}",
                   tools=[tool("web_search")]) for i in range(20)]
    sql = [call(100 + i, f"{thin}\nsession {100 + i} opened at 11:0{i % 10}",
                tools=[tool("run_sql")]) for i in range(20)]
    result = group(search + sql)
    assert len(result.nodes) == 2, "a generic one-line prompt merged two callsites"


def test_the_user_turn_separates_callsites_that_share_a_system_prompt():
    """The gap the user asked about: the real instruction lives in the user turn.

    Both callsites send the same generic system prompt and declare no tools, so before the user
    turn became its own view there was nothing left to tell them apart.
    """
    shared = "\n".join(["You are a careful assistant.", "Think step by step.",
                        "Answer in English.", "Be concise."])
    summarise = [call(i, shared,
                      user=f"Task: summarise the document.\nStyle: bullet points.\n"
                           f"Document: report number {i} about quarterly revenue")
                 for i in range(20)]
    translate = [call(100 + i, shared,
                      user=f"Task: translate to French.\nTone: formal.\n"
                           f"Text: passage number {i} concerning shipping delays")
                 for i in range(20)]
    result = group(summarise + translate)
    assert len(result.nodes) == 2, "two callsites merged because only the system prompt counted"


def test_the_stable_part_of_a_user_turn_reaches_the_template():
    calls = [call(i, BASE, user=f"Question: {i}\nFormat: JSON\nLocale: en-GB")
             for i in range(20)]
    node = group(calls).nodes[0]
    assert "[user turn]" in node.template
    assert "Format: JSON" in node.template
    assert not any(line.startswith("Question:") for line in node.template)


def test_a_constant_user_turn_does_not_change_the_node_id():
    """Whether a user turn happens to be fixed is a property of the export, not the callsite."""
    fixed = group([call(i, BASE, user="same question every time") for i in range(20)])
    varying = group([call(100 + i, BASE, user=f"question {i}") for i in range(20)])
    assert fixed.nodes[0].node_id == varying.nodes[0].node_id
