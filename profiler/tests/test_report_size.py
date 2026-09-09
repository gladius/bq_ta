"""A page must stay openable however large the requests behind it are.

Two independent ways a request can be huge, and the page has to survive both:

  one enormous line     a 200k-character retrieved document or contract on a single line
  very many lines       a 150k-token contract as 4,000 short lines

The first is capped by LINE_CAP, the second by MSG_LINES, and only the first was ever exercised
- the corpus generator emits single-line blobs. A page that grows with the payload is a page
nobody opens, and the failure is silent: it renders, it is just unusable.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from profiler.load import Call
from profiler.metrics import compute, prose_document
from profiler.pipeline import AppResult, NodeResult, RunResult
from profiler.report_html import (LINE_CAP, MAX_SAMPLES, MSG_LINES,
                                  render_agent, samples_per_node)
from profiler.segment import segment
from profiler.single import analyse_call
from profiler.identity import group_app

TAU, MIN_DF, FRACTION, FLOOR, LOW, HIGH = 0.6, 2, 0.02, 3, 0.45, 0.75
PROMPT = "\n".join(["## Role", "You review commercial contracts.", "Work clause by clause.",
                    "Quote the clause you rely on.", "Return findings as a table."])


def call(i: int, user: str) -> Call:
    return Call(
        request_id=f"r{i}", app_id="app",
        ts=datetime(2026, 6, 1) + timedelta(seconds=i * 30), model="gpt-4.1",
        messages=[{"role": "system", "content": PROMPT},
                  {"role": "user", "content": user}],
        tools=[], response_message={"role": "assistant", "content": "| clause | risk |"},
        finish_reason="stop",
        usage={"prompt_tokens": len(user) // 4, "completion_tokens": 8,
               "prompt_tokens_details": {"cached_tokens": 0}})


def build_page(calls):
    grouping = group_app("app", calls, TAU, MIN_DF, FRACTION, FLOOR, LOW, HIGH)
    by_id = {c.request_id: c for c in calls}
    nodes = []
    for node in grouping.nodes:
        members = [by_id[r] for r in node.request_ids]
        seg = segment([prose_document(c) for c in members])
        nodes.append(NodeResult(
            node=node, segmentation=seg,
            metrics=compute(members, seg, facts=[analyse_call(c) for c in members])))
    app = AppResult(app_id="app", calls=len(calls), single_call={}, nodes=nodes,
                    residual_calls=0, min_size=grouping.min_size,
                    df_histogram={}, ambiguous_pairs=0)
    run = RunResult(run_id="t", load_report={"loaded": len(calls), "total_rows": len(calls)},
                    apps=[app], judge_summary={}, gate_passed=True, gate_reason="test",
                    llm_usage={"calls": 0, "cost_usd": 0.0}, elapsed_s=0.1,
                    calls_by_app={"app": calls})
    return render_agent(run, app)


def test_one_enormous_line_does_not_grow_the_page():
    """A 200k-character contract on a single line."""
    calls = [call(i, f"CONTRACT {i}:\n" + ("clause text " * 18_000)) for i in range(12)]
    page = build_page(calls)
    assert len(page) < 400_000, f"page is {len(page):,} chars for 12 x 200k-char requests"
    assert "chars]" in page, "the cut must say how much was cut, not hide it"


def test_very_many_short_lines_do_not_grow_the_page():
    """The path the corpus never exercised: a contract as 4,000 short lines."""
    calls = [call(i, "CONTRACT %d:\n%s" % (i, "\n".join(
        f"Clause {n}.{i}: the parties agree to settlement within two business days."
        for n in range(4000)))) for i in range(12)]
    page = build_page(calls)
    assert len(page) < 400_000, f"page is {len(page):,} chars for 12 x 4,000-line requests"


def test_the_caps_are_what_bounds_it():
    """Stated as a test so raising a cap has to be a deliberate act, not a drifting default."""
    assert LINE_CAP <= 400 and MSG_LINES <= 60 and MAX_SAMPLES <= 25


def test_the_sample_budget_shrinks_as_callsites_multiply():
    """Three callsites can afford twenty requests each; thirty cannot."""
    assert samples_per_node(3) > samples_per_node(31)
    assert samples_per_node(500) >= 3       # never nothing to look at


@pytest.mark.parametrize("n_calls", [12, 400])
def test_the_page_does_not_grow_with_the_number_of_calls(n_calls):
    """Only a sample is embedded, so 400 calls must cost what 12 cost."""
    calls = [call(i, f"CONTRACT {i}:\n" + ("clause text " * 500)) for i in range(n_calls)]
    page = build_page(calls)
    assert len(page) < 300_000, f"page is {len(page):,} chars for {n_calls} calls"
