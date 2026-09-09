"""The store exists so the profile can be checked rather than believed.

The property under test is not "rows were written" but "the static/dynamic claim is recorded
per region, line by line, with the count behind it" - because that is the claim a reader needs
to be able to disagree with.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import duckdb
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from profiler.load import Call
from profiler.metrics import compute, prose_document, region_split, region_text
from profiler.segment import segment
from profiler.single import analyse_call

TEMPLATE = ["## Role", "You triage support tickets.", "Never invent an order id.",
            "Reply with JSON only."]


def call(i, extra_system="", user=None, tools=None):
    system = "\n".join(TEMPLATE) + (f"\n{extra_system}" if extra_system else "")
    return Call(
        request_id=f"r{i}", app_id="app", ts=datetime(2026, 6, 1) + timedelta(seconds=i * 10),
        model="gpt-4.1",
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user or f"Ticket {i}: refund request"}],
        tools=tools or [], response_message={"role": "assistant", "content": "{}"},
        finish_reason="stop",
        usage={"prompt_tokens": 200, "completion_tokens": 5,
               "prompt_tokens_details": {"cached_tokens": 0}})


# -- the region split, which is what the store records ------------------------------------

def test_a_stable_instruction_block_is_all_static():
    calls = [call(i) for i in range(20)]
    static, dynamic, share = region_split(calls, "instructions")
    assert set(static) == set(TEMPLATE)
    assert dynamic == []
    assert share == 1.0


def test_a_per_call_header_is_dynamic_and_the_body_is_not():
    """The cache-breaking shape: a varying line sitting above a stable block."""
    calls = [call(i, extra_system=f"Request {i} at 10:0{i % 10}") for i in range(20)]
    static, dynamic, share = region_split(calls, "instructions")
    assert set(static) == set(TEMPLATE)
    assert len(dynamic) == 20
    assert share > 0.5             # the header is small next to the template


def test_a_payload_user_turn_is_almost_entirely_dynamic():
    """The RAG shape: a couple of instruction lines above a large, different payload."""
    calls = [call(i, user=f"Question: what is case {i}?\nFormat: JSON\n"
                          f"Passage {i}: " + f"{i}word " * 500)
             for i in range(20)]
    static, dynamic, share = region_split(calls, "the user turn")
    assert "Format: JSON" in static
    assert share < 0.1, "a 3k-character payload must not read as template"


def test_a_large_but_identical_payload_is_template_not_payload():
    """Size is not the test - a big block that never changes really is template.

    The counterpart to the case above, and the reason `varies` is judged on content rather than
    on token count: these two differ only in whether the large block repeats.
    """
    fixed = "reference data " * 500
    calls = [call(i, user=f"Question: what is case {i}?\n{fixed}") for i in range(20)]
    _, _, share = region_split(calls, "the user turn")
    assert share > 0.9


def test_static_share_does_not_fall_as_the_export_grows():
    """Summing distinct dynamic lines across the node made this shrink with call count."""
    small = [call(i, user=f"Ticket {i}\nFormat: JSON") for i in range(10)]
    large = [call(i, user=f"Ticket {i}\nFormat: JSON") for i in range(200)]
    _, _, a = region_split(small, "the user turn")
    _, _, b = region_split(large, "the user turn")
    assert abs(a - b) < 0.05


def test_region_text_reads_the_right_messages():
    c = call(0, tools=[{"type": "function", "function": {"name": "lookup"}}])
    assert "You triage support tickets." in region_text(c, "instructions")
    assert "Ticket 0" in region_text(c, "the user turn")
    assert "lookup" in region_text(c, "tool schemas")
    assert region_text(c, "conversation history") == ""


# -- what the store writes ----------------------------------------------------------------

class FakeNodeQuality:
    verdict, margin, reasons = "ok", 0.5, []


class FakeNode:
    def __init__(self, request_ids):
        self.node_id, self.app_id = "app:abc123", "app"
        self.request_ids = request_ids
        self.template = list(TEMPLATE)
        self.distinct_prompts, self.confidence = len(request_ids), "established"
        self.quality = FakeNodeQuality()
        self.signature = {"shape": "SU"}


class FakeNodeResult:
    def __init__(self, calls):
        self.node = FakeNode([c.request_id for c in calls])
        seg = segment([prose_document(c) for c in calls])
        self.metrics = compute(calls, seg, facts=[analyse_call(c) for c in calls])
        self.segmentation = seg
        self.label = {"name": "triage", "purpose": "Triage tickets"}
        self.verdict = None
        self.history = None


class FakeApp:
    def __init__(self, calls):
        self.app_id, self.calls = "app", len(calls)
        self.nodes = [FakeNodeResult(calls)]
        self.residual_calls, self.min_size = 0, 3
        self.chain = None
        self.single_call = {}


class FakeRun:
    def __init__(self, calls):
        self.run_id = "testrun"
        self.apps = [FakeApp(calls)]
        self.gate_passed, self.gate_reason = True, "test"


@pytest.fixture
def store(tmp_path):
    from profiler.store import write_store

    calls = [call(i, extra_system=f"Request {i} at 10:0{i % 10}") for i in range(20)]
    path = str(tmp_path / "calls.duckdb")
    write_store(FakeRun(calls), path, {"app": calls}, "test.csv")
    return duckdb.connect(path, read_only=True)


def test_every_call_is_stored_with_its_callsite(store):
    n, = store.execute("SELECT COUNT(*) FROM calls WHERE node_id = 'app:abc123'").fetchone()
    assert n == 20


def test_the_raw_request_is_recoverable(store):
    payload, = store.execute(
        "SELECT request_json FROM calls WHERE request_id = 'r0'").fetchone()
    request = json.loads(payload)
    assert request["messages"][0]["role"] == "system"
    assert "You triage support tickets." in request["messages"][0]["content"]


def test_the_static_dynamic_verdict_is_recorded_line_by_line(store):
    """The row a reader needs in order to disagree with the profile."""
    static = store.execute(
        "SELECT line, doc_frequency, calls FROM region_lines "
        "WHERE region = 'instructions' AND kind = 'static'").fetchall()
    assert {line for line, _, _ in static} == set(TEMPLATE)
    assert all(df == total for _, df, total in static), \
        "a line called static must appear in every call"

    dynamic = store.execute(
        "SELECT line, doc_frequency FROM region_lines "
        "WHERE region = 'instructions' AND kind = 'dynamic'").fetchall()
    assert len(dynamic) == 20
    assert all(df == 1 for _, df in dynamic)
    assert all(line.startswith("Request ") for line, _ in dynamic)


def test_regions_carry_their_own_stability(store):
    rows = dict(store.execute(
        "SELECT region, varies FROM regions").fetchall())
    assert rows["instructions"] is True          # the per-call header makes it vary
    assert rows["the user turn"] is True


def test_unassigned_calls_are_still_stored(tmp_path):
    """Calls below the size floor have no callsite but must not vanish from the record."""
    from profiler.store import write_store

    calls = [call(i) for i in range(20)]
    run = FakeRun(calls[:15])
    run.apps[0].nodes[0].node.request_ids = [c.request_id for c in calls[:15]]
    path = str(tmp_path / "s.duckdb")
    write_store(run, path, {"app": calls}, "")
    con = duckdb.connect(path, read_only=True)
    assert con.execute("SELECT COUNT(*) FROM calls").fetchone()[0] == 20
    assert con.execute("SELECT COUNT(*) FROM calls WHERE node_id IS NULL").fetchone()[0] == 5
