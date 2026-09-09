"""Matching one request against callsites we already know.

The property under test is not "it finds a match" but "it refuses when it should". The purpose
is to skip work already done, so a false match silently omits a callsite nobody has ever looked
at, while a miss only costs doing the work twice. The tests are weighted accordingly.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest

from profiler.load import Call
from profiler.match import MATCH_CONTAINMENT, Matcher

TRIAGE = ["## Role", "You triage incoming support tickets.", "## Rules",
          "Never invent an order id.", "Reply with JSON only.", "Escalate refunds over $500."]
NOTES = ["## Role", "You draft release notes from merged pull requests.",
         "Group changes by component.", "Use the past tense.", "Link the PR number."]
THIN = ["You are a helpful assistant."]


def remember(path, entries):
    """Write callsites into a registry the way a real run would."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE callsites (
        node_id TEXT PRIMARY KEY, app_id TEXT, name TEXT, purpose TEXT, template TEXT,
        tools TEXT, shape TEXT, first_seen TEXT, last_seen TEXT, runs_seen INTEGER,
        version INTEGER, calls_total INTEGER, last_verdict TEXT)""")
    for node_id, app_id, template, tools in entries:
        con.execute("INSERT INTO callsites VALUES (?,?,?,?,?,?,?,?,?,1,1,10,NULL)",
                    (node_id, app_id, node_id.split(":")[-1], "", json.dumps(template),
                     json.dumps(tools), "SU", "2026-01-01", "2026-01-01"))
    con.commit()
    con.close()
    return path


def call(system, user="Ticket 1: refund request", tools=None, app="app"):
    return Call(request_id="r1", app_id=app, ts=datetime(2026, 6, 1), model="gpt-4.1",
                messages=[{"role": "system", "content": "\n".join(system)},
                          {"role": "user", "content": user}],
                tools=[{"type": "function", "function": {"name": t}} for t in (tools or [])],
                response_message={"role": "assistant", "content": "{}"},
                finish_reason="stop", usage={})


@pytest.fixture
def matcher(tmp_path):
    path = remember(str(tmp_path / "r.sqlite"), [
        ("app:triage", "app", TRIAGE, []),
        ("app:notes", "app", NOTES, []),
    ])
    return Matcher(path)


# -- the case it exists for --------------------------------------------------------------

def test_an_identical_request_is_recognised(matcher):
    m = matcher.match(call(TRIAGE))
    assert m.verdict == "matched" and m.node_id == "app:triage"
    assert m.seen_before


def test_a_request_carrying_a_large_payload_is_still_recognised(matcher):
    """Containment, not Jaccard - the payload must not count against it.

    Under Jaccard a 400-line retrieved document drowns a 6-line template and the callsite is
    reported as new on every call, which is the failure this whole module exists to avoid.
    """
    payload = [f"Retrieved passage {i} about liability and settlement." for i in range(400)]
    m = matcher.match(call(TRIAGE + payload))
    assert m.verdict == "matched" and m.node_id == "app:triage"
    assert m.containment == 1.0


def test_one_edited_line_still_matches(matcher):
    edited = TRIAGE[:-1] + ["Escalate refunds over $1000."]
    m = matcher.match(call(edited))
    assert m.verdict == "matched" and m.node_id == "app:triage"
    assert m.containment < 1.0


# -- the cases where it must refuse -------------------------------------------------------

def test_an_unknown_callsite_is_not_forced_onto_the_nearest_match(matcher):
    m = matcher.match(call(["## Role", "You reconcile custody positions.",
                            "Flag any break above 1,000,000.", "Return a markdown table."]))
    assert m.verdict == "unknown" and m.node_id is None


def test_a_heavily_rewritten_prompt_is_reported_unknown(matcher):
    """Past a point an edit is a new callsite, and guessing would skip work never done."""
    m = matcher.match(call(TRIAGE[:2] + ["Completely different instructions.",
                                         "Answer in French.", "Cite three sources."]))
    assert m.verdict == "unknown"
    assert "below" in m.reason


def test_two_callsites_that_both_fit_are_ambiguous_not_a_coin_toss(tmp_path):
    """A template that is a subset of another matches both; saying which would be a guess."""
    path = remember(str(tmp_path / "r.sqlite"), [
        ("app:base", "app", TRIAGE, []),
        ("app:extended", "app", TRIAGE + ["Also check the fraud list."], []),
    ])
    m = Matcher(path).match(call(TRIAGE + ["Also check the fraud list."]))
    assert m.verdict == "ambiguous"
    assert m.node_id is not None and m.runners_up


def test_a_boilerplate_template_cannot_carry_a_match_alone(tmp_path):
    """A generic line is unidentifiable when the ESTATE shares it, not when it is short.

    Thirty callsites all fronted by "You are a helpful assistant." and nothing else: the line
    identifies none of them, and answering with any one would be a guess. In an estate of one
    the same line is perfectly distinctive, which is why the rule counts how many callsites
    share a line rather than how many lines a callsite has.
    """
    path = remember(str(tmp_path / "r.sqlite"),
                    [(f"app:generic{i}", "app", THIN, []) for i in range(30)])
    m = Matcher(path).match(call(THIN + ["Summarise the document below.", "Be terse."]))
    assert m.verdict == "unknown"


def test_a_short_but_distinctive_template_does_match(tmp_path):
    """The case the old line-count rule silently discarded.

    `app_platform`'s amount extractor is two lines and unmistakable. Requiring three threw it
    away without even scoring it, so a callsite that had been profiled came back as new.
    """
    short = ["Extract the total amount. Return JSON.", "Invoice text:"]
    path = remember(str(tmp_path / "r.sqlite"), [
        ("app:amount", "app", short, []),
        ("app:triage", "app", TRIAGE, []),
    ])
    m = Matcher(path).match(call(short + ["Invoice 4471 dated 2026-03-02, total EUR 812.40."]))
    assert m.verdict == "matched" and m.node_id == "app:amount"


def test_a_thin_template_with_tools_can_match(tmp_path):
    """Tools carry the identity the prompt cannot."""
    path = remember(str(tmp_path / "r.sqlite"),
                    [("app:thin", "app", THIN, ["run_sql", "list_tables"])])
    m = Matcher(path).match(call(THIN, tools=["run_sql", "list_tables"]))
    assert m.verdict == "matched"


def test_the_same_prompt_with_disjoint_tools_does_not_match(tmp_path):
    """A router and an executor behind one base prompt - the grouper's veto applies here too."""
    path = remember(str(tmp_path / "r.sqlite"),
                    [("app:executor", "app", TRIAGE, ["amend_trade", "post_journal"])])
    m = Matcher(path).match(call(TRIAGE, tools=["classify_intent", "route_to_queue"]))
    assert m.verdict == "unknown"


def test_an_empty_registry_answers_unknown_rather_than_raising(tmp_path):
    path = remember(str(tmp_path / "r.sqlite"), [])
    m = Matcher(path).match(call(TRIAGE))
    assert m.verdict == "unknown" and m.node_id is None


# -- the agent id is not trusted as a partition -------------------------------------------

def test_a_callsite_is_found_even_under_a_different_agent_id(matcher):
    """One agent name routinely fronts several agents, and one agent appears under several
    names. Scoping the lookup to app_id would miss exactly the callsite it should find."""
    m = matcher.match(call(TRIAGE, app="a-completely-different-name"))
    assert m.verdict == "matched" and m.node_id == "app:triage"


def test_scoping_to_the_app_is_available_when_the_id_is_trusted(tmp_path):
    path = remember(str(tmp_path / "r.sqlite"), [("app:triage", "app", TRIAGE, [])])
    scoped = Matcher(path, scope_to_app=True)
    assert scoped.match(call(TRIAGE, app="app")).verdict == "matched"
    assert scoped.match(call(TRIAGE, app="other")).verdict == "unknown"


# -- the index must not change the answer -------------------------------------------------

def test_the_index_returns_what_a_full_scan_would(tmp_path):
    """Candidate generation is an optimisation; it must not decide anything."""
    entries = [(f"app:site{i}", "app",
                ["## Role", f"You handle workflow {i}.", f"Reference table {i}.",
                 "Escalate anything unresolved."], []) for i in range(200)]
    path = remember(str(tmp_path / "r.sqlite"), entries)
    matcher = Matcher(path)
    assert len(matcher) == 200

    target = entries[137]
    m = matcher.match(call(target[2]))
    assert m.verdict == "matched" and m.node_id == "app:site137"
