"""Memory across runs.

The property that matters: a callsite whose prompt was edited must be recognised as the SAME
callsite carrying a new version, not as a new one. Without that, a live estate where prompts
change weekly re-analyses everything forever and accumulates nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import pytest


from profiler.identity import stable_node_id
from profiler.registry import Registry, line_delta, template_similarity

BASE = ["## Role", "You triage incoming support tickets.", "## Rules",
        "Never invent an order id.", "Reply with JSON only.", "Escalate refunds over $500."]


# -- the minimum surface Registry.reconcile touches, so the test needs no full pipeline ------

@dataclass
class FakeNode:
    node_id: str
    app_id: str = "app"
    template: List[str] = field(default_factory=list)
    signature: Dict[str, Any] = field(default_factory=lambda: {"shape": "SU"})


@dataclass
class FakeMetrics:
    calls: int = 40
    avg_request_tokens: int = 900
    cacheable_now: int = 700
    recoverable: int = 0
    reported_cached_mean: Optional[float] = None
    tools_called: List[str] = field(default_factory=list)


@dataclass
class FakeResult:
    node: FakeNode
    metrics: FakeMetrics = field(default_factory=FakeMetrics)
    label: Optional[Dict[str, Any]] = None
    verdict: Any = None
    history: Optional[Dict[str, Any]] = None


@dataclass
class FakeApp:
    app_id: str
    nodes: List[FakeResult]


class FakeAudit:
    def __init__(self, run_id="run1"):
        self.run_id = run_id
        self.decisions = []

    def decision(self, kind, **kw):
        self.decisions.append((kind, kw))


def node_for(template, tools=()):
    result = FakeResult(node=FakeNode(node_id=stable_node_id("app", template, tools, "SU"),
                                      template=list(template)))
    result.metrics.tools_called = list(tools)
    result.label = {"name": "triage", "purpose": "Triage support tickets"}
    return result


@pytest.fixture
def registry(tmp_path):
    reg = Registry(str(tmp_path / "r.sqlite"))
    yield reg
    reg.close()


def test_a_callsite_seen_for_the_first_time_is_new(registry):
    node = node_for(BASE)
    counts = registry.reconcile([FakeApp("app", [node])], FakeAudit())
    assert counts["new"] == 1
    assert node.history["status"] == "new"
    assert node.history["version"] == 1


def test_the_same_prompt_next_run_is_recognised(registry):
    registry.reconcile([FakeApp("app", [node_for(BASE)])], FakeAudit())
    again = node_for(BASE)
    counts = registry.reconcile([FakeApp("app", [again])], FakeAudit())
    assert counts["seen"] == 1
    assert again.history["status"] == "seen"
    assert again.history["runs_seen"] == 2


def test_an_edited_prompt_carries_its_history_forward(registry):
    """The case the content-derived id alone cannot handle."""
    registry.reconcile([FakeApp("app", [node_for(BASE)])], FakeAudit())

    edited = BASE + ["Always cite the ticket number."]
    node = node_for(edited)
    assert node.node.node_id != stable_node_id("app", BASE, (), "SU")   # id genuinely changed

    audit = FakeAudit()
    counts = registry.reconcile([FakeApp("app", [node])], audit)
    assert counts["changed"] == 1
    assert node.history["status"] == "changed"
    assert node.history["version"] == 2
    assert node.history["lines_added"] == 1
    assert node.history["lines_removed"] == 0
    assert node.history["runs_seen"] == 2
    assert audit.decisions and audit.decisions[0][0] == "callsite_changed"


def test_a_genuinely_different_callsite_is_not_absorbed(registry):
    registry.reconcile([FakeApp("app", [node_for(BASE)])], FakeAudit())
    other = node_for(["## Role", "You write release notes from merged pull requests.",
                      "Group changes by component.", "Use the past tense."])
    counts = registry.reconcile([FakeApp("app", [other])], FakeAudit())
    assert counts["new"] == 1
    assert other.history["status"] == "new"


def test_history_survives_the_rename_so_the_series_is_unbroken(registry):
    registry.reconcile([FakeApp("app", [node_for(BASE)])], FakeAudit("run1"))
    node = node_for(BASE + ["Always cite the ticket number."])
    registry.reconcile([FakeApp("app", [node])], FakeAudit("run2"))
    # both observations now hang off the new id, not split across two
    assert len(registry.history_of(node.node.node_id)) == 2


def test_two_survivors_cannot_both_claim_the_same_ancestor(registry):
    """A split callsite must not have its history duplicated into both halves."""
    registry.reconcile([FakeApp("app", [node_for(BASE)])], FakeAudit())
    left = node_for(BASE + ["Escalate anything mentioning legal."])
    right = node_for(BASE + ["Escalate anything mentioning press."])
    counts = registry.reconcile([FakeApp("app", [left, right])], FakeAudit())
    assert counts["changed"] == 1 and counts["new"] == 1


def test_similarity_separates_an_edit_from_a_different_callsite():
    edit = template_similarity(BASE + ["One more rule."], [], "SU", BASE, [], "SU")
    unrelated = template_similarity(["## Role", "You write release notes.", "Be terse."],
                                    [], "SU", BASE, [], "SU")
    assert edit > 0.70 > unrelated


def test_same_template_different_tools_is_not_the_same_callsite():
    score = template_similarity(BASE, ["search", "fetch"], "SU",
                                BASE, ["deploy", "rollback"], "SU")
    assert score < 0.70


def test_line_delta_counts_both_directions():
    assert line_delta(["a", "b"], ["b", "c"]) == (1, 1)
