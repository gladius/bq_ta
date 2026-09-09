"""The architectural invariant: LLM output never influences a deterministic measurement.

If a judgment could silently feed back into clustering or the metrics, we would lose the
ability to tell whether the measured method works - which is the property that made every
finding in the spike checkable. So it is enforced structurally, not by convention.
"""

from __future__ import annotations

import ast
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# These modules produce the measured numbers. None of them may reach for the LLM layer.
DETERMINISTIC = ["load.py", "single.py", "identity.py", "segment.py", "metrics.py",
                 "config.py", "audit.py", "fingerprint.py", "chain.py", "registry.py"]


def _imports(path):
    with open(path, "r", encoding="utf-8") as handle:
        tree = ast.parse(handle.read())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


@pytest.mark.parametrize("filename", DETERMINISTIC)
def test_deterministic_modules_do_not_import_the_llm_layer(filename):
    path = os.path.join(ROOT, "profiler", filename)
    offenders = {n for n in _imports(path) if "llm" in n or n == "openai"}
    assert not offenders, f"{filename} reaches into the LLM layer: {offenders}"


@pytest.mark.parametrize("filename", DETERMINISTIC)
def test_deterministic_modules_are_import_clean_without_a_key(filename, monkeypatch):
    """They must work with no API key present at all."""
    monkeypatch.delenv("LLM_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    module = __import__(f"profiler.{filename[:-3]}", fromlist=["*"])
    assert module is not None


def test_llm_judgments_are_applied_only_through_explicit_calls():
    """Merges and rescues must go through named functions, so they land in the audit trail."""
    with open(os.path.join(ROOT, "profiler", "pipeline.py"), "r", encoding="utf-8") as handle:
        source = handle.read()
    assert "merge_nodes(" in source
    assert "promote_residual(" in source
    # and both must be guarded by an explicit enablement check
    assert source.count("client.enabled") >= 3


def test_the_optimisation_layer_stays_removed():
    """Profiling first.

    The auditors and optimisers were deleted because findings attached to a grouping we could
    not yet trust are worse than none - they are confidently wrong. If they come back, they
    come back deliberately, with this test updated, not by a quiet re-import.
    """
    present = set(os.listdir(os.path.join(ROOT, "profiler", "llm")))
    for gone in ("optimize.py", "auditors.py", "validate.py"):
        assert gone not in present, f"{gone} is back; the profile half must be solid first"


def test_the_judge_cannot_rewrite_what_it_judges():
    """The verifier reports; it must not mutate the profile it is checking."""
    with open(os.path.join(ROOT, "profiler", "llm", "judge.py"), "r",
              encoding="utf-8") as handle:
        source = handle.read()
    for forbidden in (".template =", ".static_lines =", ".metrics =", ".request_ids ="):
        assert forbidden not in source, f"judge.py mutates the profile via `{forbidden}`"
