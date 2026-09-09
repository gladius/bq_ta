"""Given ONE request, which known callsite is it - if any?

This is the opposite direction to `identity.py`, and it needs a different measure.

Grouping is a batch operation: it takes many requests, uses document frequency to work out which
lines are template and which are payload, and clusters what is left. None of that survives a
sample size of one. `_template` thresholds at 0.9 x len(members), so with a single request every
line counts as static, the template becomes the whole request including its payload, and the id
is different every time. **A node id cannot be derived from one request.**

So the question "have we already profiled this callsite" cannot be answered by recomputing the
id. It has to be answered by *lookup*: compare the incoming request against the templates we
already know, and decide whether it is one of them.

The right measure here is **containment**, not Jaccard:

    containment = |template lines that appear in this request| / |template lines|

Jaccard punishes the request for carrying payload the template does not have - which is the
normal case, and the larger the payload the worse it scores. A request belongs to a callsite
when it *contains* that callsite's template, whatever else it also carries.

At a few thousand callsites a linear scan per request is wasteful, so candidates come from an
inverted index over template lines: only callsites sharing at least one line are scored.

Deterministic. No LLM, no network. Reads the registry; never writes to it.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Set, Tuple

from profiler.fingerprint import (DISJOINT_VIEW_CEILING, h, lines_of, prompt_text, role_shape,
                                  tool_names, user_template_text)
from profiler.load import Call

# How much of a remembered template must appear in the request before it is that callsite.
# Deliberately high: the cost of a false match is skipping work on a callsite we have never
# actually seen, which is silent. The cost of a miss is doing the work twice, which is not.
MATCH_CONTAINMENT = 0.80

# A match must beat the next best by this much, or the answer is "ambiguous" rather than a
# coin toss between two callsites that both fit.
MIN_MARGIN = 0.10

# What makes a template unable to identify anything is that its lines are COMMON, not that
# there are few of them. Counting lines confuses the two and throws away real callsites:
# "Extract the total amount. Return JSON." / "Invoice text:" is two lines and perfectly
# distinctive, and a line-count rule discarded it without scoring it, while
# "You are a helpful assistant." is one line that half an estate contains.
#
# So identifiability is decided by whether any of a template's lines is rare in this estate -
# which is exactly the anchor test - and a callsite with no rare line can still be identified by
# its tools.

# Retrieval indexes only a callsite's rarest template lines. A line shared by more than this
# fraction of the estate is boilerplate and retrieves everything, which is no retrieval at all.
COMMON_LINE_FRACTION = 0.02
ANCHORS_PER_NODE = 6

# Stop unioning posting lists once this many candidates are in hand. Scoring is cheap; dragging
# in every callsite that shares a moderately common line is not.
ENOUGH_CANDIDATES = 24

EMPTY: FrozenSet[str] = frozenset()


@dataclass
class Known:
    """One remembered callsite, reduced to what matching needs."""

    node_id: str
    app_id: str
    name: Optional[str]
    lines: FrozenSet[int]
    tools: FrozenSet[str]
    shape: str
    version: int
    runs_seen: int
    first_seen: str

    # set at load: this template has no line rare enough to retrieve on
    no_anchors: bool = False

    @property
    def identifiable(self) -> bool:
        """Can this callsite be recognised at all? Rare prompt lines, or tools, or neither."""
        return (not self.no_anchors) or bool(self.tools)


@dataclass
class Match:
    """The verdict for one request, with the evidence behind it."""

    node_id: Optional[str] = None
    name: Optional[str] = None
    score: float = 0.0
    containment: float = 0.0
    verdict: str = "unknown"          # matched | ambiguous | unknown
    reason: str = ""
    runners_up: List[Tuple[str, float]] = field(default_factory=list)
    version: int = 0
    first_seen: str = ""

    @property
    def seen_before(self) -> bool:
        return self.verdict == "matched"

    def as_dict(self) -> Dict[str, Any]:
        return dict(self.__dict__)


def _containment(template: FrozenSet[int], request: FrozenSet[int]) -> float:
    """How much of the template the request carries. 1.0 means all of it."""
    if not template:
        return 0.0
    return len(template & request) / len(template)


def _jaccard(a: FrozenSet, b: FrozenSet) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


def request_lines(call: Call) -> FrozenSet[int]:
    """Every instruction line in this request - system and the opening user turn.

    Not noise-filtered, because a single request has no document frequency to filter by. That
    is fine in this direction: extra lines cost nothing under containment.
    """
    text = prompt_text(call) + "\n" + user_template_text(call)
    return frozenset(h(line) for line in lines_of(text))


class Matcher:
    """Answers 'have we profiled this callsite before' for one request at a time."""

    def __init__(self, registry_path: str, scope_to_app: bool = False) -> None:
        """`scope_to_app` restricts candidates to the request's own app_id.

        Off by default. An agent id from a gateway log is not a reliable partition - one name
        routinely fronts several agents, and the same agent appears under several names - so
        keying the lookup on it would miss the callsite it is supposed to find. Searching every
        remembered callsite costs little once the index is doing candidate generation.
        """
        self.scope_to_app = scope_to_app
        self.known: Dict[str, Known] = {}
        self.by_line: Dict[int, Set[str]] = defaultdict(set)
        self.by_anchor: Dict[int, Set[str]] = defaultdict(set)
        self.always_scan: Set[str] = set()
        self._load(registry_path)
        self._build_anchors()

    def _load(self, path: str) -> None:
        con = sqlite3.connect(path)
        try:
            rows = con.execute(
                "SELECT node_id, app_id, name, template, tools, shape, version, runs_seen, "
                "first_seen FROM callsites").fetchall()
        finally:
            con.close()

        for node_id, app_id, name, template, tools, shape, version, runs, first in rows:
            lines = frozenset(h(ln) for ln in json.loads(template or "[]")
                              if ln and ln != "[user turn]")
            entry = Known(node_id=node_id, app_id=app_id, name=name, lines=lines,
                          tools=frozenset(json.loads(tools or "[]")), shape=shape or "",
                          version=version, runs_seen=runs, first_seen=first)
            self.known[node_id] = entry
            for line in lines:
                self.by_line[line].add(node_id)

    def _build_anchors(self) -> None:
        """Index each callsite on its RAREST template lines, not on all of them.

        Indexing every line does no work at all. Boilerplate is boilerplate precisely because
        every callsite has it: measured over 5,000 callsites, `## Role`, `Return JSON only.` and
        `Escalate anything you cannot resolve.` each appeared in all 5,000, so every request
        retrieved every callsite and the "index" was a linear scan with extra steps.

        The distinguishing power sits in the opposite tail - the median template line belongs to
        exactly one callsite. Retrieving on the rare lines is what a search engine does with rare
        terms, and it turns candidate generation from O(estate) into O(a handful).

        A callsite built entirely from boilerplate has no rare line to be found by. Those go in
        `always_scan`, which is small by construction: if it were large, the estate would have no
        distinguishable callsites and nothing here would work anyway.
        """
        estate = max(len(self.known), 1)
        # a line in more than this share of callsites carries no retrieval value
        common = max(2, int(estate * COMMON_LINE_FRACTION))

        for known in self.known.values():
            rare = sorted((ln for ln in known.lines if len(self.by_line[ln]) <= common),
                          key=lambda ln: len(self.by_line[ln]))
            anchors = rare[:ANCHORS_PER_NODE]
            if not anchors:
                known.no_anchors = True
                # only worth scoring at all if something else can identify it
                if known.tools:
                    self.always_scan.add(known.node_id)
                continue
            for line in anchors:
                self.by_anchor[line].add(known.node_id)

    def __len__(self) -> int:
        return len(self.known)

    def candidates(self, lines: FrozenSet[int], app_id: str) -> List[Known]:
        """Callsites worth scoring for this request.

        Retrieval is on anchors - the rare lines - plus the small set of callsites that have no
        rare line at all. This narrows what is scored; it must never change what is chosen, so
        `test_the_anchor_index_returns_what_a_full_scan_would` compares it against scoring the
        whole estate.
        """
        # Union the request's anchor lines RAREST FIRST and stop once there is enough to score.
        #
        # Taking every anchor makes candidates grow with the estate: a line like
        # "Apply desk rule 137" is shared by 125 callsites at 50,000, still "rare" by any
        # relative threshold, and unioning it drags in all 125. Measured that way candidates went
        # 3 -> 18 -> 71 -> 179 as the estate grew, which is linear wearing an index's clothes.
        #
        # If a callsite's template is contained in this request then every one of its anchors is
        # in this request too - including its rarest. So the rarest lines find it, and the common
        # ones only add candidates that a rarer line would have found anyway.
        found: Set[str] = set(self.always_scan)
        present = sorted((ln for ln in lines if ln in self.by_anchor),
                         key=lambda ln: len(self.by_anchor[ln]))
        for line in present:
            if len(found) >= ENOUGH_CANDIDATES:
                break
            posting = self.by_anchor[line]
            # Once a rare line has produced candidates, refuse a big posting list rather than
            # unioning it to reach the target. Breaking only AFTER the union let one 500-entry
            # list in to get from 1 candidate to 24, and candidates grew with the estate again:
            # 3 -> 18 -> 29 -> 71 -> 286. The true match is found by its own rarest anchor,
            # which is in this request and is processed first, so a longer list adds only
            # callsites a rarer line would have surfaced anyway.
            if found and len(posting) > ENOUGH_CANDIDATES:
                continue
            found |= posting

        out = [self.known[n] for n in found]
        if self.scope_to_app:
            out = [k for k in out if k.app_id == app_id]
        return out

    def score(self, known: Known, lines: FrozenSet[int], tools: FrozenSet[str],
              shape: str) -> Tuple[float, float]:
        """(score, containment) for one candidate."""
        contained = _containment(known.lines, lines)

        # tools and shape corroborate; they cannot carry a match on their own
        agreement = [contained] * 3
        if known.tools or tools:
            tool_score = _jaccard(known.tools, tools)
            agreement.append(tool_score)
            # The same veto the grouper uses: both sides declare tools and share none, which is
            # affirmative evidence of a different job rather than a weak signal.
            #
            # It has to reach the VERDICT, not just the score. Capping the score alone left
            # containment at 1.0, so a router carrying an executor's prompt sailed through the
            # containment gate and was reported as the executor - the exact collision this veto
            # exists to stop, arriving through the one number the gate actually reads.
            if known.tools and tools and tool_score == 0.0:
                capped = min(contained, DISJOINT_VIEW_CEILING)
                return capped, capped
        if known.shape and shape:
            agreement.append(1.0 if known.shape == shape else 0.5)

        return sum(agreement) / len(agreement), contained

    def match(self, call: Call) -> Match:
        """Which remembered callsite produced this request, if any."""
        lines = request_lines(call)
        tools = tool_names(call)
        shape = role_shape(call)

        scored: List[Tuple[float, float, Known]] = []
        for known in self.candidates(lines, call.app_id):
            if not known.identifiable:
                # no rare prompt line and no tools - nothing here can identify anything, and
                # refusing is more useful than a match on boilerplate
                continue
            score, contained = self.score(known, lines, tools, shape)
            if contained > 0:
                scored.append((score, contained, known))

        if not scored:
            return Match(verdict="unknown",
                         reason="no remembered callsite shares a template line with this "
                                "request")

        scored.sort(key=lambda t: -t[0])
        score, contained, best = scored[0]
        runners = [(k.node_id, round(s, 3)) for s, _, k in scored[1:4]]

        result = Match(node_id=best.node_id, name=best.name, score=round(score, 3),
                       containment=round(contained, 3), runners_up=runners,
                       version=best.version, first_seen=best.first_seen)

        if contained < MATCH_CONTAINMENT:
            result.verdict = "unknown"
            result.node_id = None
            result.reason = (f"closest is {best.node_id} but the request carries only "
                             f"{contained:.0%} of its template - below {MATCH_CONTAINMENT:.0%}, "
                             f"so this is a callsite we have not profiled")
            return result

        margin = score - (scored[1][0] if len(scored) > 1 else 0.0)
        if len(scored) > 1 and margin < MIN_MARGIN:
            result.verdict = "ambiguous"
            result.reason = (f"{best.node_id} and {scored[1][2].node_id} both fit, separated by "
                             f"{margin:.3f} - too close to call, so treat this callsite as "
                             f"unprofiled rather than guess")
            return result

        result.verdict = "matched"
        result.reason = (f"carries {contained:.0%} of this callsite's template; next best is "
                         f"{margin:.2f} behind")
        return result
