"""Callsite discovery: noise filter -> weighted similarity -> clustering -> stable ids.

Two things changed after the first production review, both because the earlier version could
not be trusted in a dynamic estate:

1. **Identity is no longer the system prompt alone.** It is a weighted blend of prompt lines,
   tool set, message shape and response format (see `fingerprint.py`). The old version merged
   any two callsites that shared a base template and fragmented any agent that had no system
   prompt.

2. **Every node now carries quality signals computed without ground truth** - intra-node
   similarity, nearest-neighbour distance, distinct-prompt ratio, template size. In production
   there is no answer key, so the grouping has to be able to report its own reliability.

The noise filter, the linkage and the threshold are unchanged and were measured in the spike
(../spike/DECISIONS.md D26-D32).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import xxhash

from profiler import fingerprint as fp
from profiler.fingerprint import Fingerprint, Similarity, h, lines_of, prompt_text
from profiler.load import Call

STABLE_FRACTION = 0.9
MAX_LINKAGE_POINTS = 4000     # scale guard on the O(n^2) step
USER_DF_FRACTION = 0.05       # a user line must recur across 5% of an app to be template


def identity_text(call: Call) -> str:
    """Kept for callers that want the raw instruction surface of one call."""
    return prompt_text(call)


def jaccard(a: FrozenSet[int], b: FrozenSet[int]) -> float:
    if not a and not b:
        return 1.0
    union = len(a | b)
    return len(a & b) / union if union else 1.0


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


@dataclass
class NodeQuality:
    """How much to trust one node, computed with no answer key.

    In production nothing can be checked against truth, so these stand in for it. Each is a
    number a reader can argue with.
    """

    min_internal: float = 1.0        # weakest link holding the node together
    mean_internal: float = 1.0
    nearest_other: float = 0.0       # closest competing node - how nearly it merged
    margin: float = 1.0              # min_internal - nearest_other; small means arbitrary
    distinct_ratio: float = 0.0      # distinct prompts / calls; ~1.0 means nothing is shared
    template_lines: int = 0
    verdict: str = "ok"
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in self.__dict__.items()}


def assess(quality: NodeQuality, size: int) -> NodeQuality:
    """Turn the raw signals into a verdict a human can act on."""
    reasons: List[str] = []
    if quality.template_lines <= 2:
        reasons.append("template is 2 lines or fewer - almost no shared identity")
    if quality.distinct_ratio > 0.95 and size > 5:
        reasons.append("nearly every call has a different prompt - the group may be arbitrary")
    if quality.margin < 0.05:
        reasons.append(f"only {quality.margin:.3f} separates this node from the next closest - "
                       f"the split is near-arbitrary")
    if quality.min_internal < 0.35:
        reasons.append(f"weakest internal link is {quality.min_internal:.2f} - the node is held "
                       f"together by a chain, not by mutual similarity")
    if size < 5:
        reasons.append(f"only {size} calls - too few to be confident")

    quality.reasons = reasons
    if any("almost no shared identity" in r or "may be arbitrary" in r for r in reasons):
        quality.verdict = "suspect"
    elif reasons:
        quality.verdict = "weak"
    else:
        quality.verdict = "ok"
    return quality


@dataclass
class Node:
    node_id: str
    app_id: str
    request_ids: List[str]
    template: List[str] = field(default_factory=list)
    variable_lines: int = 0
    distinct_prompts: int = 0
    confidence: str = "established"
    medoid_request_id: Optional[str] = None
    quality: NodeQuality = field(default_factory=NodeQuality)
    signature: Dict[str, Any] = field(default_factory=dict)
    llm_verdict: Optional[Dict[str, Any]] = None

    @property
    def size(self) -> int:
        return len(self.request_ids)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "node_id": self.node_id, "app_id": self.app_id, "size": self.size,
            "distinct_prompts": self.distinct_prompts, "confidence": self.confidence,
            "template_lines": len(self.template), "variable_lines": self.variable_lines,
            "medoid_request_id": self.medoid_request_id,
            "quality": self.quality.as_dict(), "identity_signature": self.signature,
            "llm_verdict": self.llm_verdict,
        }


@dataclass
class AmbiguousPair:
    app_id: str
    score: float
    left_request_id: str
    right_request_id: str
    explain: str = ""


@dataclass
class GroupingResult:
    nodes: List[Node]
    residual: List[Node]
    ambiguous: List[AmbiguousPair]
    doc_frequency: Dict[int, int]
    line_text: Dict[int, str]
    min_size: int
    df_histogram: Dict[str, int]
    # calls the O(n^2) scale guard could not compare. Never silently dropped: they land in
    # `residual` and are counted here so the report can say so.
    unlinked_calls: int = 0
    signature: Dict[str, Any] = field(default_factory=dict)
    # The noise-filtered comparison keys the clustering actually used. Exposed so a diagnostic
    # scores the same thing the grouper scored: `diagnose.py` recomputed similarity from the RAW
    # sets and reported OVERLAP for an agent that had grouped at purity 1.000 - a histogram that
    # contradicts the decision it is meant to explain is worse than none.
    filtered: Dict[str, FrozenSet[int]] = field(default_factory=dict)
    filtered_user: Dict[str, FrozenSet[int]] = field(default_factory=dict)


def document_frequency(line_sets: Sequence[FrozenSet[int]]) -> Dict[int, int]:
    counts: Counter = Counter()
    for lines in line_sets:
        counts.update(lines)
    return dict(counts)


def df_histogram(df: Dict[int, int], n: int) -> Dict[str, int]:
    bands = {"singleton": 0, "<10%": 0, "10-40%": 0, "40-90%": 0, ">=90%": 0}
    for count in df.values():
        ratio = count / max(n, 1)
        if count == 1:
            bands["singleton"] += 1
        elif ratio < 0.10:
            bands["<10%"] += 1
        elif ratio < 0.40:
            bands["10-40%"] += 1
        elif ratio < 0.90:
            bands["40-90%"] += 1
        else:
            bands[">=90%"] += 1
    return bands


MAX_ADAPTIVE_FLOOR = 25       # a busy agent must not raise the bar out of the tail's reach


def adaptive_min_size(n_calls: int, fraction: float, floor: int) -> int:
    """How many calls a cluster needs before it is reported as a callsite.

    The fraction is capped. A proportional floor assumes callsites are of comparable size, and
    a real estate is skewed: `app_swarm` has one hot path of 2,400 calls beside thirty genuine
    callsites of 4-25. Two percent of that agent is 56, so **all thirty disappeared** - the same
    failure a fixed floor of 20 caused in the spike, arriving from the other direction. A busy
    hot path is not evidence that a small callsite is noise.
    """
    return max(floor, min(math.ceil(fraction * n_calls), MAX_ADAPTIVE_FLOOR))


def is_coherent(node: "Node") -> bool:
    """Is a below-floor cluster a real callsite, or noise?

    The floor exists to reject noise, not to reject small callsites, and the two are
    distinguishable without counting calls: noise has no shared template and a different prompt
    every time. A cluster of eight calls sharing a five-line template is a callsite that happens
    to be quiet.
    """
    return (len(node.template) >= 3
            and node.quality.distinct_ratio <= 0.95
            and node.size >= 3)


def _template(members: Sequence[FrozenSet[int]], line_text: Dict[int, str],
              order_hint: Sequence[int]) -> Tuple[List[str], int]:
    counts: Counter = Counter()
    for lines in members:
        counts.update(lines)
    threshold = STABLE_FRACTION * len(members)
    stable = {line for line, count in counts.items() if count >= threshold}
    ordered = [line_text.get(line, "") for line in order_hint if line in stable]
    seen: Set[str] = set()
    template = [ln for ln in ordered if ln and not (ln in seen or seen.add(ln))]
    return template, len(counts) - len(stable)


def stable_node_id(app_id: str, template: Sequence[str], tools: Sequence[str] = (),
                   shape: str = "") -> str:
    """Content-derived id, now including the tool set and shape.

    Two callsites that share a template but differ in tools are different callsites, so they
    must not collide - which the template-only version did.
    """
    material = "\n".join(template) + " " + ",".join(sorted(tools)) + " " + shape
    return f"{app_id}:{xxhash.xxh64(material.encode()).hexdigest()[:12]}"


def group_app(
    app_id: str, calls: Sequence[Call], tau: float, min_doc_freq: int,
    min_node_fraction: float, min_node_floor: int,
    ambiguous_low: float, ambiguous_high: float,
) -> GroupingResult:
    """Discover the callsites of one app, and report how much to trust each."""
    prints = [fp.build(c) for c in calls]
    by_id = {p.request_id: p for p in prints}

    line_text: Dict[int, str] = {}
    for call in calls:
        for line in lines_of(prompt_text(call)):
            line_text.setdefault(h(line), line)
        for line in lines_of(fp.user_template_text(call)):
            line_text.setdefault(h(line), line)

    df = document_frequency([p.prompt_lines for p in prints])
    # The filter changes only the COMPARISON KEY. Nothing is removed from stored data.
    kept = {line for line, count in df.items() if count >= min_doc_freq}
    filtered = {p.request_id: frozenset(p.prompt_lines & kept) for p in prints}

    # The user turn is a second template surface and needs a MUCH higher noise floor than the
    # system prompt, because a replayed history makes payload look like template.
    #
    # Measured: a tool loop resends its opening user turn on every step, so "Investigate break
    # 7 on the EUR book" appears 3-9 times - past a df floor of 2 - and is kept as though it
    # were template. Every run then had a template line no other run shared, the disjoint-view
    # veto fired between runs, and one callsite of 144 calls shattered into 24 clusters, 60 of
    # them below the size floor and dropped entirely.
    #
    # A genuine user template recurs across a large share of the app's calls, not merely within
    # one run, so the floor is proportional. Where a small callsite falls under it the user view
    # simply drops out and identity falls back on prompt and tools - degradation, not breakage.
    user_min_df = max(min_doc_freq, math.ceil(USER_DF_FRACTION * len(calls)))
    df_user = document_frequency([p.user_lines for p in prints])
    kept_user = {line for line, count in df_user.items() if count >= user_min_df}
    filtered_user = {p.request_id: frozenset(p.user_lines & kept_user) for p in prints}

    # Deduplicate on the full identity, not the prompt alone: two calls agree only if their
    # user template, tools, shape and format agree too.
    def key_of(p: Fingerprint) -> Tuple:
        return (filtered[p.request_id], filtered_user[p.request_id], p.tools, p.shape,
                p.response_format)

    distinct: Dict[Tuple, List[int]] = {}
    for index, p in enumerate(prints):
        distinct.setdefault(key_of(p), []).append(index)
    # The O(n^2) linkage is bounded, but the bound must not eat calls. The previous guard sliced
    # `points` and left the calls behind the dropped keys in no cluster at all - not a node, not
    # residual, simply absent from the profile with nothing said about it. An agent with more
    # than 4,000 distinct prompts (routine for high-variance RAG traffic) silently lost the tail.
    #
    # Now the keys are ordered by how many calls sit behind them, so the guard keeps the most
    # significant, and everything it cannot compare is carried out explicitly as `unlinked`.
    points = sorted(distinct, key=lambda k: -len(distinct[k]))
    unlinked_keys: List[Tuple] = []
    if len(points) > MAX_LINKAGE_POINTS:
        points, unlinked_keys = points[:MAX_LINKAGE_POINTS], points[MAX_LINKAGE_POINTS:]
    reps = [prints[distinct[k][0]] for k in points]

    scores: Dict[Tuple[int, int], float] = {}
    uf = _UnionFind(len(points))
    ambiguous: List[AmbiguousPair] = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            sim = fp.similarity(reps[i], reps[j], filtered, filtered_user)
            scores[(i, j)] = sim.score
            if sim.score >= tau:
                uf.union(i, j)
            elif ambiguous_low <= sim.score < min(tau, ambiguous_high):
                ambiguous.append(AmbiguousPair(
                    app_id=app_id, score=round(sim.score, 4),
                    left_request_id=reps[i].request_id,
                    right_request_id=reps[j].request_id, explain=sim.explain()))

    clusters: Dict[int, List[int]] = {}
    point_cluster: Dict[int, int] = {}
    for i, key in enumerate(points):
        root = uf.find(i)
        clusters.setdefault(root, []).extend(distinct[key])
        point_cluster[i] = root

    # whatever the guard could not compare becomes its own residual cluster, so the calls stay
    # in the accounting even though nothing was decided about them
    unlinked_calls = 0
    for key in unlinked_keys:
        members = distinct[key]
        unlinked_calls += len(members)
        root = -(len(clusters) + 1)
        clusters[root] = list(members)

    min_size = adaptive_min_size(len(calls), min_node_fraction, min_node_floor)
    nodes: List[Node] = []
    residual: List[Node] = []
    # Two clusters can derive the same identity material and so the same id. Downstream that is
    # corrupting rather than merely untidy - the registry keys history on the id, and a
    # request_id -> node_id map silently loses one of them. Ids are made unique here, and the
    # collision is surfaced on the node rather than hidden.
    minted: Counter = Counter()

    for root, members in sorted(clusters.items(), key=lambda kv: -len(kv[1])):
        member_prints = [prints[i] for i in members]
        member_keys = [filtered[p.request_id] for p in member_prints]
        longest = max(members, key=lambda i: len(prints[i].ordered_lines))
        template, variable = _template(member_keys, line_text, prints[longest].ordered_lines)

        # the stable part of the user turn belongs in the template too - it is instruction
        # text that happens to be sent under a different role
        #
        # The user section joins the PROFILE but not the node id. Whether a user turn happens
        # to be constant is a property of the export, not of the callsite: one month's sample
        # may hold it fixed and the next may not, and an id that flips between them would make
        # every callsite look new. The id therefore keys on the surface that is stable by
        # construction - the system template, the tool set and the shape.
        user_keys = [filtered_user[p.request_id] for p in member_prints]
        system_template = list(template)
        if any(user_keys):
            user_template, user_variable = _template(
                user_keys, line_text, prints[longest].ordered_user_lines)
            if user_template:
                template = template + ["[user turn]"] + user_template
            variable += user_variable

        base_id = stable_node_id(app_id, system_template, prints[longest].tools,
                                 prints[longest].shape)
        minted[base_id] += 1
        node = Node(
            node_id=(base_id if minted[base_id] == 1 else f"{base_id}#{minted[base_id]}"),
            app_id=app_id,
            request_ids=[calls[i].request_id for i in members],
            template=template, variable_lines=variable,
            distinct_prompts=len({(filtered[prints[i].request_id],
                                   filtered_user[prints[i].request_id])
                                  for i in members}),
            medoid_request_id=calls[longest].request_id,
            signature=fp.identity_signature(member_prints),
        )
        if minted[base_id] > 1:
            node.quality.reasons.append(
                f"another cluster derived the same identity ({base_id}); single-linkage did not "
                f"join them, so they are reported apart")
        node.quality = _quality(root, point_cluster, scores, len(points), node, template)
        if root < 0:
            node.quality.verdict = "suspect"
            node.quality.reasons.append(
                f"beyond the {MAX_LINKAGE_POINTS:,}-point comparison limit for this agent - "
                f"these calls were never compared with anything and are reported unprofiled")
        if minted[base_id] > 1:
            node.quality.reasons.append(
                f"shares its identity material with {base_id}")
        node.confidence = ("established" if node.size >= 30 and node.quality.verdict == "ok"
                           else "provisional" if node.size >= 8 else "weak")

        # Below the floor but coherent: report it, marked as small rather than hidden. Being
        # quiet is not being noise, and a callsite nobody is told about cannot be acted on.
        if len(members) >= min_size:
            nodes.append(node)
        elif is_coherent(node):
            node.confidence = "small"
            node.quality.reasons.append(
                f"only {node.size} calls, below this agent's floor of {min_size} - reported "
                f"because it has a {len(node.template)}-line shared template, so it is a quiet "
                f"callsite rather than noise")
            nodes.append(node)
        else:
            residual.append(node)

    ambiguous.sort(key=lambda p: -p.score)
    return GroupingResult(
        nodes=nodes, residual=residual, ambiguous=ambiguous, doc_frequency=df,
        line_text=line_text, min_size=min_size,
        df_histogram=df_histogram(df, len(calls)),
        unlinked_calls=unlinked_calls,
        signature=fp.identity_signature(prints),
        filtered=filtered, filtered_user=filtered_user,
    )


def _quality(root: int, point_cluster: Dict[int, int], scores: Dict[Tuple[int, int], float],
             n_points: int, node: Node, template: Sequence[str]) -> NodeQuality:
    """Internal cohesion and external separation, from the pairwise scores already computed."""
    inside = [i for i, r in point_cluster.items() if r == root]
    internal, external = [], []
    for i in inside:
        for j in range(n_points):
            if j == i:
                continue
            score = scores.get((min(i, j), max(i, j)))
            if score is None:
                continue
            (internal if point_cluster.get(j) == root else external).append(score)

    quality = NodeQuality(
        min_internal=min(internal) if internal else 1.0,
        mean_internal=sum(internal) / len(internal) if internal else 1.0,
        nearest_other=max(external) if external else 0.0,
        distinct_ratio=node.distinct_prompts / max(node.size, 1),
        template_lines=len(template),
    )
    quality.margin = quality.min_internal - quality.nearest_other
    return assess(quality, node.size)


def merge_nodes(result: GroupingResult, pairs: Iterable[Tuple[str, str]]) -> int:
    by_id = {n.node_id: n for n in result.nodes}
    merged = 0
    for left_id, right_id in pairs:
        left, right = by_id.get(left_id), by_id.get(right_id)
        if left is None or right is None or left is right:
            continue
        keep, drop = (left, right) if left.size >= right.size else (right, left)
        keep.request_ids.extend(drop.request_ids)
        keep.distinct_prompts += drop.distinct_prompts
        result.nodes.remove(drop)
        by_id.pop(drop.node_id, None)
        merged += 1
    return merged


def promote_residual(result: GroupingResult, node_ids: Iterable[str]) -> int:
    wanted = set(node_ids)
    promoted = [n for n in result.residual if n.node_id in wanted]
    for node in promoted:
        result.residual.remove(node)
        node.confidence = "weak"
        result.nodes.append(node)
    return len(promoted)
