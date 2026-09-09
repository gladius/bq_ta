"""Score the deterministic grouping against a corpus with known answers.

    python samples/hard_corpus.py --out samples/hard.csv
    python stress.py --csv samples/hard.csv

Deliberately deterministic-only and offline: this measures the part that has to be right before
any LLM sees the data. It prints what FAILED - which true callsites were split, which nodes are
mixed, which agents cannot have their runs reconstructed - rather than a single score.

Ambiguous cases (a prompt mid-rollout, where "one callsite or two" has no correct answer) are
excluded from the score and reported on their own.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from typing import Any, Dict, List, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from profiler.chain import reconstruct
from profiler.config import load_settings
from profiler.identity import group_app
from profiler.load import group_by_app, load_calls
from profiler.metrics import compute
from profiler.segment import segment
from profiler.metrics import prose_document


def score_app(nodes, residual, truth: Dict[str, str], skip: set,
              app_request_ids: set) -> Dict[str, Any]:
    """Purity and completeness, weighted by calls, ignoring ambiguous requests."""
    # keyed by position, not node_id: two clusters can carry the same id and one would
    # silently overwrite the other, flattering the score
    assign: Dict[int, List[str]] = {}
    for i, node in enumerate(nodes):
        members = [r for r in node.request_ids if r in truth and r not in skip]
        if members:
            assign[i] = members

    scored = [r for r in app_request_ids if r in truth and r not in skip]
    if not scored:
        return {}

    # purity: within one node, how many calls come from its dominant true callsite
    pure_hits = 0
    mixed = []
    for node_id, members in assign.items():
        counts = collections.Counter(truth[r] for r in members)
        top, n = counts.most_common(1)[0]
        pure_hits += n
        if len(counts) > 1:
            mixed.append({"node": nodes[node_id].node_id, "dominant": top,
                          "contamination": dict(counts.most_common()[1:])})

    # completeness: for one true callsite, how many of its calls land in one node
    node_of = {r: nid for nid, members in assign.items() for r in members}
    complete_hits = 0
    split = []
    by_true: Dict[str, List[str]] = collections.defaultdict(list)
    for r in scored:
        by_true[truth[r]].append(r)
    for callsite, members in by_true.items():
        counts = collections.Counter(node_of[r] for r in members if r in node_of)
        if not counts:
            split.append({"callsite": callsite, "calls": len(members), "nodes": 0,
                          "note": "no calls reached any node"})
            continue
        complete_hits += counts.most_common(1)[0][1]
        if len(counts) > 1:
            split.append({"callsite": callsite, "calls": len(members),
                          "nodes": len(counts), "sizes": [n for _, n in counts.most_common()]})

    assigned = sum(len(m) for m in assign.values())
    return {
        "true_callsites": len(by_true), "nodes": len(assign),
        "calls_scored": len(scored), "calls_assigned": assigned,
        "coverage": round(assigned / len(scored), 3),
        "purity": round(pure_hits / assigned, 3) if assigned else 0.0,
        "completeness": round(complete_hits / len(scored), 3),
        "mixed_nodes": mixed, "split_callsites": split,
        "residual_calls": sum(n.size for n in residual),
    }


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default=os.path.join(HERE, "samples", "hard.csv"))
    parser.add_argument("--truth", default=None)
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--max-rows-per-app", type=int, default=100000)
    args = parser.parse_args(argv)

    truth_path = args.truth or os.path.splitext(args.csv)[0] + ".truth.json"
    with open(truth_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    truth: Dict[str, str] = payload["callsite_of"]
    ambiguous = set(payload.get("ambiguous") or [])

    settings = load_settings(use_llm=False, tau=args.tau)
    settings.mapping.max_rows_per_app = args.max_rows_per_app

    started = time.time()
    calls, load_report = load_calls(args.csv, settings.mapping)
    by_app = group_by_app(calls)
    print(f"loaded {load_report.loaded:,}/{load_report.total_rows:,} rows · "
          f"{len(by_app)} apps · {time.time() - started:.1f}s")
    skipped = load_report.as_dict()["skipped"]
    if any(skipped.values()):
        print(f"  skipped: {skipped}")
    print()

    totals = collections.Counter()
    all_mixed: List[Tuple[str, Dict]] = []
    all_split: List[Tuple[str, Dict]] = []
    chain_notes: List[Tuple[str, Any]] = []

    header = f"{'agent':<16} {'calls':>7} {'true':>5} {'found':>6} {'purity':>7} " \
             f"{'complete':>9} {'cover':>6}"
    print(header)
    print("-" * len(header))

    for app_id in sorted(by_app):
        app_calls = by_app[app_id]
        grouping = group_app(app_id, app_calls, settings.tau, settings.min_doc_freq,
                             settings.min_node_fraction, settings.min_node_floor,
                             settings.ambiguous_low, settings.ambiguous_high)
        result = score_app(grouping.nodes, grouping.residual, truth, ambiguous,
                           {c.request_id for c in app_calls})
        if not result:
            print(f"{app_id:<16} {len(app_calls):>7} {'-':>5} "
                  f"{len(grouping.nodes):>6}   (all ambiguous)")
            continue

        print(f"{app_id:<16} {len(app_calls):>7,} {result['true_callsites']:>5} "
              f"{result['nodes']:>6} {result['purity']:>7.3f} "
              f"{result['completeness']:>9.3f} {result['coverage']:>6.3f}")

        totals["calls"] += result["calls_scored"]
        totals["pure"] += result["purity"] * result["calls_assigned"]
        totals["complete"] += result["completeness"] * result["calls_scored"]
        totals["assigned"] += result["calls_assigned"]
        totals["true"] += result["true_callsites"]
        totals["found"] += result["nodes"]
        all_mixed += [(app_id, m) for m in result["mixed_nodes"]]
        all_split += [(app_id, s) for s in result["split_callsites"]]

        node_of = {r: n.node_id for n in grouping.nodes for r in n.request_ids}
        chain_notes.append((app_id, reconstruct(app_calls, node_of)))

    print("-" * len(header))
    if totals["calls"]:
        print(f"{'OVERALL':<16} {totals['calls']:>7,} {totals['true']:>5} "
              f"{totals['found']:>6} {totals['pure'] / max(totals['assigned'], 1):>7.3f} "
              f"{totals['complete'] / totals['calls']:>9.3f} "
              f"{totals['assigned'] / totals['calls']:>6.3f}")
    print()

    print("=== callsites SPLIT across several nodes " + "=" * 34)
    if not all_split:
        print("  none")
    for app_id, s in all_split:
        print(f"  {app_id:<16} {s['callsite']:<22} {s['calls']:>4} calls -> "
              f"{s['nodes']} nodes {s.get('sizes', '')}")
    print()

    print("=== nodes MIXING several callsites " + "=" * 40)
    if not all_mixed:
        print("  none")
    for app_id, m in all_mixed:
        print(f"  {app_id:<16} {m['dominant']:<22} contaminated by {m['contamination']}")
    print()

    print("=== run reconstruction " + "=" * 52)
    for app_id, report in chain_notes:
        print(f"  {app_id:<16} {report.verdict:<12} {report.runs:>5} runs, "
              f"mean {report.mean_steps} steps")
        if report.note:
            print(f"                   {report.note[:110]}")
    print()

    if ambiguous:
        print(f"=== excluded as ambiguous by construction " + "=" * 33)
        print(f"  {len(ambiguous)} requests (a prompt mid-rollout: 'one callsite or two' has "
              f"no correct answer)")
    print(f"\ntotal {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
