"""Per-agent diagnostic: is the grouping real, or is it fooling us?

    python diagnose.py --csv samples/sample.csv                 # every agent, one line each
    python diagnose.py --csv samples/sample.csv --agent app_chat  # one agent, in full

In production there is no answer key, so "did it work" has to be answerable from the shape of
the data alone. The single most telling picture is the **pairwise similarity histogram**:

  - two clear humps, one high (calls inside a callsite) and one low (calls in different
    callsites), separated by a gap -> the callsites are real and the threshold sits in the gap
  - one continuous smear with the threshold cutting through the middle -> the grouping is an
    arbitrary slice of a continuum, and no threshold would be right

Everything else here supports reading that one picture.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from profiler import fingerprint as fp
from profiler.config import load_settings
from profiler.identity import group_app
from profiler.load import Call, group_by_app, load_calls
from profiler.segment import segment
from profiler.single import text_of

BARS = " .:-=+*#%@"


def sparkline(counts: Sequence[int], width: int = 40) -> str:
    top = max(counts) or 1
    return "".join(BARS[min(len(BARS) - 1, int(c / top * (len(BARS) - 1)))] for c in counts)


def histogram(values: Sequence[float], bins: int = 20) -> List[int]:
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, max(0, int(value * bins)))
        counts[index] += 1
    return counts


def bar(count: int, total: int, width: int = 30) -> str:
    filled = 0 if not total else int(count / total * width)
    return "#" * filled + "." * (width - filled)


def separation(within, across):
    """How cleanly the two populations part, reported two ways.

    `gap` is the bulk separation - the 10th percentile of within-callsite scores against the
    90th percentile of across-callsite scores. `worst` is min(within) - max(across).

    Only `worst` was reported before, and it is the wrong headline. Single linkage joins a node
    through a chain, so ONE weakly-linked pair drags min(within) to the floor: app_agent grouped
    at purity 1.000 and completeness 1.000 and was still labelled OVERLAP. A statistic that
    contradicts a perfect result will cost more confidence than it earns. `worst` is kept as the
    pessimistic bound, beside the bulk figure rather than instead of it.
    """
    if not within or not across:
        return None, None

    def pct(values, q):
        ordered = sorted(values)
        i = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
        return ordered[i]

    return pct(within, 0.10) - pct(across, 0.90), min(within) - max(across)


def analyse_agent(app_id: str, calls: Sequence[Call], settings, truth: Dict[str, str],
                  verbose: bool) -> Dict[str, object]:
    result = group_app(app_id, calls, settings.tau, settings.min_doc_freq,
                       settings.min_node_fraction, settings.min_node_floor,
                       settings.ambiguous_low, settings.ambiguous_high)

    # every pairwise score, labelled by whether the two calls ended up together
    prints = [fp.build(c) for c in calls]
    by_request = {p.request_id: p for p in prints}
    node_of: Dict[str, str] = {}
    for node in result.nodes:
        for request_id in node.request_ids:
            node_of[request_id] = node.node_id

    sample = prints if len(prints) <= 220 else prints[:: max(1, len(prints) // 220)]
    within: List[float] = []
    across: List[float] = []
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            # the same filtered keys the grouper used, or the picture describes
            # a comparison that never happened
            score = fp.similarity(sample[i], sample[j],
                                  result.filtered, result.filtered_user).score
            same = (node_of.get(sample[i].request_id) is not None
                    and node_of.get(sample[i].request_id) == node_of.get(sample[j].request_id))
            (within if same else across).append(score)

    summary = {
        "app_id": app_id, "calls": len(calls), "nodes": len(result.nodes),
        "residual": sum(n.size for n in result.residual),
        "within": within, "across": across, "result": result,
    }
    if truth:
        true_sites = {truth.get(c.request_id) for c in calls}
        assigned = sum(n.size for n in result.nodes)
        hit = sum(max(Counter(truth.get(r) for r in n.request_ids).values())
                  for n in result.nodes) if result.nodes else 0
        summary["true_callsites"] = len(true_sites)
        summary["purity"] = hit / assigned if assigned else float("nan")
    return summary


def print_agent(summary, settings, calls: Sequence[Call], truth: Dict[str, str]) -> None:
    result = summary["result"]
    app_id = summary["app_id"]
    within, across = summary["within"], summary["across"]

    print("=" * 84)
    print(f"AGENT  {app_id}")
    print("=" * 84)
    print(f"{summary['calls']} calls -> {summary['nodes']} callsites, "
          f"{summary['residual']} calls unassigned (floor {result.min_size})")
    if "purity" in summary:
        print(f"ground truth available: {summary['true_callsites']} true callsites, "
              f"purity {summary['purity']:.3f}")
    sig = result.signature
    print(f"identity available: {sig.get('with_system_prompt')}/{summary['calls']} have a system "
          f"prompt, median {sig.get('median_prompt_lines')} prompt lines, "
          f"{sig.get('distinct_tool_sets')} tool sets, {sig.get('distinct_shapes')} shapes")
    print()

    # --- the picture that matters -------------------------------------------------------
    print("PAIRWISE SIMILARITY  (the question: are there two humps, or one smear?)")
    bins = 20
    w_hist, a_hist = histogram(within, bins), histogram(across, bins)
    total = max(max(w_hist, default=0), max(a_hist, default=0)) or 1
    print(f"  {'score':<8}{'same callsite':<34}{'different callsite'}")
    for i in range(bins):
        low = i / bins
        marker = " <- threshold" if abs(low - settings.tau) < 1.0 / bins / 2 else ""
        print(f"  {low:.2f}-{low + 1/bins:.2f} "
              f"{bar(w_hist[i], total, 28)} {bar(a_hist[i], total, 28)}{marker}")
    if within and across:
        gap, worst = separation(within, across)
        print(f"\n  within : n={len(within):<7} min={min(within):.3f} mean="
              f"{sum(within)/len(within):.3f}")
        print(f"  across : n={len(across):<7} max={max(across):.3f} mean="
              f"{sum(across)/len(across):.3f}")
        print(f"  worst case: {worst:+.3f} (closest single pair across the boundary)")
        if gap > 0:
            print(f"  SEPARATION: clean gap of {gap:.3f} between the bulk of the two groups "
                  f"-> the callsites are genuinely distinct")
        else:
            print(f"  SEPARATION: the groups OVERLAP by {-gap:.3f} -> the threshold is cutting "
                  f"through a continuum, not a gap")
    print()

    # --- where the prompt text sits ------------------------------------------------------
    hist = result.df_histogram
    total_lines = sum(hist.values()) or 1
    print("PROMPT-LINE FREQUENCY  (how much of the text is stable vs per-call noise)")
    for band, count in hist.items():
        print(f"  {band:<12} {bar(count, total_lines, 30)} {count:>6}  "
              f"({count/total_lines:.0%})")
    middle = hist.get("10-40%", 0) + hist.get("40-90%", 0)
    if middle / total_lines > 0.25:
        print(f"  NOTE: {middle/total_lines:.0%} of lines sit in the middle bands - conditional "
              f"blocks, not clean template-vs-payload. Grouping is working harder than it was "
              f"validated for.")
    print()

    # --- per node ------------------------------------------------------------------------
    print("CALLSITES")
    print(f"  {'verdict':<9}{'calls':>6}{'distinct':>9}{'tmpl':>6}{'minInt':>8}{'nearest':>8}"
          f"{'margin':>8}  notes")
    for node in sorted(result.nodes, key=lambda n: -n.size):
        q = node.quality
        print(f"  {q.verdict:<9}{node.size:>6}{node.distinct_prompts:>9}"
              f"{len(node.template):>6}{q.min_internal:>8.2f}{q.nearest_other:>8.2f}"
              f"{q.margin:>8.2f}  {'; '.join(q.reasons)[:60]}")
    print()

    verdicts = Counter(n.quality.verdict for n in result.nodes)
    print(f"  quality: {dict(verdicts)}")
    if result.ambiguous:
        print(f"  {len(result.ambiguous)} borderline pairs sit in the band where the threshold "
              f"is unreliable (an LLM would be asked about these)")
    print()


def print_node_detail(summary, calls: Sequence[Call], top: int = 2) -> None:
    """For the largest nodes: what is template, what varies, and a real example."""
    result = summary["result"]
    by_id = {c.request_id: c for c in calls}
    for node in sorted(result.nodes, key=lambda n: -n.size)[:top]:
        members = [by_id[r] for r in node.request_ids if r in by_id]
        seg = segment(["\n".join(text_of(m.get("content")) for m in c.messages
                                 if m.get("role") in ("system", "developer", "user"))
                       for c in members])
        print("-" * 84)
        print(f"NODE {node.node_id}   {node.size} calls   quality={node.quality.verdict}")
        print("-" * 84)
        print(f"  STATIC ({len(seg.static_lines)} lines, sent on every call):")
        for line in seg.static_lines[:8]:
            print(f"    | {line[:96]}")
        if len(seg.static_lines) > 8:
            print(f"    | ... {len(seg.static_lines) - 8} more")
        print(f"  VARIES ({len(seg.fields)} field patterns):")
        for field in seg.fields[:5]:
            example = field.examples[0][:40] if field.examples else ""
            print(f"    | {field.template[:70]:<72} x{field.occurrences}  e.g. {example}")
        print()


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--agent", default=None, help="one agent id; omit for all")
    parser.add_argument("--truth", default=None, help="optional ground-truth csv")
    parser.add_argument("--detail", type=int, default=2, help="nodes to expand per agent")
    args = parser.parse_args(argv)

    csv.field_size_limit(10 ** 9)
    settings = load_settings(use_llm=False)
    calls, report = load_calls(args.csv, settings.mapping)
    truth: Dict[str, str] = {}
    guess = args.truth or os.path.splitext(args.csv)[0] + "_truth.csv"
    if os.path.exists(guess):
        with open(guess, encoding="utf-8") as handle:
            truth = {r["request_id"]: r["gt_callsite"] for r in csv.DictReader(handle)}

    by_app = group_by_app(calls)
    targets = [args.agent] if args.agent else sorted(by_app)

    if not args.agent:
        print(f"{report.loaded:,} calls, {len(by_app)} agents\n")
        print(f"  {'agent':<24}{'calls':>7}{'nodes':>7}{'resid':>7}{'gap':>8}  quality")
        for app_id in targets:
            s = analyse_agent(app_id, by_app[app_id], settings, truth, False)
            gap, _ = separation(s["within"], s["across"])
            verdicts = Counter(n.quality.verdict for n in s["result"].nodes)
            flag = "" if gap is None or gap > 0 else "  <- OVERLAP"
            print(f"  {app_id:<24}{s['calls']:>7}{s['nodes']:>7}{s['residual']:>7}"
                  f"{(f'{gap:+.3f}' if gap is not None else '  n/a'):>8}  {dict(verdicts)}{flag}")
        print("\nrun with --agent <id> for the full picture on one agent")
        return 0

    for app_id in targets:
        if app_id not in by_app:
            print(f"no such agent: {app_id}")
            return 1
        summary = analyse_agent(app_id, by_app[app_id], settings, truth, True)
        print_agent(summary, settings, by_app[app_id], truth)
        print_node_detail(summary, by_app[app_id], args.detail)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
