"""The shapes the standard corpus does not reach.

`hard_corpus.py` is adversarial but small: nothing over 10k tokens, six agents, 1,300 well-formed
rows. A production estate is none of those things, and every scale guard in the code was written
and then never exercised:

    MAX_IDENTITY_CHARS   200,000   what happens past it had never been observed
    MAX_LINKAGE_POINTS     4,000   silently ate every call beyond it until this corpus
    max_rows_per_app       1,000   caps an agent; the cap's effect on grouping was unmeasured
    USER_DF_FRACTION        0.05   a proportional floor, only ever tested at n=254

Kept in its own file and its own CSV so the everyday corpus stays a four-second feedback loop.
These rows are large.

    python samples/extreme_corpus.py --out samples/extreme.csv
    python stress.py --csv samples/extreme.csv --max-rows-per-app 100000

Shapes, and the guard each one leans on:

  contract_review     200k-token user turn            MAX_IDENTITY_CHARS
  repo_qa             120k static system prompt       the cache case at scale
  repo_qa_uncached    the same, with a per-call header in front - cache destroyed at scale
  hot_path + tail_NN  one callsite of 2,400 calls beside 30 of 4-25   the adaptive size floor
  release_notes       one prompt edited five times across the export  the registry's real job
  adhoc_analysis      4,500 near-unique prompts       MAX_LINKAGE_POINTS
  invoice_extract     retry storms within seconds     retry detection
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import timedelta
from typing import List

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from samples.hard_corpus import RNG, T0, Builder, Row, Truth, _text


def app_giant() -> Builder:
    """Requests in the 50k-200k token range: the top of the stated production range."""
    b = Builder("app_giant")

    for i in range(12):
        b.add("contract_review", [
            {"role": "system", "content": "\n".join([
                "## Role", "You review commercial contracts for the legal desk.",
                "## Method", "Work clause by clause.", "Quote the clause you rely on.",
                "Never assert a governing law that is not stated.",
                "## Output", "Return findings as a markdown table."])},
            {"role": "user", "content": f"CONTRACT {i}:\n{_text(200_000, 30_000 + i)}"}],
            offset_s=i * 400, reply="| clause | risk |")

    # an enormous STATIC context - the cache case at scale, done right
    repo = _text(120_000, 31_000)
    for i in range(12):
        b.add("repo_qa", [
            {"role": "system", "content":
                f"## Repository context\n{repo}\n## Task\nAnswer questions about this code."},
            {"role": "user", "content": f"Question {i}: where is settlement retried?"}],
            offset_s=10_000 + i * 400, reply="In the retry module.")

    # the same, with a per-call header in front: the cache broken at 120k tokens
    for i in range(12):
        b.add("repo_qa_uncached", [
            {"role": "system", "content":
                f"session {i} | user u{i % 5} | {(T0 + timedelta(minutes=i)).isoformat()}\n"
                f"## Repository context\n{repo}\n## Task\nAnswer questions about this code."},
            {"role": "user", "content": f"Question {i}: where is the ledger written?"}],
            offset_s=20_000 + i * 400, reply="In the ledger module.")
    return b


def app_swarm() -> Builder:
    """Many callsites in one agent, brutally skewed.

    The size floor is max(3, 2% of the agent's calls). At ~3,000 calls that is 60, and every
    one of the thirty small callsites falls under it - so the question is whether they are
    reported as unprofiled or quietly disappear.
    """
    b = Builder("app_swarm")
    for i in range(2400):
        b.add("hot_path", [
            {"role": "system", "content": "Classify the message. Return one word."},
            {"role": "user", "content": f"message {i}"}], offset_s=i, reply="ok")
    for site in range(30):
        for i in range(RNG.randint(4, 25)):
            b.add(f"tail_{site:02d}", [
                {"role": "system", "content": "\n".join([
                    "## Role", f"You handle workflow {site} for the operations desk.",
                    f"Step {site}: apply the desk rules for this workflow.",
                    "Escalate anything you cannot resolve.",
                    f"Reference table {site} before answering."])},
                {"role": "user", "content": f"case {site}-{i}"}],
                offset_s=5000 + site * 100 + i, reply="done")
    return b


def app_drift() -> Builder:
    """One prompt edited five times across the export - what the registry is really for."""
    b = Builder("app_drift")
    base = ["## Role", "You draft release notes from merged pull requests.",
            "Group changes by component.", "Use the past tense."]
    additions = ["Link the pull request number.", "Call out breaking changes first.",
                 "Mention the author for external contributions.",
                 "Keep each entry under twenty words."]
    for version in range(5):
        prompt = "\n".join(base + additions[:version])
        for i in range(40):
            b.add("release_notes", [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"PRs merged in build {version}-{i}"}],
                offset_s=version * 20_000 + i * 60, reply="- fixed a thing", ambiguous=True)
    return b


def app_highvariance() -> Builder:
    """Thousands of near-unique prompts in one agent - the case that hit the linkage guard."""
    b = Builder("app_highvariance")
    areas = ("risk", "ops", "finance", "legal")
    for i in range(4500):
        b.add("adhoc_analysis", [
            {"role": "system", "content":
                f"You are analyst {i % 900}. Focus area: {areas[i % 4]} {i}.\n"
                f"Answer in one paragraph."},
            {"role": "user", "content": f"Dataset {i}: {_text(300, 40_000 + i)}"}],
            offset_s=i * 2, reply="An analysis.")
    return b


def app_retrystorm() -> Builder:
    """The same request repeated within seconds, beside genuine repeats hours apart."""
    b = Builder("app_retrystorm")
    prompt = "\n".join(["## Role", "You extract totals from invoices.",
                        "Return JSON with an `amount` field.", "Never guess a currency."])
    for run in range(30):
        messages = [{"role": "system", "content": prompt},
                    {"role": "user", "content": f"Invoice {run}"}]
        for attempt in range(RNG.choice([1, 1, 1, 3, 5])):
            b.add("invoice_extract", list(messages), offset_s=run * 600 + attempt,
                  reply='{"amount": 1}')
    return b


BUILDERS = (app_giant, app_swarm, app_drift, app_highvariance, app_retrystorm)


def build():
    rows: List[Row] = []
    truth: Truth = {}
    ambiguous: set = set()
    for make in BUILDERS:
        b = make()
        rows.extend(b.rows)
        truth.update(b.truth)
        ambiguous |= b.ambiguous
    rows.sort(key=lambda r: r["startTime"])
    return rows, truth, ambiguous


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=os.path.join(HERE, "samples", "extreme.csv"))
    args = parser.parse_args(argv)

    rows, truth, ambiguous = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    truth_path = os.path.splitext(args.out)[0] + ".truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump({"callsite_of": truth, "ambiguous": sorted(ambiguous)}, handle)

    print(f"{len(rows):,} rows, {len(set(truth.values()))} true callsites, "
          f"{os.path.getsize(args.out) / 1e6:.1f} MB")
    print(f"  {os.path.relpath(args.out, HERE)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
