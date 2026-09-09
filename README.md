# Agent Profiler

Reads a litellm/BigQuery CSV export and tells you, for every agent flowing through your gateway:
**what its LLM callsites are**, what is template versus injected in each request, how much a
provider cache could hold, and how the calls chain into runs — with no instrumentation of the
calling applications.

A gateway log gives you a bill. `app_x spent $4,000` is not something you can act on. This turns
that into *"`app_x` has five callsites; one is 80% of the spend; 7,514 of its 7,537 tokens are
identical every call and are not being cached."*

```bash
pip install -r requirements.txt

python samples/hard_corpus.py --out samples/hard.csv   # build the test corpus (4s)
python run.py --csv samples/hard.csv                   # profile it
python run.py --csv samples/hard.csv --no-llm          # deterministic stages only
```

Each run writes one directory:

```
out/run-<id>/index.html      the estate, ranked by tokens   <- start here
out/run-<id>/<agent>.html    one page per agent, standalone
out/run-<id>/report.md       the same in prose
out/run-<id>/report.json     everything, for pipelines
out/run-<id>/calls.duckdb    every grouped request, and the static/dynamic verdict per line
```

---

## Scope: this profiles, it does not optimise

An earlier version also proposed prompt rewrites. That layer was **deleted**. Findings hung on a
grouping we could not yet trust are worse than no findings, because they are confidently wrong,
and `tests/test_separation.py` fails if those modules reappear. It comes back when the profile
half is trustworthy on real data.

So the tool measures and names. It does not tell you what to change.

## What it does

| stage | engine | what it produces |
|---|---|---|
| load | deterministic | validated chat calls; malformed rows counted, never fatal |
| single-call | deterministic | token split, provider cache-hit ratio, tool load — works at N=1 |
| group | deterministic | callsites, discovered without instrumentation |
| segment | deterministic | the stable template, and `User: <*> \| Date: <*>` for what varies |
| profile | deterministic | request anatomy, cacheable now vs recoverable, tool findings |
| chain | deterministic | runs; **declares itself unusable** where history is compacted |
| name | LLM | what each callsite is for — no deterministic method can say "this is the planner" |
| judge | LLM | is this profile *true* of its own requests? plus an intruder test on the boundary |
| registry | deterministic | seen this callsite before? edited since? |

Two LLM stages, and each earns it. Production has no answer key, so something has to read the
evidence and say whether the profile holds.

## The rule the design rests on

> **LLM output never silently influences a deterministic measurement.**

Judgments are applied through named functions (`merge_nodes`, `promote_residual`) and land in the
audit trail, so a reader can always tell what was measured and what was decided. Enforced by
`tests/test_separation.py`, not by convention — if a judgment could quietly change a measured
number, we would lose the ability to tell whether the method works.

## Configuration

`mapping.yaml` declares where the fields live. It should be the only file that changes when a
different export arrives.

```yaml
columns:
  request:  request_payload
  response: response_payload
app_id_paths:                       # a JSON path, not a column; first non-null wins
  - "request_payload.labels.vz_key_name"
  - "client_metadata.headers.x-subject"
```

Columns named here that the export does not have are reported and skipped, not fatal. If there
is no `request_id` column, one is derived from the row's own content — never from its position,
which would rename every call when the same window is re-exported in a different order.

Disagreements between the two id paths are counted and reported: the agent id is the outer
partition, so a high rate would undermine everything downstream.

`LLM_KEY` is read from `.env` (or the environment) and never logged. Without it the deterministic
stages still run; pass `--no-llm` to skip the LLM explicitly.

## Have we profiled this callsite already?

Grouping cannot answer that from a single request. It is a batch operation: document frequency
decides which lines are template and which are payload, and with one request every line looks
stable, so the derived id differs every time. **A node id is not computable from one request.**

`match.py` answers it by lookup instead - compare the request against templates already known:

```python
from profiler.match import Matcher
m = Matcher("registry.sqlite")
result = m.match(call)        # .seen_before, .node_id, .containment, .verdict
```

The measure is **containment**, not Jaccard: a request belongs to a callsite when it *contains*
that callsite's template, whatever payload it also carries. Jaccard punishes the request for its
payload, so a 400-line retrieved document drowns a 6-line template and the callsite reads as new
on every call.

Measured by feeding an export's own requests back one at a time, against callsites profiled from
it: **1,214 correct, 0 wrong, 84 refused, 2 recovered** that grouping had left below its size
floor. The 84 are one callsite whose stored template is empty - a sliding-window agent with no
system prompt and a different user turn every call. It has no stable content, so nothing can
recognise it from content.

Retrieval is on each callsite's *rarest* template lines, so it does not slow down as the estate
grows: at 200,000 callsites a lookup scores **1 candidate in 0.36 ms**. It refuses rather than
guesses below 80% containment, when two callsites fit within 0.10 of each other, and when a
template's lines are all common across the estate.

## Checking the profile instead of believing it

The point is not to be told there are twenty callsites. It is to open one and satisfy yourself.

```bash
python inspect_calls.py                            # list every callsite
python inspect_calls.py --node <id>                # fingerprint, regions, static vs varying
python inspect_calls.py --node <id> --all          # EVERY request in full, each line marked
python inspect_calls.py --node <id> --export dir/  # one file per request, to diff
python inspect_calls.py --sql "SELECT ..."         # the store directly

python serve.py                                    # browse the store live in a browser
```

`--all` prints the shared template once and then every request whole, each line marked `=` shared
by all / `~` differs / `?` not in the split. Showing only the differences would ask you to take
on trust that the rest is identical — which is the claim being checked.

`calls.duckdb` holds `calls`, `nodes`, `regions` and **`region_lines`** — every line of every
region marked static or dynamic *with the count behind it*. "This line is in 100 of 100 requests
so we called it template; that one is in 1, so we called it payload" is a claim anyone can check
with a `SELECT`, and disagree with.

`serve.py` reads that store live, so nothing is sampled: request 2,400 of 2,400 is one click
away. Read-only, bound to `127.0.0.1`, stdlib only.

## Why the grouping works the way it does

Every parameter below was chosen by measurement. What survived is in
[STRESS_FINDINGS.md](STRESS_FINDINGS.md); the raw spike lives in `sandbox/spike/` and is not part
of this repo.

- **Identity is five views**, not the system prompt alone: prompt lines (0.60), user template
  (0.20), tool set (0.25), message shape (0.10), response format (0.05), renormalised over
  whichever apply. Keying on the system prompt merged any two callsites sharing a base template
  and collapsed every agent whose real work is in the user turn.
- **A disjoint discriminating view vetoes the pair.** A router and an executor sharing a base
  prompt and sharing *no* tools scored 0.75 and merged. Weighted averaging cannot express "this
  disagreement is decisive", so such a pair is capped at 0.50 — inside the adjudication band,
  where the LLM can still overrule it on the evidence.
- **Ignore lines seen once per app before comparing.** Changes only the comparison key; nothing
  is removed from stored data. The user turn needs a much higher floor (5% of the agent's calls)
  because a replayed history makes payload look like template.
- **Similarity threshold ≤ 0.6.** Above it, one callsite silently splits in two, then five.
- **Plain Jaccard, not TF-IDF.** IDF over-weights a per-tenant header that repeats a handful of
  times, collapsing a chat app from full coverage to none.
- **A capped size floor**, `max(3, min(2% of calls, 25))`, plus a rescue for small-but-coherent
  clusters. A fixed floor of 20 hid 38 of 56 callsites; an uncapped 2% hid all 30 quiet callsites
  of an agent with one busy hot path. The floor exists to reject noise, not quiet callsites.
- **Node ids are content-derived** — a hash of template, tools and shape — so they survive across
  exports. Whether a user turn happens to be constant is a property of the export, not the
  callsite, so it stays out of the id.

## What a run is scored against

Two adversarial corpora. The generators are committed; their output is not, because
`samples/*.py` rebuilds them in seconds.

| corpus | rows | callsites | purity | completeness | coverage |
|---|---|---|---|---|---|
| `hard_corpus.py` | 1,300 | 21 | 0.928 | 0.968 | 0.998 |
| `extreme_corpus.py` | 7,575 | 37 | 0.998 | 1.000 | 1.000 |

```bash
python stress.py --csv samples/hard.csv      # score against ground truth, offline
python diagnose.py --csv samples/hard.csv    # separation histogram, no ground truth needed
```

`stress.py` prints **what failed** — which callsites were split, which nodes are mixed, which
agents cannot have their runs reconstructed — rather than a single number. `diagnose.py` needs no
answer key, so it is the one to point at a real export first.

## Verification gate

Every node is checked twice against its own sampled requests: does the stated profile hold, and
can an intruder from the nearest competing callsite be spotted. Below 80% confirmation the run is
marked **NOT PASSED** and `run.py` exits non-zero. With no LLM key the run is **UNVERIFIED**,
which is neither passed nor failed.

Verification *reports*; it never edits. A flagged profile stays in the report, marked — so a
systematic failure is visible instead of looking like a clean run.

## Audit trail

`audit/run-<id>.jsonl` records every step, every LLM call (model, tokens, cost, latency, prompt
hash) and every decision made by judgment rather than measurement. `audit/audit-<id>.md` renders
it. Responses are cached on disk, so a re-run costs nothing and reproduces exactly.

## Layout

```
run.py                 profile an export
inspect_calls.py       read a group, request by request
serve.py               browse the store live
stress.py              score against ground truth
diagnose.py            separation histogram, no ground truth needed
mapping.yaml           column mapping

profiler/
  load.py              CSV -> validated calls
  single.py            per-call analysis
  fingerprint.py       the five views, and how two requests are compared
  identity.py          noise filter, clustering, stable ids, quality signals
  segment.py           static/dynamic split, dynamic field templates
  metrics.py           cache, waste, and per-callsite anatomy
  chain.py             run reconstruction, and where it is impossible
  registry.py          memory across runs; re-identification across prompt edits
  match.py             one request -> which known callsite, if any (containment lookup)
  store.py             every grouped request -> DuckDB
  report.py            Markdown + JSON
  report_html.py       index + one page per agent
  audit.py             event log
  pipeline.py          orchestration
  llm/
    client.py          the one place that calls a model
    adjudicate.py      borderline pairs, residual rescue, callsite naming
    judge.py           per-node profile audit and intruder test

samples/               corpus generators (their CSV output is gitignored)
tests/                 92 tests
```

## Known limits

Stated here because a profile that hides its unreliable parts is worse than none.

- **Compacting and windowing agents cannot have their runs reconstructed.** They destroy the
  shared message prefix *before the gateway sees the request* (measured: 0.344 and 0.054 against
  1.000 where history is replayed). Not recoverable from logs — it needs a run-id header.
- **Two callsites identical in content merge.** If they differ only by a per-call header that
  noise filtering removes, nothing in the request separates them. Attribution blurs; the cache
  finding survives.
- **Fingerprint weights are judgement.** No sweep has been run.
- **Nothing here has yet seen a real export.** Every number above describes a fixture written by
  the same person who wrote the code.
