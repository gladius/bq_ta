# Agent Profiler

Reads a litellm/BigQuery CSV export and tells you, for every agent flowing through your gateway:
what its LLM callsites are, what is stable versus injected in each prompt, how much of it a
provider cache could hold, and **the rewritten prompt that would make that true** — with no
instrumentation of the calling applications.

```bash
pip install -r requirements.txt
python run.py --csv samples/sample.csv              # full pipeline
python run.py --csv samples/sample.csv --no-llm     # deterministic stages only
```

Outputs land in `out/` (Markdown + JSON) and `audit/` (event log + rendered audit).

---

## What it does

| stage | engine | what it produces |
|---|---|---|
| load | deterministic | validated chat calls; malformed rows counted, never fatal |
| single-call | deterministic | token split, provider cache-hit ratio, tool load — works at N=1 |
| group | deterministic | callsites, discovered without instrumentation |
| segment | deterministic | the stable template, and `User: <*> \| Date: <*>` for what varies |
| profile | deterministic | cacheable now vs recoverable, wasted tokens, tool findings |
| optimise | LLM | **rewritten prompts**, mechanically checked for lost instructions |
| audit | LLM | efficiency and security findings, grounded in the measured profile |
| verify | LLM | re-checks a sample of findings against the raw rows; gates the report |

## The rule the design rests on

> **LLM output never silently influences a deterministic measurement.**

The LLM annotates, adjudicates decisions that are explicitly marked as unresolved, and verifies
afterwards. It never feeds back into clustering or the metrics. This is enforced by tests
(`tests/test_separation.py`), not by convention — if a judgment could quietly change a measured
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

Disagreements between the two id paths are counted and reported — the agent id is the outer
partition, so a high rate would undermine everything downstream.

`LLM_KEY` is read from `.env` (or the environment). Without it the deterministic stages still
run; pass `--no-llm` to skip the LLM explicitly.

## Why the grouping works the way it does

The method was chosen by measurement, not preference — the full evidence is in
[`../spike/DECISIONS.md`](../spike/DECISIONS.md) and [`../spike/ASSESSMENT.md`](../spike/ASSESSMENT.md).

- **Ignore lines seen once per app before comparing.** This changes only the comparison key;
  nothing is removed from the stored data. Without it a RAG-style callsite scores 0.107
  similarity and 200 rows stay unassigned. With it, one clean node.
- **No structural bucket key** — no archetype, no tool set, no prompt-section signature. A bucket
  key is a *hard* partition clustering cannot undo, and every structural signal tested fragments
  under real conditions: a section signature split one callsite into 27 buckets; a tool-set key
  lost an entire callsite to per-call tool injection. Removing bucketing beat seven alternatives.
- **Similarity threshold ≤ 0.6.** Above it, one callsite silently splits in two, then five.
- **Plain Jaccard, not TF-IDF.** IDF over-weights a per-tenant header that repeats a handful of
  times, collapsing a chat app from full coverage to none.
- **Relative size floor.** A fixed floor of 20 hid 38 of 56 callsites; `max(3, 2% of app rows)`
  recovers them, which matters when the export is small.
- **Node ids are content-derived** (a hash of the template), so they survive across exports —
  measured at 18/18 nodes re-identified across two independent windows.

## Proposal safety

Three optimiser skills with three different safety profiles:

| skill | safety | auto-accepted? |
|---|---|---|
| `cache_prefix_rewrite` | reordering only; verified to lose no instruction | yes, if the check passes |
| `input_compression` | may reword; verified against a content-word budget | yes, if the check passes |
| `output_optimization` | changes what the model *emits* | **never** — reported for human review |

Every rewrite is checked mechanically before it is reported: `verify_rewrite` re-derives which
content words disappeared, so a rewrite that quietly drops a policy line is rejected and shown
as rejected. **Nothing is ever applied automatically.**

That split was not designed up front — the verification stage caught the original code
auto-accepting output-format changes, which would have broken downstream consumers.

## Verification gate

After the run, an LLM re-checks a sample of findings against the raw rows they came from:
does this template really describe these requests, is this region really dynamic, is this advice
correct and safe. The report leads with the result. Below 80% confirmation the run is marked
**NOT PASSED** and `run.py` exits non-zero.

Verification *reports*; it never edits. A finding it rejects stays in the report, marked — so a
systematic failure is visible instead of looking like a clean run.

## Audit trail

`audit/run-<id>.jsonl` records every step, every LLM call (model, tokens, cost, latency, prompt
hash) and every decision made by judgment rather than measurement. `audit/audit-<id>.md` renders
it. Responses are cached on disk, so a re-run costs nothing and reproduces exactly.

## Development fixture

`samples/make_sample_csv.py` exports the spike corpus into a litellm-shaped CSV, including
malformed JSON, non-chat rows, missing `usage` blocks and disagreeing id paths. It ships with
ground truth (`sample_truth.csv`), so grouping accuracy is verifiable:

```
56 nodes for 56 true callsites · purity 1.000 · coverage 1.000
```

## Layout

```
run.py                 entry point
mapping.yaml           column mapping
profiler/
  load.py              CSV -> validated calls
  single.py            per-call analysis
  identity.py          comparison key, grouping, stable ids
  segment.py           static/dynamic split, dynamic field templates
  metrics.py           cache, waste and structure metrics
  audit.py             event log
  report.py            Markdown + JSON
  pipeline.py          orchestration
  llm/
    client.py          the one place that calls a model
    adjudicate.py      borderline pairs, residual rescue, node labelling
    auditors.py        efficiency and security
    optimize.py        the three rewrite skills
    validate.py        the verification gate
tests/                 46 tests
```
