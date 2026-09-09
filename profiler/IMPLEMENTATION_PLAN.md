# Agent Profiler — implementation plan

> **Status: built and verified end to end.** See `README.md` to run it. Measured on the
> development fixture (3,229 rows, 16 apps, ground truth known):
>
> | | |
> |---|---|
> | callsites discovered | **56**, against 56 true callsites |
> | calls assigned | **3,210 / 3,210 (100%)** |
> | verification gate | **PASSED — 20/20 sampled findings confirmed** |
> | optimisation found | **38,912 tokens** recoverable across the export |
> | LLM cost | **$0.90** first run, $0.00 cached; 289 calls |
> | runtime | 47 s with concurrency; 4 s with `--no-llm` |
> | tests | 46 passing |
>
> The verification stage earned its place immediately: it rejected three `output_optimization`
> proposals that would have changed a model's output format, which led to the safety-class
> split now described in §7.

Turns a BigQuery/litellm CSV export into per-callsite profiles, security and efficiency findings,
and concrete optimisation advice — with no instrumentation of the calling applications.

Design choices are carried over from the spike and cite the measurement that settled them
(`../spike/DECISIONS.md`, `../spike/ASSESSMENT.md`). Where something is *not* yet measured, it
says so.

---

## 1. Two engines, one rule

**Deterministic** does the bulk: every row, every comparison, reproducible, free.
**LLM** does the low-volume judgment: per node, per ambiguous pair, per sampled finding.

> **The rule that keeps this honest: LLM output must never silently influence a deterministic
> measurement.** The LLM annotates, adjudicates explicitly-marked decisions, and verifies
> afterwards. If a hypothesis from the LLM fed back into clustering, we could no longer tell
> whether the method worked — which is the whole point of having measured it.

Every LLM call is recorded in the audit trail with its input, output, and the decision it changed.

---

## 2. The flow

```
CSV export (litellm)
   |
[0] RECON        sample 3-5 calls per app -> LLM orientation          [LLM, ~5/app]
   |                (a hypothesis, never an input to grouping)
[1] LOAD         parse request_payload / response_payload JSON
   |
[2] SINGLE-CALL  per-row facts needing no group        --> emit immediately
   |             + LLM audit on a sample                             [LLM, ~5/app]
   |  partition by app_id
[3] NOISE KEY    build the comparison key: ignore lines seen once in this app
   |             (the stored data is NOT modified — see §4)
[4] GROUP        single-linkage Jaccard, tau <= 0.6, NO structural bucket key
   |             + LLM adjudication of borderline pairs              [LLM, ~10-30/app]
   |             + LLM rescue of below-floor clusters                [LLM, ~5-20/app]
[5] STABLE ID    hash each node's template -> id that survives across exports
   |
[6] SEGMENT      df for dynamic regions + Drain for the values inside them
   |
[7] PROFILE      cacheable prefix, recoverable tokens, waste, structure
   |             + 3 LLM auditors: efficiency / security / correctness  [LLM, ~60/app]
   |
[8] CHAIN        message-prefix walk into runs, with a confidence flag
   |
[9] VALIDATE     sample findings, ask an LLM to confirm them against raw rows  [LLM, ~30/app]
   |
[10] REPORT      JSON + Markdown + AUDIT.md
```

**Why this order** — each justified by a measurement:

| step | why here |
|---|---|
| [2] before [3] | Single-call facts need no group, so they can be emitted on row 1 and survive a tiny export. |
| [3] before [4] | **Measured.** Ignoring per-call noise *first* is what lets a RAG-style callsite group at all — 200 rows unassigned before, 200 in one node after (D28). After grouping is too late: a callsite that failed to group never gets a template. |
| [4] with no bucket key | **Measured.** A bucket key is a *hard* partition clustering cannot undo. A section signature split one callsite into 27 buckets (D31); a tool-set key lost an entire callsite to per-call tool injection (D32). Removing bucketing beat seven alternatives: purity 1.000, completeness 1.000, 18 nodes for 18 callsites (D32). |
| [5] after [4] | Ids must derive from content, not position, or nothing is comparable between exports. **Measured: 18/18 nodes re-identified across two independent windows.** |
| [6] after [4] | Segmentation is a within-node frequency question; it needs the members. |
| [7] after [6] | The advice depends on knowing which regions are dynamic. |
| [8] independent of [4] | Chaining needs no nodes; placed after so its confidence flag can mark which run metrics to trust. |
| [9] last | Verification must run on finished findings, not on intermediate state. |

**Rejected, with evidence**, so nobody re-proposes them: structural bucketing (D32), TF-IDF
weighting (D28), containment (D24), MinHash/LSH (unnecessary below ~20k calls per app, D24),
embeddings for identity (would blur prompt versions differing by three lines).

---

## 3. Small exports — no assumed window, no re-fetching

The export may hold far less than 24h and we cannot go back for more. That breaks one inherited
parameter and forces three changes:

1. **The minimum cluster size must be relative, not absolute.** A fixed floor of 20 made *38 of
   56 callsites invisible* in the spike corpus. Use `max(3, ceil(0.02 * app_rows))`.
2. **LLM rescue below the floor** — "is this 7-call cluster a real callsite, or noise?" This is
   the long tail that a fixed floor silently deletes.
3. **Per-node confidence from N.** A node built from 8 calls is reported *provisional*; one built
   from 400 is reported *established*. Measured basis: the static core is recovered at
   precision/recall 1.000 from N = 10 in our corpus, but that corpus is bimodal and real prompts
   will be harder, so N < 30 is flagged.

**The pipeline must never silently under-report.** If N is too small for a conclusion, it says
"insufficient data for X" rather than omitting X.

---

## 4. "Ignore lines seen once" does not alter data

Worth stating plainly because the wording invites the opposite reading.

The filter applies **only to the temporary key used to compare two calls**. Nothing is modified,
dropped, or rewritten in storage. The full original text is used for the template, the cache
maths, the profile, and everything in the report.

It is the same idea as ignoring stopwords when comparing two documents: the documents are
untouched, only the comparison changes.

Why it is needed: without it a RAG callsite has pairwise similarity 0.107 and never groups
(200 rows unassigned). With it, the same 200 rows form one correct node.

---

## 5. Input contract

Entry point is a CSV export. Column names are declared, never assumed:

```yaml
# mapping.yaml
columns:
  timestamp: "startTime"
  request:   "request_payload"
  response:  "response_payload"
  model:     "model"

# the agent identity appears in two places; first non-null wins, and disagreements are counted
app_id_paths:
  - "request_payload.labels.vz_key_name"
  - "client_metadata.headers.x-subject"

options:
  json_is_string: true
  max_rows_per_app: 1000
```

`app_id` is a **JSON path, not a column** — with a documented fallback order. When both are
present and disagree, the run records how often, because that would undermine the outer partition.

**Real exports are messy**, so: unparseable JSON is counted and skipped, never fatal; a missing
`usage` disables cache metrics for that row only; non-chat rows (embeddings) are filtered out;
streamed responses may lack `usage` entirely.

---

## 6. Audit trail

No new service dependency. A generic tracer records *calls*; we need *decisions*.

- **`audit/run-<id>.jsonl`** — one event per step: parameters, row counts in and out, and for
  every LLM call the model, token counts, cost, latency, prompt hash, and the decision it changed
- **`AUDIT.md`** — rendered from the log: total LLM calls and cost, decisions per step, every
  case where the LLM disagreed with the deterministic result

The run is reproducible from the JSONL: same input plus same recorded decisions replays exactly,
which also allows re-running without paying for the LLM twice.

---

## 7. Where the LLM is used, and why each one earns it

| # | step | job | volume | why it flips something |
|---|---|---|---|---|
| 1 | [4] | adjudicate borderline pairs near tau | 10-30/app | tau is our most fragile parameter — 0.7 silently split a callsite into 2, 0.8 into 5. This replaces a brittle constant with judgment on a handful of pairs |
| 2 | [4] | rescue below-floor clusters | 5-20/app | a size floor hid 38 of 56 callsites; this recovers the long tail |
| 3 | [7] | efficiency / security / correctness auditors | ~60/app | reads the template, dynamic slots, cache numbers and tool list, and finds what no metric encodes |
| 4 | [7] | rewrite, not just diagnose | ~20/app | deterministic advice says "move the header"; the LLM emits the rewritten prompt |
| 5 | [0][2] | orientation from a few sampled calls | ~10/app | makes the deterministic output interpretable before it exists |
| 6 | [9] | **verify sampled findings against raw rows** | ~30/app | catches systematic error — the layer that caught real bugs by hand during the spike |
| 7 | [6] | judge conditional-block vs separate callsite | as needed | deterministic frequency analysis genuinely cannot tell these apart |

### The three auditors (step 7)

Each receives the node's template, its dynamic slots, cache metrics, tool schemas and 3 sample
calls — grounded in the deterministic profile rather than speculating.

- **Efficiency** — bloated tool schemas, redundant or contradictory instructions, missing output
  contract, cache-hostile ordering, model over-specified for the task
- **Security** — secrets or PII in the system prompt, untrusted content concatenated without
  delimiters (prompt-injection surface), over-permissive tool set, data-egress paths through tools
- **Correctness** — unreachable or conflicting rules, tool descriptions that do not match actual
  usage, missing failure handling in the loop

### The validation stage (step 9)

| sampled | question |
|---|---|
| K nodes | "Here is the template and 5 raw requests. Do these belong together? Does the template describe them?" |
| cache advice | "Given this prompt, is this suggestion correct and safe?" |
| static/dynamic split | "Is this region correctly marked dynamic?" |
| reconstructed runs | "Do these N requests look like one agent run?" |

Output: *"X of Y findings confirmed"*, with every disagreement listed in `AUDIT.md`. Verification
never edits the findings — it reports on them, so a systematic failure is visible rather than
silently patched.

---

## 8. Module layout

```
profiler/
  IMPLEMENTATION_PLAN.md
  README.md
  mapping.yaml
  run.py                     entry point: CSV in, report out
  profiler/
    load.py                  [1] CSV -> normalised calls (DuckDB read_csv_auto, chunked)
    single.py                [2] per-call analysis
    identity.py              [3][4][5] comparison key, grouping, stable ids
    segment.py               [6] df regions + Drain values
    profile.py               [7] cache / waste / structure metrics
    chain.py                 [8] prefix walk + confidence
    llm/
      client.py              one place that calls a model; retries, cost accounting
      adjudicate.py          borderline pairs, below-floor clusters
      audit.py               efficiency / security / correctness skills
      validate.py            [9] sampled verification of findings
      recon.py               [0] orientation
    audit.py                 event log + AUDIT.md
    report.py                JSON / Markdown output
  tests/
  samples/
    make_sample_csv.py       export the spike corpus into a litellm-shaped CSV
    sample.csv               generated development fixture
```

DuckDB does the load, JSON extraction, per-app filtering and aggregates — it handles files larger
than memory. Python does the clustering, which is O(n²) over *distinct* prompts and therefore
tens of points after deduplication, not thousands. **Processing is one app at a time**, so memory
is bounded regardless of file size.

Algorithms are **ported from the spike, not rewritten** — they are the validated part.

---

## 9. Build order

| # | step | why in this position |
|---|---|---|
| 1 | `make_sample_csv.py` + `load.py` | nothing is testable without an input |
| 2 | `single.py` + `report.py` | delivers value on row 1, independent of everything else, works on a tiny export |
| 3 | `identity.py` | the validated core; must reproduce the spike's numbers on the sample |
| 4 | `audit.py` | before any LLM code, so every call is recorded from the first one |
| 5 | `profile.py` + `segment.py` | the product output |
| 6 | `llm/` | adjudication, auditors, then validation |
| 7 | `chain.py` | run view, with confidence |

We develop against `samples/sample.csv` — the spike's 13,564 rows re-shaped as a litellm export,
JSON stored as strings, with some deliberately malformed rows. It has **known ground truth**, so
the port can be proved to reproduce purity 1.000 / completeness 1.000 / 18 nodes for 18 callsites
before it ever sees production data. When the real CSV arrives, only `mapping.yaml` should change.

---

## 10. Open questions for the real export

Answers change configuration, not architecture:

1. **Do the two agent-id paths ever disagree?** The run counts it; a high rate would undermine
   the outer partition.
2. **Line document-frequency distribution.** Ours is bimodal — a line is in ~100% of calls or in
   exactly one. Production will have lines behind conditionals at 30-60%, which is where the
   noise filter and `tau` are least tested. **One histogram from one real app is worth more than
   any further synthetic work.**
3. **How many rows, over what span, per app?** Decides whether nodes are established or
   provisional, and how much of the pipeline is meaningful at all.
4. **Are streamed rows missing `usage`?** Decides how much cache analysis is available.
