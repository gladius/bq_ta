# What the hard corpus found

The earlier fixture was generated from clean templates and then used to show that template
matching recovers templates. That is close to circular, and it is why the numbers from it never
transferred to any argument about production.

`samples/hard_corpus.py` replaces it. **1,300 rows, 8.8 MB, 21 callsites, 4 seconds
deterministic.** Small on purpose — the point is difficulty, not volume. Every shape in it is
one that is known or suspected to break the method.

```
python samples/hard_corpus.py --out samples/hard.csv
python stress.py --csv samples/hard.csv          # deterministic, offline, ~4s
python run.py --csv samples/hard.csv             # full pipeline, ~90s, ~$0.06
```

---

## Score

| | before | after |
|---|---|---|
| purity | 0.943 | **0.928** |
| completeness | 0.730 | **0.968** |
| coverage | 0.730 | **0.998** |
| callsites found / true | 18 / 20 | **19 / 20** |
| boundary (intruder) test | — | **100% over 19 nodes** |
| node profiles confirmed | 13 / 18 | **16 / 20** |

Purity dips slightly because coverage rose: calls that used to be silently dropped now have to
be placed, and two of them land in a merged node. That is the correct trade.

---

## Seven bugs it found

Each was invisible to the unit tests, because a unit test isolates one mechanism and these all
live in the interaction between mechanisms.

### 1 · `weak_prompt` measured the wrong quantity
`prompt_line_count` counted lines **before** noise filtering. A prompt of one static line plus
forty per-call lines was never flagged thin, even though filtering cut its comparison key to
that single line — where Jaccard returns only 1.0 or 0.0 and everything sharing the line merges.
Weakness is a property of the key actually compared, so `effective_weak` now measures it there.

### 2 · The user turn was thrown away
It was consulted only as a *fallback* when no system message existed. Every agent that has both
lost the signal entirely — and in RAG-shaped traffic the user turn is where the instruction
lives. It is now a fifth view with its own document frequency.

### 3 · Disjoint views could not outvote a matching prompt
A router and an executor sharing a base prompt and sharing **no** tools scored
`0.60 + 0.00 + 0.10 + 0.05 = 0.75`, past tau, and merged. Putting tools in the node id does not
help: the merge happens first, so only one id is ever minted. Weighted averaging cannot express
"this disagreement is decisive", so a shared-nothing discriminating view now caps the pair at
0.50 — inside the adjudication band, where the LLM can still overrule it.

### 4 · A replayed history made payload look like template
A tool loop resends its opening user turn on every step, so `"Investigate break 7 on the EUR
book"` appears 3–9 times — past a document-frequency floor of 2 — and was kept as template.
Every run then had a line no other run shared, fix #3 fired between runs, and **one callsite of
144 calls shattered into 24 clusters, 60 of which fell below the size floor and were dropped**.
The user surface now uses a proportional floor (5% of the app's calls).

*This one is the reason coverage was 0.730.*

### 5 · Two nodes could carry the same id
Two clusters deriving identical identity material minted identical ids. Downstream that is
corrupting, not untidy: the registry keys history on the id, and a `request_id -> node_id` map
silently loses one of them. Ids are now unique, and the collision is reported on the node.

### 6 · The loader aborted on a missing optional column
`mapping.yaml` names two paths for the agent id. If the column backing one of them is absent —
and exports differ between environments — DuckDB raised a binder error before a single row was
read. Missing optional columns now degrade to "try the next path" and are reported.

### 7 · The adjudicator was shown only the system prompt
The deterministic layer computes four views, then handed the LLM one of them and asked it to
overrule the decision. On this corpus it merged the executor and the router, reporting *"the
prompts are identical in role and rules"*. They are — the tools are the entire difference, and
it had not been shown them. **That single change moved the run from 18 nodes to 20 and the gate
from NOT PASSED to passed.**

---

## What still fails, and whether it can be fixed

### Two callsites that share their whole identifiable surface — *not fixable from logs*

`app_lossy` gives a research agent and a feed monitor the same system prompt, no tools, and user
turns that are unique per run (so they fall under the noise floor). After filtering, the two are
**identical on every view we have**. Purity 0.833.

This is an information limit, not a defect. Nothing in the request distinguishes them.

### `policy_check` merged with `policy_check_bad` — *attribution blurred, finding preserved*

Both carry the same 30k policy manual. One puts a dynamic header in front of it; the other does
not. The header is per-call, so noise filtering removes it, and the two become identical by
content. They merge.

The important part: **the money finding survives the merge.**

```
app_platform:fe8005f38220   calls=100  avg=7,537 tok
    cacheable_now  2
    recoverable    7,514
    "a dynamic region sits ahead of the static content; moving it after the stable
     text would make ~7514 more tokens cacheable (751,400 tokens across this export)"
```

Under the old `recoverable` formula this read **zero** — it subtracted a prose-derived number
from a whole-request-derived one, so stable tool schemas inflated the prefix and clamped the
result. 49 of 56 callsites declared tools and exactly one ever reported a non-zero figure. Both
sides are now whole-request quantities.

So the callsite boundary is wrong here, but the report still says "751,400 tokens are being
re-sent uncached because something dynamic sits at the front".

### A prompt mid-rollout — *no correct answer*

`app_rollout` sends v1 to 90% of calls and v2 to 10%. Whether that is one callsite or two has no
right answer within a single export: it is one place in the code, and two templates. It is
excluded from the score and reported separately. **This is what the registry is for** — across
runs, the fuzzy re-identification recognises v2 as v1 edited and carries the history forward
with a version bump.

### Four nodes flagged `profile_wrong`

Three are the judge conflating a label line with the section beneath it — `"Invoice text:"` is
static even though the invoice below it is not, and the instruction to judge the line rather
than the section did not fully take. The fourth is real and unreported: `fragment_order`
assembles its system prompt from four blocks **in a varying order**, so the template implies a
sequence that does not hold. The metrics do catch the consequence (`cacheable_now` 3 tokens
against a 41-token static body), but the profile does not say "the order varies".

---

---

# The extreme corpus

`samples/extreme_corpus.py` - **7,575 rows, 11.5 MB, 37 callsites, 3 seconds.** Separate file and
separate CSV so the everyday loop stays four seconds. It exists because every scale guard in the
code had been written and never exercised.

| shape | what it leans on |
|---|---|
| `contract_review` 200k-token user turn | `MAX_IDENTITY_CHARS` (200,000) |
| `repo_qa` 120k static system prompt | the cache case at scale |
| `repo_qa_uncached` the same with a per-call header | cache destroyed at 120k tokens |
| `hot_path` 2,400 calls beside 30 callsites of 4-25 | the adaptive size floor |
| `release_notes` one prompt edited five times | drift within one export |
| `adhoc_analysis` 4,500 near-unique prompts | `MAX_LINKAGE_POINTS` (4,000) |
| `invoice_extract` retry storms | retry detection |

## Two more bugs, one found by reading

### 8 · The linkage guard ate calls

`points = points[:MAX_LINKAGE_POINTS]` dropped keys outright. The calls behind them entered no
cluster - **not a node, not residual, simply absent** from the profile with nothing said. An
agent with more than 4,000 distinct prompts, routine for high-variance RAG traffic, silently
lost its tail.

Keys are now ordered by how many calls sit behind them, so the guard keeps the most significant,
and everything it cannot compare is carried out explicitly as `unlinked` and marked `suspect`.

### 9 · A busy hot path hid thirty real callsites

The floor is `max(3, 2% of the agent's calls)`. `app_swarm` has one callsite of 2,400 calls beside
thirty genuine ones of 4-25, so the floor was **56 and all thirty disappeared** - the same failure
a fixed floor of 20 caused in the spike, arriving from the opposite direction.

A proportional floor assumes callsites are of comparable size. A real estate is skewed, and a busy
hot path is not evidence that a quiet callsite is noise. Two changes:

- the fraction is capped at 25
- a below-floor cluster is still reported when it is **coherent** - three or more shared template
  lines and not a different prompt every call - marked `small` rather than hidden

The floor rejects noise; it must not reject quiet callsites, and the two are distinguishable
without counting calls.

```
app_swarm    before:  31 true,  1 found,  completeness 0.864
             after:   31 true, 31 found,  completeness 1.000
```

## Extreme corpus score

| agent | calls | true | found | purity | complete | cover |
|---|---|---|---|---|---|---|
| app_giant | 36 | 3 | 2 | 0.667 | 1.000 | 1.000 |
| app_highvariance | 4,500 | 1 | 1 | 1.000 | 1.000 | 1.000 |
| app_retrystorm | 60 | 1 | 1 | 1.000 | 1.000 | 1.000 |
| app_swarm | 2,779 | 31 | 31 | 1.000 | 1.000 | 1.000 |
| app_drift | 200 | ambiguous | 1 | - | - | - |
| **overall** | **7,375** | **36** | **35** | **0.998** | **1.000** | **1.000** |

The one failure is `repo_qa` merging with `repo_qa_uncached` - the same class as `policy_check`:
two callsites identical in content, differing only by a per-call header that noise filtering
removes. Attribution blurs; the cache finding survives.

`app_drift` merging five prompt versions into one callsite is the **right** answer within a single
export - it is one place in the code. Across runs the registry records it as versions.

---

## What this does not establish

The corpus is still synthetic. It is adversarial rather than circular, which is a real
improvement, but it was written by the same person who wrote the code and therefore contains
the failures that person thought of.

**Nothing here is evidence about a real export.** The next measurement that matters is one real
CSV through `stress.py` and `diagnose.py`.
