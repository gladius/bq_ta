"""Report rendering: JSON for pipelines, Markdown for humans.

This reports a **profile**, not a plan. The optimisation layer was removed, so the Markdown no
longer opens with "what to change" - it opens with what each agent is actually made of, and
how much of that we can defend:

  1. what we found, and whether the verifier believed it
  2. one row per agent: callsites, runs, tokens, cache
  3. per agent: each callsite - purpose, shape, size, cache position, our confidence
  4. what we could not do, stated plainly rather than omitted

The last section matters most. A profile that hides its own unreliable parts is worse than no
profile, because someone will act on it.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

from profiler.pipeline import AppResult, NodeResult, RunResult

VERDICT_MARK = {"confirmed": "confirmed", "profile_wrong": "PROFILE WRONG",
                "impure": "IMPURE", "boundary_weak": "boundary weak",
                "unverified": "unverified"}


def write_json(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.as_dict(), handle, indent=1, default=str)


def _table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> List[str]:
    if not rows:
        return ["_none_", ""]
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    out.append("")
    return out


def _name(node: NodeResult) -> str:
    return (node.label or {}).get("name") or node.node.node_id.split(":")[-1]


def _cache_state(node: NodeResult) -> str:
    """One phrase for where this callsite stands on caching."""
    m = node.metrics
    if m.reported_cached_mean is not None and m.avg_request_tokens:
        share = m.reported_cached_mean / max(m.avg_request_tokens, 1)
        if share >= 0.5:
            return f"cached {share:.0%}"
        if m.cacheable_now >= m.cache_minimum:
            return f"cached {share:.0%} — below what the prefix allows"
    if m.cacheable_now >= m.cache_minimum:
        return f"{m.cacheable_now:,} tok prefix, cacheable"
    if m.cacheable_now + m.recoverable >= m.cache_minimum:
        return f"{m.cacheable_now:,} tok — reorder would reach the minimum"
    return f"{m.cacheable_now:,} tok — below the {m.cache_minimum} minimum"


def _summary(result: RunResult) -> List[str]:
    nodes = [n for a in result.apps for n in a.nodes]
    judged = result.judge_summary or {}
    calls = sum(a.calls for a in result.apps)

    lines = ["## What we found", ""]
    lines.append(f"**{len(result.apps)} agents · {len(nodes)} callsites · {calls:,} calls.** "
                 f"A callsite is one place in a program that calls a model; the count is "
                 f"discovered from the requests, not declared anywhere.")
    lines.append("")

    rate = judged.get("confirmed_rate")
    if rate is not None:
        confirmed = (judged.get("by_verdict") or {}).get("confirmed", 0)
        lines.append(f"**Verification: {confirmed}/{judged.get('nodes_judged', 0)} callsite "
                     f"profiles were confirmed** against their own sampled requests "
                     f"({rate:.0%}). ")
        correct = judged.get("boundary_correct")
        if correct is not None:
            lines.append(f"On the boundary test — a planted request from the nearest competing "
                         f"callsite — the judge was right {correct:.0%} of the time over "
                         f"{judged.get('boundary_tests', 0)} nodes.")
    else:
        lines.append("**Not verified**: no LLM key was available, so nothing checked whether "
                     "these profiles are true of their own requests.")
    lines += ["", f"> Gate: **{'passed' if result.gate_passed else 'NOT PASSED'}** — "
                  f"{result.gate_reason}", ""]

    weak = [n for n in nodes if n.node.quality.verdict != "ok"]
    if weak:
        lines.append(f"{len(weak)} of {len(nodes)} callsites carry a quality warning "
                     f"(computed without any answer key). They are listed per agent below and "
                     f"should be read as provisional.")
        lines.append("")
    return lines


def _agents_table(result: RunResult) -> List[str]:
    rows = []
    for app in sorted(result.apps, key=lambda a: -a.calls):
        single = app.single_call or {}
        cached = single.get("total_cached_tokens")
        prompt = single.get("total_prompt_tokens") or 0
        share = f"{cached / prompt:.0%}" if cached and prompt else "—"
        runs = app.chain.runs if app.chain else "—"
        if app.chain and app.chain.verdict == "unusable":
            runs = "n/a"
        rows.append([app.app_id, f"{app.calls:,}", len(app.nodes), runs,
                     f"{prompt:,}" if prompt else "—", share])
    return (["## Agents", "",
             "`cache` is the provider's own reported figure, not our estimate.", ""]
            + _table(["agent", "calls", "callsites", "runs", "prompt tokens", "cache"], rows))


def _shape_note(node: NodeResult) -> str:
    """Message shapes seen at this callsite. Several is normal, not a defect."""
    shapes = node.node.signature.get("shapes") or []
    if not shapes:
        return "—"
    if len(shapes) == 1:
        return f"`{shapes[0]['shape']}`"
    top = ", ".join(f"`{s['shape']}`×{s['calls']}" for s in shapes[:3])
    return f"{top}{' …' if len(shapes) > 3 else ''}"


def _app_section(app: AppResult) -> List[str]:
    lines = [f"### `{app.app_id}`", "",
             f"{app.calls:,} calls · {len(app.nodes)} callsites"
             + (f" · {app.residual_calls} calls below the size floor "
                f"({app.min_size}) and not profiled" if app.residual_calls else ""), ""]

    if app.chain:
        c = app.chain
        if c.verdict == "unusable":
            lines += [f"**Runs: not recoverable.** {c.note}", ""]
        elif c.verdict == "single_step":
            lines += [f"**Runs:** {c.note}.", ""]
        else:
            lines += [f"**Runs:** {c.runs:,} reconstructed, {c.mean_steps} steps on average "
                      f"(longest {c.max_steps}), ~{c.mean_run_tokens:,} tokens per run"
                      + (f". {c.note}" if c.note else "."), ""]
            if c.top_paths:
                lines += ["Most common call sequences:", ""]
                lines += _table(["path", "runs"],
                                [[p["path"], p["runs"]] for p in c.top_paths])

    rows = []
    for node in sorted(app.nodes, key=lambda n: -n.metrics.calls):
        verdict = VERDICT_MARK.get(node.verdict.verdict if node.verdict else "unverified", "—")
        quality = node.node.quality
        rows.append([
            _name(node), node.metrics.calls, f"{node.metrics.avg_request_tokens:,}",
            _shape_note(node), _cache_state(node),
            f"{quality.verdict} (margin {quality.margin:.2f})", verdict,
        ])
    lines += _table(["callsite", "calls", "avg tokens", "message shape", "cache",
                     "grouping quality", "judge"], rows)

    for node in sorted(app.nodes, key=lambda n: -n.metrics.calls):
        lines += _node_detail(node)
    return lines


def _node_detail(node: NodeResult) -> List[str]:
    m, label = node.metrics, node.label or {}
    lines = [f"<details><summary><b>{_name(node)}</b> — {m.calls} calls, "
             f"{m.avg_request_tokens:,} tokens</summary>", ""]

    if label.get("purpose"):
        lines += [f"**Purpose.** {label['purpose']}", ""]

    lines += [f"**Identity.** `{node.node.node_id}` · {len(node.node.template)} template lines "
              f"· {node.node.variable_lines} lines vary · {node.node.distinct_prompts} distinct "
              f"prompts across {m.calls} calls · confidence _{node.node.confidence}_", ""]

    if node.history:
        h = node.history
        if h["status"] == "seen":
            lines += [f"**Previously profiled.** First seen {h['first_seen']}, "
                      f"{h['runs_seen']} runs. The prompt has not changed since.", ""]
        elif h["status"] == "changed":
            lines += [f"**Changed since last run.** Matched the earlier callsite at "
                      f"{h['similarity']:.2f} — version {h['version']}, "
                      f"+{h['lines_added']}/-{h['lines_removed']} template lines. "
                      f"First seen {h['first_seen']}.", ""]
        else:
            lines += ["**New.** Not seen in any previous run.", ""]

    if node.node.quality.reasons:
        lines += ["**Grouping caveats**", ""] + \
                 [f"- {r}" for r in node.node.quality.reasons] + [""]

    if m.findings:
        lines += ["**Measured**", ""] + [f"- {f}" for f in m.findings] + [""]

    if m.tools_declared:
        lines += [f"**Tools.** {m.tools_declared} declared "
                  f"(~{m.tool_schema_tokens:,} tokens), {len(m.tools_called)} ever called"
                  + (f"; never called: {', '.join(m.tools_never_called[:8])}"
                     if m.tools_never_called else ""), ""]

    fields = node.segmentation.fields[:8]
    if fields:
        lines += ["**What varies per call**", ""]
        lines += _table(["template", "occurrences", "example"],
                        [[f"`{f.template[:90]}`", f.occurrences,
                          (f.examples[0][:60] + "…") if f.examples else "—"]
                         for f in fields])

    if node.verdict and node.verdict.verdict not in ("confirmed", "unverified"):
        v = node.verdict
        lines += [f"**The judge disagreed** — _{v.verdict}_", ""]
        for note in v.notes:
            lines += [f"- {note}"]
        for err in v.template_errors:
            lines += [f"- claimed static but varies: `{err}`"]
        for miss in v.missed_dynamic:
            lines += [f"- varies but was not listed: `{miss}`"]
        lines += [""]

    if node.node.template:
        lines += ["<details><summary>template</summary>", "", "````"]
        lines += node.node.template[:120]
        if len(node.node.template) > 120:
            lines.append(f"...[{len(node.node.template)} lines total]")
        lines += ["````", "", "</details>", ""]

    lines += ["</details>", ""]
    return lines


def _limits(result: RunResult) -> List[str]:
    """What this report cannot tell you. Kept in, never trimmed."""
    lines = ["## What this does not tell you", ""]

    unusable = [a.app_id for a in result.apps
                if a.chain and a.chain.verdict == "unusable"]
    if unusable:
        lines.append(f"- **Runs could not be reconstructed for {len(unusable)} agents** "
                     f"({', '.join(unusable[:6])}). Those agents compact or window their "
                     f"history, which destroys the shared message prefix before the gateway "
                     f"sees the request. This is not recoverable from logs; it needs a run-id "
                     f"header from the client.")
    residual = sum(a.residual_calls for a in result.apps)
    if residual:
        lines.append(f"- **{residual:,} calls fell below the per-agent size floor** and were "
                     f"not profiled. They are counted in the totals but have no callsite.")
    if result.judge_summary.get("problems"):
        lines.append(f"- **{len(result.judge_summary['problems'])} callsite profiles were "
                     f"flagged by the judge** and are reported as-is. Nothing was silently "
                     f"corrected.")
    lines.append("- **Cost figures are token counts**, not billed amounts; no price list is "
                 "applied.")
    lines.append("- **No optimisation advice is produced.** That layer was removed until the "
                 "profile half is trustworthy on real data.")
    lines.append("")
    return lines


def markdown(result: RunResult) -> str:
    lines = [f"# Agent profile — {result.run_id}", ""]
    inp = result.load_report
    lines.append(f"_{inp['loaded']:,} of {inp['total_rows']:,} rows loaded · "
                 f"{result.llm_usage['calls']} LLM calls "
                 f"(${result.llm_usage['cost_usd']}) · {result.elapsed_s}s_")
    lines.append("")

    lines += _summary(result)
    lines += _agents_table(result)
    lines += ["## Callsites", ""]
    for app in sorted(result.apps, key=lambda a: -a.calls):
        lines += _app_section(app)
    lines += _limits(result)
    return "\n".join(lines)


def write_markdown(result: RunResult, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(markdown(result))
