"""A small static site per run: an index over the estate, then one page per agent.

Generated locally and opened from disk: no network, no CDN, no upload. Gateway logs are the most
sensitive thing an estate has, so nothing here reaches out.

**One page per agent, not one page per run.** A single page was fine for six agents and will not
survive an enterprise estate: a few hundred agents with a dozen callsites each is a document
nobody scrolls, and it makes the browser lay out every callsite to show one. The index carries
the ranking and the warnings; each agent page stands alone and is small enough to hand to the
team that owns it.

    out/run-<id>/index.html      the estate, ranked by tokens
    out/run-<id>/<agent>.html    one agent: its callsites, its runs, its caveats
    out/run-<id>/report.json     everything, for pipelines
    out/run-<id>/report.md       the same in prose

The Markdown reads top to bottom; these are for looking. Three things they show that prose
cannot:

  the token bar    every callsite as a share of the agent's spend, so the one that matters is
                   obvious before reading a word
  the cache bar    each callsite split into cached / stable prefix / recoverable by reorder /
                   genuinely dynamic. The money picture in one row.
  the shape strip  the message shapes a callsite fires at, with counts. Several is normal for an
                   agent loop, and seeing that stops a reader mistaking it for a defect.

Everything is drawn with CSS from numbers the deterministic layer measured. No charting library,
nothing inferred at render time. **No optimisation advice** - a bar showing that 7,514 tokens sit
behind a dynamic region is a measurement; what to do about it is not this tool's business yet.
"""

from __future__ import annotations

import html
import os
from typing import Any, Sequence

from profiler.pipeline import AppResult, NodeResult, RunResult

VERDICT_CLASS = {"confirmed": "ok", "profile_wrong": "bad", "impure": "bad",
                 "boundary_weak": "warn", "unverified": "muted"}
QUALITY_CLASS = {"ok": "ok", "weak": "warn", "suspect": "bad"}

CSS = """
:root {
  --bg:#f7f7f5; --panel:#fff; --ink:#1a1a18; --dim:#6b6b66; --line:#e2e2dd;
  --ok:#2f7d4f; --warn:#b07d20; --bad:#b0402f; --accent:#3a5fa8;
  --cached:#2f7d4f; --now:#5aa06f; --recover:#d9a441; --dynamic:#c9c9c2;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg:#16161a; --panel:#1e1e23; --ink:#e8e8e4; --dim:#9a9a94; --line:#32323a;
    --ok:#5fbe86; --warn:#d9a441; --bad:#e0715c; --accent:#8aa9e8;
    --cached:#5fbe86; --now:#3d8a5c; --recover:#d9a441; --dynamic:#4a4a52;
  }
}
:root[data-theme="dark"] {
  --bg:#16161a; --panel:#1e1e23; --ink:#e8e8e4; --dim:#9a9a94; --line:#32323a;
  --ok:#5fbe86; --warn:#d9a441; --bad:#e0715c; --accent:#8aa9e8;
  --cached:#5fbe86; --now:#3d8a5c; --recover:#d9a441; --dynamic:#4a4a52;
}
* { box-sizing:border-box; }
body { background:var(--bg); color:var(--ink); font:14px/1.55 -apple-system,BlinkMacSystemFont,
       "Segoe UI",Roboto,Helvetica,Arial,sans-serif; margin:0; padding:28px 20px 80px; }
.wrap { max-width:1180px; margin:0 auto; }
a { color:var(--accent); }
h1 { font-size:20px; margin:0 0 4px; letter-spacing:-.2px; }
h2 { font-size:15px; margin:34px 0 12px; text-transform:uppercase; letter-spacing:.09em;
     color:var(--dim); font-weight:600; }
h3 { font-size:15px; margin:0; }
.sub { color:var(--dim); font-size:12.5px; margin-bottom:20px; }
.mono { font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }

.banner { padding:12px 16px; border-radius:8px; border:1px solid var(--line);
          background:var(--panel); margin-bottom:22px; font-size:13.5px; }
.banner.ok   { border-left:4px solid var(--ok); }
.banner.warn { border-left:4px solid var(--warn); }
.banner.bad  { border-left:4px solid var(--bad); }

.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:8px;
        padding:12px 14px; }
.card .n { font-size:22px; font-weight:600; letter-spacing:-.5px; }
.card .l { color:var(--dim); font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; }

.agent { background:var(--panel); border:1px solid var(--line); border-radius:10px;
         margin-bottom:14px; overflow:hidden; }
.agent > summary { cursor:pointer; padding:14px 18px; display:flex; align-items:center;
                   gap:14px; flex-wrap:wrap; list-style:none; }
.agent > summary::-webkit-details-marker { display:none; }
.agent > summary::before { content:"\\25B8"; color:var(--dim); font-size:11px; }
.agent[open] > summary::before { content:"\\25BE"; }
.agent .body { padding:0 18px 18px; }
.chip { font-size:11.5px; color:var(--dim); white-space:nowrap; }
.chip b { color:var(--ink); font-weight:600; }
.chip.bad b, .chip.bad { color:var(--bad); }

.bar { display:flex; height:9px; border-radius:5px; overflow:hidden; background:var(--dynamic);
       margin:5px 0 3px; }
.bar i { display:block; height:100%; }
.legend { font-size:11px; color:var(--dim); display:flex; gap:12px; flex-wrap:wrap; }
.legend b { font-weight:600; }
.sw { display:inline-block; width:9px; height:9px; border-radius:2px; vertical-align:-1px;
      margin-right:4px; }

table { width:100%; border-collapse:collapse; font-size:13px; }
th { text-align:left; font-weight:600; color:var(--dim); font-size:11px; text-transform:uppercase;
     letter-spacing:.06em; padding:8px 10px; border-bottom:1px solid var(--line); }
td { padding:9px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
tr:last-child td { border-bottom:none; }
.scroll { overflow-x:auto; }

.tag { display:inline-block; font-size:11px; padding:1px 7px; border-radius:10px;
       border:1px solid currentColor; font-weight:600; margin-right:4px; }
.ok{color:var(--ok);} .warn{color:var(--warn);} .bad{color:var(--bad);} .muted{color:var(--dim);}
.shape { font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px;
         background:rgba(128,128,128,.14); padding:1px 5px; border-radius:3px;
         margin-right:4px; white-space:nowrap; }

details.node { border-top:1px solid var(--line); }
details.node > summary { cursor:pointer; padding:10px 2px; font-size:12.5px; color:var(--accent); }
pre { background:rgba(128,128,128,.1); padding:12px; border-radius:6px; overflow-x:auto;
      font-size:12px; line-height:1.5; margin:8px 0; max-height:420px; }
ul { margin:6px 0; padding-left:20px; }
li { margin:3px 0; }
.note { font-size:12.5px; color:var(--dim); margin:6px 0; }
.limits li { margin:7px 0; }
.msg { margin:8px 0; font-family:ui-monospace,Menlo,Consolas,monospace; font-size:11.5px;
       line-height:1.45; word-break:break-word; }
.msg > div { padding-left:16px; text-indent:-16px; }
.sample { border-left:2px solid var(--line); padding-left:12px; margin:12px 0; }
#filter { width:100%; max-width:440px; padding:7px 10px; border:1px solid var(--line);
          border-radius:6px; background:var(--panel); color:var(--ink); font-size:13px;
          margin-bottom:12px; }
.hidden { display:none !important; }
.pager { display:flex; align-items:center; gap:10px; margin:10px 0 4px; flex-wrap:wrap; }
.pager button { font:inherit; font-size:12px; padding:3px 10px; border:1px solid var(--line);
                border-radius:6px; background:var(--panel); color:var(--ink); cursor:pointer; }
.pager button:hover:not(:disabled) { border-color:var(--accent); color:var(--accent); }
.pager button:disabled { opacity:.4; cursor:default; }
.pager .count { font-size:12px; color:var(--dim); min-width:56px; text-align:center; }
.req { border-left:2px solid var(--line); padding-left:12px; margin:10px 0; }
"""


PAGER_SCRIPT = """<script>
document.querySelectorAll('.pager').forEach(function(bar){
  var reqs = document.querySelector('.reqs[data-for="' + bar.id + '"]');
  if (!reqs) return;
  var items = reqs.querySelectorAll('.req'), at = 0;
  var count = bar.querySelector('.count b');
  var buttons = bar.querySelectorAll('button');
  function show(next){
    at = Math.max(0, Math.min(items.length - 1, next));
    items.forEach(function(el, i){ el.classList.toggle('hidden', i !== at); });
    count.textContent = at + 1;
    buttons[0].disabled = at === 0;
    buttons[1].disabled = at === items.length - 1;
  }
  buttons.forEach(function(b){
    b.addEventListener('click', function(){ show(at + parseInt(b.dataset.step, 10)); });
  });
  show(0);
});
</script>"""

def e(text: Any) -> str:
    return html.escape(str(text if text is not None else ""))


def _name(node: NodeResult) -> str:
    return (node.label or {}).get("name") or node.node.node_id.split(":")[-1]


def _slug(app_id: str) -> str:
    """A filename for an agent id, which in a real export may contain anything at all."""
    safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(app_id)).strip("-")
    return (safe or "agent")[:80]


def _tokens_of(app: AppResult) -> int:
    return sum(n.metrics.calls * n.metrics.avg_request_tokens for n in app.nodes)


def _bar(segments: Sequence[tuple]) -> str:
    """segments: (share 0..1, css colour var, label). Zero-width pieces are dropped."""
    parts = "".join(f'<i style="width:{share * 100:.4f}%;background:var(--{colour})"></i>'
                    for share, colour, _ in segments if share > 0.0005)
    return f'<div class="bar">{parts}</div>'


def _cache_bar(node: NodeResult) -> str:
    """Where this callsite's tokens sit: cached, stable prefix, recoverable, dynamic."""
    m = node.metrics
    total = max(m.avg_request_tokens, 1)
    cached = min(m.reported_cached_mean or 0, total)
    now = max(0.0, min(m.cacheable_now, total) - cached)
    recover = max(0.0, min(m.recoverable, total - cached - now))
    dynamic = max(0.0, total - cached - now - recover)

    bar = _bar([(cached / total, "cached", ""), (now / total, "now", ""),
                (recover / total, "recover", ""), (dynamic / total, "dynamic", "")])
    return bar + (
        '<div class="legend">'
        f'<span><span class="sw" style="background:var(--cached)"></span>'
        f'cached <b>{cached:,.0f}</b></span>'
        f'<span><span class="sw" style="background:var(--now)"></span>'
        f'stable prefix <b>{now:,.0f}</b></span>'
        f'<span><span class="sw" style="background:var(--recover)"></span>'
        f'recoverable by reorder <b>{recover:,.0f}</b></span>'
        f'<span><span class="sw" style="background:var(--dynamic)"></span>'
        f'varies <b>{dynamic:,.0f}</b></span>'
        '</div>')


def _shapes(node: NodeResult) -> str:
    """Message role patterns. A clustering feature, kept in the detail pane only.

    It used to be a headline column, where "SU x60" told the reader nothing: one system message
    and one user message describes most of an estate. Several shapes at one callsite is normal
    for an agent loop, which is the only thing worth saying about it.
    """
    shapes = node.node.signature.get("shapes") or []
    if not shapes:
        return "&mdash;"
    out = "".join(f'<span class="shape">{e(s["shape"])}&times;{s["calls"]}</span>'
                  for s in shapes[:6])
    note = ("" if len(shapes) == 1 else
            '<span class="note">several shapes at one callsite is normal for an agent loop &mdash; '
            "it is the same code at different depths</span>")
    return out + note


REGION_COLOUR = {"instructions": "now", "tool schemas": "accent",
                 "the user turn": "recover", "conversation history": "dynamic"}


def _anatomy_cell(node: NodeResult) -> str:
    """What the request is made of, and which parts hold still.

    This replaces the role-shape column. A reader gains nothing from "SU" and a great deal from
    "850 tokens of instructions that never move, 12,000 tokens of user payload that always do".
    """
    a = node.metrics.anatomy
    if not a or not a.regions:
        return '<span class="note">&mdash;</span>'

    total = max(sum(r.mean_tokens for r in a.regions), 1)
    bar = _bar([(r.mean_tokens / total, REGION_COLOUR.get(r.name, "dynamic"), r.name)
                for r in a.regions])
    rows = "".join(
        '<span><span class="sw" style="background:var(--%s)"></span>%s <b>%s</b> %s</span>'
        % (REGION_COLOUR.get(r.name, "dynamic"), e(r.name), f"{r.mean_tokens:,}",
           ('<span class="bad">varies</span>' if r.varies else "fixed"))
        for r in a.regions)
    return bar + f'<div class="legend">{rows}</div>'


def _node_row(node: NodeResult, app_tokens: int) -> str:
    m = node.metrics
    share = (m.calls * m.avg_request_tokens) / max(app_tokens, 1)
    verdict = node.verdict.verdict if node.verdict else "unverified"
    q = node.node.quality
    purpose = (node.label or {}).get("purpose") or ""
    return (
        "<tr>"
        f'<td><b>{e(_name(node))}</b><div class="note">{e(purpose[:150])}</div>'
        f'<div class="mono chip">{e(node.node.node_id)}</div></td>'
        f'<td class="mono">{m.calls:,}<div class="note">{share:.0%} of tokens</div></td>'
        f'<td class="mono">{m.avg_request_tokens:,}</td>'
        f'<td style="min-width:230px">{_cache_bar(node)}</td>'
        f"<td style=\"min-width:210px\">{_anatomy_cell(node)}</td>"
        f'<td><span class="tag {QUALITY_CLASS.get(q.verdict, "muted")}">{e(q.verdict)}</span>'
        f'<div class="note">margin {q.margin:.2f}</div></td>'
        f'<td><span class="tag {VERDICT_CLASS.get(verdict, "muted")}">{e(verdict)}</span></td>'
        "</tr>")


LINE_CAP = 240          # one rendered line
MSG_LINES = 40          # lines shown per message before it is cut
PAGE_TEXT_BUDGET = 600_000   # characters of request text per agent page
MIN_SAMPLES, MAX_SAMPLES = 3, 20


def samples_per_node(n_nodes: int) -> int:
    """How many requests to embed per callsite on this page.

    A fixed count does not work in both directions: three callsites can afford twenty requests
    each, thirty callsites cannot. Each request costs about MSG_LINES x LINE_CAP characters, so
    the budget divides by that.
    """
    per_request = MSG_LINES * LINE_CAP
    affordable = PAGE_TEXT_BUDGET // max(n_nodes, 1) // max(per_request, 1)
    return max(MIN_SAMPLES, min(MAX_SAMPLES, affordable))


def _region_lines(node: NodeResult) -> str:
    """Every region's static and varying lines, with the count behind the verdict.

    The claim the whole profile rests on, stated so it can be disagreed with: this line is in
    100 of 100 requests so we called it template; that one is in 1, so we called it payload.
    """
    a = node.metrics.anatomy
    if not a:
        return ""
    out = []
    for r in a.regions:
        if not r.static_lines and not r.dynamic_lines:
            continue
        rows = "".join(
            '<tr><td class="ok mono">static</td><td class="mono">%s</td></tr>' % e(ln[:LINE_CAP])
            for ln in r.static_lines[:40])
        rows += "".join(
            '<tr><td class="warn mono">varies</td><td class="mono note">%s</td></tr>'
            % e(ln[:LINE_CAP]) for ln in r.dynamic_lines[:20])
        hidden = max(0, len(r.static_lines) - 40) + max(0, len(r.dynamic_lines) - 20)
        if hidden:
            rows += ('<tr><td></td><td class="note">... %d more lines '
                     '(all of them are in calls.duckdb)</td></tr>' % hidden)
        out.append('<b>%s</b> &mdash; %d static, %d varying'
                   '<div class="scroll"><table>%s</table></div>'
                   % (e(r.name), len(r.static_lines), len(r.dynamic_lines), rows))
    return "".join(out)


ROLE_REGION = {"system": "instructions", "developer": "instructions",
               "user": "the user turn", "assistant": "conversation history",
               "tool": "conversation history"}


def _sample_requests(node: NodeResult, calls: Sequence[Any], budget: int) -> str:
    """Real member requests, each line carrying the verdict the profile gave it.

    A template listing is a claim; this is the claim applied to text you can read. Requests are
    sampled and truncated, or a 200k-token contract review would be the whole page.
    """
    from profiler.segment import lines_of
    from profiler.single import text_of

    a = node.metrics.anatomy
    if not a or not calls:
        return ""
    verdict = {}
    for r in a.regions:
        for ln in r.static_lines:
            verdict[(r.name, ln)] = "static"
        for ln in r.dynamic_lines:
            verdict[(r.name, ln)] = "dynamic"

    mark = {"static": ("ok", "="), "dynamic": ("warn", "~")}
    # spread the sample across the group rather than taking the first N: the opening calls of a
    # tool loop are the shallow ones, and a reader who saw only those would conclude the
    # callsite is simpler than it is
    step = max(1, len(calls) // budget)
    chosen = list(calls)[::step][:budget]
    out = []
    for index, call in enumerate(chosen):
        blocks = []
        for message in call.messages:
            role = message.get("role")
            region = ROLE_REGION.get(role, "conversation history")
            content = message.get("content")
            if isinstance(content, list):
                content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            body = lines_of(str(content or ""))
            rendered = []
            for ln in body[:MSG_LINES]:
                cls, glyph = mark.get(verdict.get((region, ln)), ("muted", "?"))
                # say how much was cut rather than hiding it: a silently truncated line looks
                # like a short line, and the reader is here to check exactly this
                shown = e(ln[:LINE_CAP])
                if len(ln) > LINE_CAP:
                    shown += ('<span class="note">[+%s chars]</span>'
                              % f"{len(ln) - LINE_CAP:,}")
                rendered.append('<div><span class="%s">%s</span> %s</div>'
                                % (cls, glyph, shown))
            if len(body) > MSG_LINES:
                rendered.append('<div class="note">... %d more lines</div>'
                                % (len(body) - MSG_LINES))
            blocks.append('<div class="msg"><b>[%s]</b>%s</div>'
                          % (e(role), "".join(rendered)))
        reply = text_of(call.response_message.get("content"))
        blocks.append('<div class="msg"><b>[response]</b> <span class="note">%s</span></div>'
                      % e(str(reply)[:300]))
        payload = getattr(call, "prompt_tokens", None)
        out.append('<div class="req%s"><div class="note mono">%s%s</div>%s</div>'
                   % ("" if index == 0 else " hidden", e(call.request_id),
                      f" &middot; {payload:,} tokens" if payload else "",
                      "".join(blocks)))

    gid = "p" + node.node.node_id.replace(":", "-").replace("#", "-").replace(".", "-")
    more = ("" if len(chosen) == len(calls) else
            " of %s in the group &mdash; <code>python inspect_calls.py --node %s --all</code> "
            "reads every one" % (f"{len(calls):,}", e(node.node.node_id)))
    pager = (
        '<div class="pager" id="%s">'
        '<button data-step="-1">&lsaquo; prev</button>'
        '<span class="count"><b>1</b> / %d</span>'
        '<button data-step="1">next &rsaquo;</button>'
        '<span class="note">%d requests shown%s</span>'
        '</div>' % (gid, len(chosen), len(chosen), more))
    legend = ('<div class="legend"><span class="ok">= shared by every request</span>'
              '<span class="warn">~ differs</span>'
              '<span class="muted">? not in the split</span></div>')
    return (legend + pager + '<div class="reqs" data-for="%s">%s</div>'
            % (gid, "".join(out)))


def _node_detail(node: NodeResult, calls: Sequence[Any] = (),
                 budget: int = MIN_SAMPLES) -> str:
    m = node.metrics
    out = [f'<details class="node"><summary>{e(_name(node))} &mdash; detail</summary>']

    a = m.anatomy
    if a:
        if a.description:
            out.append(f'<p><b>{e(a.description)}</b></p>')
        facts = []
        if a.distinct_ratio or node.node.distinct_prompts:
            facts.append(f"{node.node.distinct_prompts:,} distinct prompts across "
                         f"{m.calls:,} calls")
        if a.models:
            facts.append("models: " + ", ".join(f"{e(k)} ({v})" for k, v in a.models.items()))
        if a.finish_reasons:
            facts.append("finish reasons: "
                         + ", ".join(f"{e(k)} {v}" for k, v in a.finish_reasons.items()))
        if a.completion_p50:
            facts.append(f"reply tokens: p50 {a.completion_p50:,}, p95 {a.completion_p95:,}")
        facts.append(f"message shapes: {_shapes(node)}")
        out.append('<ul class="note">' + "".join(f"<li>{f}</li>" for f in facts) + "</ul>")

        rows = "".join(
            f'<tr><td>{e(r.name)}</td><td class="mono">{r.mean_tokens:,}</td>'
            f'<td class="mono">{r.share:.0%}</td>'
            f'<td class="mono">{r.p10:,} - {r.p90:,}</td>'
            f'<td>{"<span class=bad>varies per call</span>" if r.varies else "identical every call"}'
            f'</td></tr>' for r in a.regions)
        out.append('<b>Request anatomy</b><div class="scroll"><table>'
                   "<tr><th>region</th><th>mean tokens</th><th>share</th>"
                   "<th>p10 - p90</th><th>stability</th></tr>"
                   f"{rows}</table></div>")

    if node.history:
        h = node.history
        if h["status"] == "changed":
            out.append(f'<p class="note"><b>Changed since the last run.</b> Matched the earlier '
                       f'callsite at {h["similarity"]:.2f} &mdash; version {h["version"]}, '
                       f'+{h["lines_added"]}/-{h["lines_removed"]} template lines, first seen '
                       f'{e(h["first_seen"])}.</p>')
        elif h["status"] == "seen":
            out.append(f'<p class="note"><b>Seen before.</b> First profiled '
                       f'{e(h["first_seen"])}, {h["runs_seen"]} runs, prompt unchanged.</p>')
        else:
            out.append('<p class="note"><b>New</b> &mdash; not in any previous run.</p>')

    if m.findings:
        out.append("<b>Measured</b><ul>"
                   + "".join(f"<li>{e(f)}</li>" for f in m.findings) + "</ul>")
    if node.node.quality.reasons:
        out.append("<b>Grouping caveats</b><ul>"
                   + "".join(f"<li>{e(r)}</li>" for r in node.node.quality.reasons) + "</ul>")
    if m.tools_declared:
        never = (f" &middot; never called: {e(', '.join(m.tools_never_called[:10]))}"
                 if m.tools_never_called else "")
        out.append(f'<p class="note"><b>Tools.</b> {m.tools_declared} declared '
                   f'(~{m.tool_schema_tokens:,} tokens), {len(m.tools_called)} called{never}</p>')

    fields = node.segmentation.fields[:8]
    if fields:
        rows = "".join(
            f'<tr><td class="mono">{e(f.template[:110])}</td>'
            f'<td class="mono">{f.occurrences}</td>'
            f'<td class="note">{e((f.examples[0][:70] + chr(8230)) if f.examples else "-")}</td>'
            "</tr>" for f in fields)
        out.append('<b>What varies per call</b><div class="scroll"><table>'
                   "<tr><th>template</th><th>seen</th><th>example</th></tr>"
                   f"{rows}</table></div>")

    v = node.verdict
    if v and v.verdict not in ("confirmed", "unverified"):
        items = ([f"<li>{e(n)}</li>" for n in v.notes]
                 + [f"<li>claimed static but varies: <code>{e(x)}</code></li>"
                    for x in v.template_errors]
                 + [f"<li>varies but was not listed: <code>{e(x)}</code></li>"
                    for x in v.missed_dynamic])
        out.append(f'<p><b class="bad">The judge disagreed &mdash; {e(v.verdict)}</b></p>'
                   f"<ul>{''.join(items)}</ul>")

    sig = node.node.signature or {}
    if sig:
        total = max(m.calls, 1)
        shapes = _shapes(node)
        rows = [
            ("system prompt present", f"{sig.get('with_system_prompt', 0):,} of {total:,}"),
            ("user template present", f"{sig.get('with_user_template', 0):,} of {total:,}"),
            ("thin prompts", f"{sig.get('weak_prompts', 0):,}"),
            ("median prompt lines", f"{sig.get('median_prompt_lines', 0):,}"),
            ("distinct tool sets", f"{sig.get('distinct_tool_sets', 0):,}"),
            ("distinct response formats", f"{sig.get('distinct_formats', 0):,}"),
            ("message shapes", shapes),
        ]
        out.append("<details><summary><b>Fingerprint</b> &mdash; what the five views saw"
                   "</summary><div class='scroll'><table>"
                   + "".join("<tr><td>%s</td><td class='mono'>%s</td></tr>" % (e(k), v)
                             for k, v in rows)
                   + "</table></div></details>")

    lines = _region_lines(node)
    if lines:
        out.append("<details><summary><b>Static vs varying, line by line</b></summary>"
                   + lines + "</details>")

    samples = _sample_requests(node, calls, budget)
    if samples:
        out.append("<details><summary><b>Real requests from this group</b> &mdash; whole, "
                   "with every line marked</summary>" + samples + "</details>")

    if node.node.template:
        shown = node.node.template[:150]
        more = (f"\n...[{len(node.node.template)} lines total]"
                if len(node.node.template) > 150 else "")
        out.append(f"<b>Template</b><pre>{e(chr(10).join(shown))}{e(more)}</pre>")

    out.append("</details>")
    return "".join(out)


def _agent(app: AppResult, calls: Sequence[Any] = ()) -> str:
    single = app.single_call or {}
    prompt = single.get("total_prompt_tokens") or 0
    cached = single.get("total_cached_tokens") or 0
    app_tokens = _tokens_of(app)

    ordered = sorted(app.nodes, key=lambda n: -(n.metrics.calls * n.metrics.avg_request_tokens))
    palette = ["accent", "recover", "ok", "warn", "bad", "now"]
    split = _bar([((n.metrics.calls * n.metrics.avg_request_tokens) / max(app_tokens, 1),
                   palette[i % len(palette)], "") for i, n in enumerate(ordered)])

    chips = [f'<span class="chip"><b>{app.calls:,}</b> calls</span>',
             f'<span class="chip"><b>{len(app.nodes)}</b> callsites</span>']
    if app.chain:
        c = app.chain
        if c.verdict == "unusable":
            chips.append('<span class="chip bad"><b>runs not recoverable</b></span>')
        elif c.verdict == "single_step":
            chips.append('<span class="chip">single-step traffic</span>')
        else:
            chips.append(f'<span class="chip"><b>{c.runs:,}</b> runs, '
                         f"{c.mean_steps} steps avg</span>")
    if prompt:
        chips.append(f'<span class="chip"><b>{prompt:,}</b> prompt tokens</span>')
        chips.append(f'<span class="chip">cache <b>{cached / prompt:.0%}</b></span>')
    if app.residual_calls:
        chips.append(f'<span class="chip"><b>{app.residual_calls}</b> unprofiled</span>')

    body = [f'<div class="body">{split}']
    if app.chain and app.chain.note:
        body.append(f'<p class="note">{e(app.chain.note)}</p>')
    if app.chain and app.chain.top_paths:
        rows = "".join(f'<tr><td class="mono">{e(p["path"])}</td>'
                       f'<td class="mono">{p["runs"]}</td></tr>'
                       for p in app.chain.top_paths)
        body.append("<div class='scroll'><table><tr><th>most common call sequence</th>"
                    f"<th>runs</th></tr>{rows}</table></div>")

    rows = "".join(_node_row(n, app_tokens) for n in ordered)
    body.append("<div class='scroll'><table><tr><th>callsite</th><th>calls</th>"
                "<th>avg tok</th><th>where the tokens sit</th><th>what it is made of</th>"
                f"<th>grouping</th><th>judge</th></tr>{rows}</table></div>")
    by_id = {c.request_id: c for c in calls}
    budget = samples_per_node(len(ordered))
    body.append("".join(
        _node_detail(n, [by_id[r] for r in n.node.request_ids if r in by_id], budget)
        for n in ordered))
    body.append("</div>")

    return (f'<details class="agent" open><summary><h3>{e(app.app_id)}</h3>'
            f'{"".join(chips)}</summary>{"".join(body)}</details>')


def _limits_for(result: RunResult, apps: Sequence[AppResult]) -> str:
    """State the limits that apply to THESE agents.

    An agent page repeating the estate's caveats would assert things untrue of the agent being
    read, which is how a limits section stops being read at all.
    """
    items = []
    unusable = [a.app_id for a in apps if a.chain and a.chain.verdict == "unusable"]
    if unusable:
        subject = ("This agent compacts or windows its history"
                   if len(unusable) == 1 else
                   f"{len(unusable)} agents compact or window their history "
                   f"({e(', '.join(unusable[:6]))})")
        items.append(f"<b>Runs could not be reconstructed.</b> {subject}, which destroys the "
                     f"shared message prefix before the gateway sees the request. Not "
                     f"recoverable from logs &mdash; it needs a run-id header from the client.")
    residual = sum(a.residual_calls for a in apps)
    if residual:
        items.append(f"<b>{residual:,} calls fell below the size floor</b> and have no callsite. "
                     f"They are counted in the totals.")
    ids = {n.node.node_id for a in apps for n in a.nodes}
    problems = [p for p in ((result.judge_summary or {}).get("problems") or [])
                if p.get("node_id") in ids]
    if problems:
        items.append(f"<b>{len(problems)} callsite profiles were flagged by the judge</b> and "
                     f"are shown as produced. Nothing was silently corrected.")
    items.append("<b>Figures are token counts</b>, not billed amounts &mdash; no price list is "
                 "applied.")
    items.append("<b>No optimisation advice is produced.</b> That layer was removed until the "
                 "profile half is trustworthy on real data.")
    return ("<h2>What this does not tell you</h2><ul class='limits'>"
            + "".join(f"<li>{i}</li>" for i in items) + "</ul>")


def render_agent(result: RunResult, app: AppResult) -> str:
    """One agent, standalone and sendable to the team that owns it."""
    return (f"<title>{e(app.app_id)} - agent profile</title>\n<style>{CSS}</style>\n"
            '<div class="wrap">'
            '<div class="sub"><a href="index.html">&larr; all agents</a></div>'
            f"<h1>{e(app.app_id)}</h1>"
            f'<div class="sub mono">run {e(result.run_id)}</div>'
            f"{_agent(app, result.calls_by_app.get(app.app_id, ()))}"
            f"{_limits_for(result, [app])}</div>" + PAGER_SCRIPT)


def render_index(result: RunResult) -> str:
    """The estate: every agent ranked by tokens, with its warnings, linking out."""
    nodes = [n for a in result.apps for n in a.nodes]
    calls = sum(a.calls for a in result.apps)
    judged = result.judge_summary or {}
    inp = result.load_report

    if not result.llm_usage["calls"]:
        cls, verdict = "warn", "UNVERIFIED &mdash; no LLM key, so nothing checked these profiles"
    elif result.gate_passed:
        cls, verdict = "ok", f"VERIFIED &mdash; {e(result.gate_reason)}"
    else:
        cls, verdict = "bad", f"NOT VERIFIED &mdash; {e(result.gate_reason)}"

    cards = [("agents", f"{len(result.apps)}"), ("callsites", f"{len(nodes)}"),
             ("calls", f"{calls:,}")]
    rate = judged.get("confirmed_rate")
    if rate is not None:
        cards.append(("profiles confirmed", f"{rate:.0%}"))
    correct = judged.get("boundary_correct")
    if correct is not None:
        cards.append(("boundary test", f"{correct:.0%}"))
    cards.append(("quality warnings",
                  f"{sum(1 for n in nodes if n.node.quality.verdict != 'ok')}"))
    card_html = "".join(f'<div class="card"><div class="n">{v}</div>'
                        f'<div class="l">{k}</div></div>' for k, v in cards)

    ranked = sorted(result.apps, key=lambda a: -_tokens_of(a))
    biggest = max((_tokens_of(a) for a in ranked), default=1) or 1

    rows = []
    for app in ranked:
        tok = _tokens_of(app)
        single = app.single_call or {}
        prompt = single.get("total_prompt_tokens") or 0
        cached = single.get("total_cached_tokens") or 0

        flags = []
        if app.chain and app.chain.verdict == "unusable":
            flags.append('<span class="tag bad">runs unrecoverable</span>')
        bad = sum(1 for n in app.nodes
                  if n.verdict and n.verdict.verdict not in ("confirmed", "unverified"))
        if bad:
            flags.append(f'<span class="tag warn">{bad} flagged</span>')
        suspect = sum(1 for n in app.nodes if n.node.quality.verdict == "suspect")
        if suspect:
            flags.append(f'<span class="tag warn">{suspect} suspect</span>')
        if app.residual_calls:
            flags.append(f'<span class="tag muted">{app.residual_calls} unprofiled</span>')

        # tokens re-sent uncached because something dynamic sits ahead of stable text: the one
        # number worth ranking an estate on
        recover = sum(n.metrics.recoverable * n.metrics.calls for n in app.nodes)

        rows.append(
            "<tr>"
            f'<td><a href="{e(_slug(app.app_id))}.html"><b>{e(app.app_id)}</b></a>'
            f'<div>{"".join(flags)}</div></td>'
            f'<td class="mono">{app.calls:,}</td>'
            f'<td class="mono">{len(app.nodes)}</td>'
            f'<td style="min-width:180px">{_bar([(tok / biggest, "accent", "")])}'
            f'<div class="note mono">{tok:,} tok</div></td>'
            f'<td class="mono">{f"{cached / prompt:.0%}" if prompt else "&mdash;"}</td>'
            f'<td class="mono">{f"{recover:,}" if recover else "&mdash;"}</td>'
            "</tr>")

    return (f"<title>Agent profile {e(result.run_id)}</title>\n<style>{CSS}</style>\n"
            '<div class="wrap"><h1>Agent profile</h1>'
            f'<div class="sub mono">{e(result.run_id)} &middot; {inp["loaded"]:,} of '
            f'{inp["total_rows"]:,} rows &middot; {result.llm_usage["calls"]} LLM calls '
            f'(${result.llm_usage["cost_usd"]}) &middot; {result.elapsed_s}s</div>'
            f'<div class="banner {cls}">{verdict}</div>'
            f'<div class="cards">{card_html}</div>'
            '<p class="note">A <b>callsite</b> is one place in a program that calls a model. '
            "The count is discovered from the requests &mdash; nothing declares it. Open an "
            "agent for its callsites.</p>"
            "<h2>Agents, by tokens</h2>"
            "<input id='filter' placeholder='filter agents by name or flag "
            "(try: flagged, suspect, unrecoverable)'>"
            "<div class='scroll'><table id='agents'>"
            "<tr><th>agent</th><th>calls</th><th>callsites</th><th>tokens</th><th>cache</th>"
            "<th>recoverable</th></tr>"
            f"{''.join(rows)}</table></div>"
            "<script>(function(){var box=document.getElementById('filter'),rows=document.querySelectorAll('#agents tr');box.addEventListener(\"input\",function(){var q=box.value.toLowerCase();for(var i=1;i<rows.length;i++){rows[i].classList.toggle('hidden',q!==\"\"&&rows[i].textContent.toLowerCase().indexOf(q)<0);}});})();</script>"
            f"{_limits_for(result, result.apps)}</div>")


def _write(path: str, body: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html>\n<html><head><meta charset='utf-8'>"
                     "<meta name='viewport' content='width=device-width,initial-scale=1'>"
                     "</head><body>\n")
        handle.write(body)
        handle.write("\n</body></html>\n")


def write_site(result: RunResult, directory: str) -> str:
    """Write index.html plus one page per agent. Returns the index path."""
    os.makedirs(directory, exist_ok=True)
    index = os.path.join(directory, "index.html")
    _write(index, render_index(result))
    for app in result.apps:
        _write(os.path.join(directory, f"{_slug(app.app_id)}.html"), render_agent(result, app))
    return index
