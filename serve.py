"""Browse the store live, instead of reading a page that was frozen at write time.

    python serve.py                       # latest run, opens a browser
    python serve.py --store path/to/calls.duckdb --port 8900

The static pages have to choose what to embed: roughly 600k characters per page, divided among
that agent's callsites, which on a busy agent works out at three requests each. That is enough
to skim and not enough to be sure. Reading straight from DuckDB removes the choice - **every**
request in a group is one click away, whatever the group's size, because nothing is embedded
until it is asked for.

The two are for different things and both stay:

    out/run-<id>/*.html   frozen, shareable, works with no Python - send it to a team
    python serve.py       live, complete, local - convince yourself

Read-only, bound to 127.0.0.1, stdlib only. It never writes to the store and never leaves the
machine.
"""

from __future__ import annotations

import argparse
import glob
import html
import json
import os
import sys
import threading
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from profiler.report_html import CSS, ROLE_REGION

# The line-listing table shows hundreds of rows at once, so a long line is cut there to keep
# the table readable. A REQUEST is different: it is fetched on demand, one at a time, straight
# from the store - nothing is embedded ahead of time and there is no page budget to protect. So
# a request renders whole. Cutting it was inherited from the static page, where the cap is a
# real constraint, and it made the live view answer a question nobody asked it.
TABLE_LINE_CAP = 400    # one line inside the static/dynamic listing
HUGE_REQUEST = 4_000_000  # only a pathological request is cut, and it says so

EXTRA_CSS = """
.crumbs { font-size:12.5px; color:var(--dim); margin-bottom:14px; }
.crumbs a { margin-right:6px; }
.kv { display:grid; grid-template-columns:220px 1fr; gap:2px 14px; font-size:13px;
      margin:10px 0; }
.kv div:nth-child(odd) { color:var(--dim); }
.req { border-left:2px solid var(--line); padding-left:12px; margin:10px 0; }
.pager { display:flex; align-items:center; gap:10px; margin:12px 0; flex-wrap:wrap; }
.pager a, .pager span.btn { font-size:12px; padding:4px 11px; border:1px solid var(--line);
      border-radius:6px; text-decoration:none; }
.pager span.btn { opacity:.4; }
.jump { font-size:12px; }
.jump input { width:70px; padding:3px 6px; border:1px solid var(--line); border-radius:5px;
      background:var(--panel); color:var(--ink); }
"""


def url_of(value: Any) -> str:
    """URL-quote a path segment, keeping `:` readable.

    Node ids look like `app_agent:f5c8190b87e5`. Escaping the colon turns every link
    into `app_agent%3Af5c8...`, unreadable both in the address bar and in a link
    someone pastes to a colleague. A colon is legal in a path segment. `#`, which
    the id can carry from de-duplication, is not, and stays escaped.
    """
    return urllib.parse.quote(str(value), safe=":")


def e(x: Any) -> str:
    return html.escape(str(x if x is not None else ""))


def page(title: str, body: str) -> bytes:
    return ("<!doctype html><html><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{e(title)}</title><style>{CSS}{EXTRA_CSS}</style></head><body>"
            f"<div class='wrap'>{body}</div></body></html>").encode("utf-8")


def crumbs(*parts: Tuple[str, Optional[str]]) -> str:
    out = []
    for label, href in parts:
        out.append(f"<a href='{e(href)}'>{e(label)}</a> /" if href else f"<span>{e(label)}</span>")
    return f"<div class='crumbs'>{''.join(out)}</div>"


def table(headers, rows) -> str:
    if not rows:
        return "<p class='note'>none</p>"
    head = "".join(f"<th>{e(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<div class='scroll'><table><tr>{head}</tr>{body}</table></div>"


class Store:
    """Read-only access to one run's DuckDB file."""

    def __init__(self, path: str) -> None:
        self.path = path
        # one connection per thread: a DuckDB connection is not shared safely across threads,
        # and ThreadingHTTPServer will hand requests to several at once
        self._local = threading.local()

    @property
    def con(self):
        if not hasattr(self._local, "con"):
            self._local.con = duckdb.connect(self.path, read_only=True)
        return self._local.con

    def q(self, sql: str, args: Optional[list] = None) -> List[tuple]:
        return self.con.execute(sql, args or []).fetchall()


# ---------------------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------------------

def view_index(store: Store) -> bytes:
    run = store.q("SELECT run_id, calls, apps, callsites, gate_passed, gate_reason FROM runs")
    run_id, calls, apps, callsites, passed, reason = (
        run[0] if run else ("?", 0, 0, 0, None, ""))

    rows = store.q(
        "SELECT app_id, COUNT(*) AS n, SUM(calls) AS c, SUM(recoverable * calls) AS rec "
        "FROM nodes GROUP BY app_id ORDER BY c DESC")
    body = [crumbs(("all agents", None)),
            f"<h1>Agent profile</h1><div class='sub mono'>{e(run_id)} &middot; {calls:,} calls "
            f"&middot; {apps} agents &middot; {callsites} callsites</div>"]
    if reason:
        cls = "ok" if passed else "bad"
        body.append(f"<div class='banner {cls}'>{e(reason)}</div>")
    body.append(table(
        ["agent", "callsites", "calls", "recoverable tokens"],
        [[f"<a href='/agent/{url_of(a)}'><b>{e(a)}</b></a>",
          f"<span class='mono'>{n}</span>", f"<span class='mono'>{c:,}</span>",
          f"<span class='mono'>{int(rec or 0):,}</span>"]
         for a, n, c, rec in rows]))

    orphans = store.q("SELECT app_id, COUNT(*) FROM calls WHERE node_id IS NULL GROUP BY 1")
    if orphans:
        body.append("<h2>Calls in no callsite</h2>")
        body.append(table(["agent", "calls"],
                          [[e(a), f"<span class='mono'>{n:,}</span>"] for a, n in orphans]))
    return page("Agent profile", "".join(body))


def view_agent(store: Store, app_id: str) -> bytes:
    rows = store.q(
        "SELECT node_id, name, calls, avg_request_tokens, cacheable_now, recoverable, "
        "quality, judge_verdict, description FROM nodes WHERE app_id = ? ORDER BY calls DESC",
        [app_id])
    body = [crumbs(("all agents", "/"), (app_id, None)), f"<h1>{e(app_id)}</h1>"]
    body.append(table(
        ["callsite", "calls", "avg tokens", "cacheable", "recoverable", "grouping", "judge"],
        [[f"<a href='/node/{url_of(nid)}'><b>{e(name or nid)}</b></a>"
          f"<div class='note'>{e((desc or '')[:150])}</div>",
          f"<span class='mono'>{c:,}</span>", f"<span class='mono'>{avg:,}</span>",
          f"<span class='mono'>{cache:,}</span>", f"<span class='mono'>{rec:,}</span>",
          f"<span class='tag'>{e(q)}</span>", f"<span class='tag'>{e(j or 'n/a')}</span>"]
         for nid, name, c, avg, cache, rec, q, j, desc in rows]))
    return page(app_id, "".join(body))


def view_node(store: Store, node_id: str) -> bytes:
    row = store.q(
        "SELECT app_id, name, purpose, calls, avg_request_tokens, cacheable_now, recoverable, "
        "distinct_prompts, quality, quality_margin, judge_verdict, description, findings, "
        "fingerprint FROM nodes WHERE node_id = ?", [node_id])
    if not row:
        return page("not found", "<h1>No such callsite</h1>")
    (app, name, purpose, calls, avg, cache, rec, distinct, quality, margin, judge,
     description, findings, fingerprint) = row[0]

    body = [crumbs(("all agents", "/"), (app, f"/agent/{url_of(app)}"),
                   (name or node_id, None)),
            f"<h1>{e(name or node_id)}</h1><div class='sub mono'>{e(node_id)}</div>"]
    if purpose:
        body.append(f"<p>{e(purpose)}</p>")
    if description:
        body.append(f"<p><b>{e(description)}</b></p>")

    body.append("<div class='kv'>"
                f"<div>calls</div><div class='mono'>{calls:,}</div>"
                f"<div>average request</div><div class='mono'>{avg:,} tokens</div>"
                f"<div>distinct prompts</div><div class='mono'>{distinct:,}</div>"
                f"<div>cacheable now</div><div class='mono'>{cache:,}</div>"
                f"<div>recoverable by reorder</div><div class='mono'>{rec:,}</div>"
                f"<div>grouping quality</div><div>{e(quality)} (margin {margin:.2f})</div>"
                f"<div>judge</div><div>{e(judge or 'not run')}</div>"
                "</div>")

    for finding in json.loads(findings or "[]"):
        body.append(f"<p class='note'>&bull; {e(finding)}</p>")

    fp = json.loads(fingerprint or "{}")
    if fp:
        shapes = ", ".join(f"{s['shape']}&times;{s['calls']}"
                           for s in (fp.get("shapes") or [])[:6])
        body.append("<h2>Fingerprint</h2><div class='kv'>"
                    f"<div>system prompt present</div><div class='mono'>"
                    f"{fp.get('with_system_prompt', 0):,} of {calls:,}</div>"
                    f"<div>user template present</div><div class='mono'>"
                    f"{fp.get('with_user_template', 0):,} of {calls:,}</div>"
                    f"<div>thin prompts</div><div class='mono'>{fp.get('weak_prompts', 0):,}"
                    f"</div>"
                    f"<div>median prompt lines</div><div class='mono'>"
                    f"{fp.get('median_prompt_lines', 0):,}</div>"
                    f"<div>distinct tool sets</div><div class='mono'>"
                    f"{fp.get('distinct_tool_sets', 0):,}</div>"
                    f"<div>message shapes</div><div class='mono'>{shapes or '&mdash;'}</div>"
                    "</div>")

    regions = store.q(
        "SELECT region, mean_tokens, share, p10, p90, static_share, varies FROM regions "
        "WHERE node_id = ? ORDER BY mean_tokens DESC", [node_id])
    if regions:
        body.append("<h2>What it is made of</h2>")
        body.append(table(
            ["region", "mean tokens", "share", "p10 - p90", "static", "stability"],
            [[e(r), f"<span class='mono'>{m:,}</span>", f"<span class='mono'>{sh:.0%}</span>",
              f"<span class='mono'>{p10:,} - {p90:,}</span>",
              f"<span class='mono'>{st:.0%}</span>",
              "<span class='warn'>varies per call</span>" if v
              else "<span class='ok'>identical every call</span>"]
             for r, m, sh, p10, p90, st, v in regions]))

    payloads = [r[0] for r in store.q(
        "SELECT dynamic_tokens FROM calls WHERE node_id = ? AND dynamic_tokens IS NOT NULL "
        "ORDER BY dynamic_tokens", [node_id])]
    if payloads:
        spread = payloads[-1] / max(payloads[0], 1)
        note = (f"<span class='warn'>the largest carries {spread:,.0f}&times; the smallest"
                f"</span>" if spread >= 10 else "<span class='ok'>uniform</span>")
        body.append(f"<p class='note'>Payload per request: min {payloads[0]:,}, median "
                    f"{payloads[len(payloads) // 2]:,}, max {payloads[-1]:,} tokens &mdash; "
                    f"{note}</p>")

    body.append("<h2>Static vs varying, line by line</h2>")
    for region, *_ in regions:
        lines = store.q(
            "SELECT kind, doc_frequency, calls, line FROM region_lines WHERE node_id = ? "
            "AND region = ? ORDER BY kind, doc_frequency DESC LIMIT 400", [node_id, region])
        if not lines:
            continue
        body.append(f"<h3 class='note'>{e(region)}</h3>")
        body.append(table(
            ["seen in", "line"],
            [[f"<span class='mono {'ok' if k == 'static' else 'warn'}'>{df}/{tot}</span>",
              f"<span class='mono'>{e(ln[:TABLE_LINE_CAP])}</span>"]
             for k, df, tot, ln in lines]))

    total = store.q("SELECT COUNT(*) FROM calls WHERE node_id = ?", [node_id])[0][0]
    body.append(f"<h2>Every request in this group</h2>"
                f"<p class='note'>{total:,} of them, read one at a time straight from the "
                f"store &mdash; nothing is sampled here.</p>"
                f"<p><a href='/node/{url_of(node_id)}/call/0'>"
                f"open the first request &rsaquo;</a></p>")
    return page(name or node_id, "".join(body))


def view_call(store: Store, node_id: str, index: int) -> bytes:
    from profiler.segment import lines_of

    total = store.q("SELECT COUNT(*) FROM calls WHERE node_id = ?", [node_id])[0][0]
    if total == 0:
        return page("not found", "<h1>No calls in that group</h1>")
    index = max(0, min(index, total - 1))

    row = store.q(
        "SELECT request_id, ts, prompt_tokens, dynamic_tokens, finish_reason, request_json, "
        "response_json FROM calls WHERE node_id = ? ORDER BY ts LIMIT 1 OFFSET ?",
        [node_id, index])[0]
    request_id, ts, prompt_tokens, dyn, finish, request_json, response_json = row

    verdict = {}
    for region, kind, line in store.q(
            "SELECT region, kind, line FROM region_lines WHERE node_id = ?", [node_id]):
        verdict[(region, line[:4000])] = kind

    name = store.q("SELECT name, app_id FROM nodes WHERE node_id = ?", [node_id])
    label, app = (name[0] if name else (node_id, ""))

    def link(i: int, text: str) -> str:
        if 0 <= i < total:
            return (f"<a href='/node/{url_of(node_id)}/call/{i}'>{text}</a>")
        return f"<span class='btn'>{text}</span>"

    pager = (f"<div class='pager'>{link(index - 1, '&lsaquo; prev')}"
             f"<span class='mono'>{index + 1:,} / {total:,}</span>"
             f"{link(index + 1, 'next &rsaquo;')}"
             f"<form class='jump' action='/jump' method='get'>"
             f"<input type='hidden' name='node' value='{e(node_id)}'>"
             f"<input name='i' placeholder='{index + 1}' inputmode='numeric'> "
             f"<button type='submit' style='display:none'></button>go to</form>"
             f"<span class='note'>{e(request_id)} &middot; {prompt_tokens or 0:,} tokens, "
             f"{dyn or 0:,} of it payload &middot; {e(finish)}</span></div>")

    request = json.loads(request_json)
    response = json.loads(response_json)
    oversized = len(request_json) > HUGE_REQUEST
    out = [crumbs(("all agents", "/"), (app, f"/agent/{url_of(app)}"),
                  (label or node_id, f"/node/{url_of(node_id)}"),
                  (f"request {index + 1}", None)),
           f"<h1>{e(label or node_id)}</h1>",
           "<div class='legend'><span class='ok'>= shared by every request</span>"
           "<span class='warn'>~ differs</span>"
           "<span class='muted'>? not in the split</span>"
           f"<span class='note'>{prompt_tokens or 0:,} tokens, shown whole</span></div>",
           pager, "<div class='req'>"]

    for tool in request.get("tools") or []:
        fn = tool.get("function") or {}
        out.append(f"<div class='msg note'>[tool] {e(fn.get('name'))}: "
                   f"{e(str(fn.get('description'))[:200])}</div>")

    for message in request.get("messages") or []:
        role = message.get("role")
        region = ROLE_REGION.get(role, "conversation history")
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        rendered = []
        for line in lines_of(str(content or "")):
            kind = verdict.get((region, line[:4000]))
            cls, glyph = {"static": ("ok", "="),
                          "dynamic": ("warn", "~")}.get(kind, ("muted", "?"))
            rendered.append(f"<div><span class='{cls}'>{glyph}</span> {e(line)}</div>")
        out.append(f"<div class='msg'><b>[{e(role)}]</b>{''.join(rendered)}</div>")

    message = response.get("message") or {}
    out.append(f"<div class='msg'><b>[response]</b> <span class='note'>"
               f"{e(str(message.get('content') or '')[:2000])}</span></div>")
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        out.append(f"<div class='msg note'>&rarr; {e(fn.get('name'))}"
                   f"({e(str(fn.get('arguments'))[:200])})</div>")

    out.append("</div>")
    out.append(pager)
    return page(f"{label} - request {index + 1}", "".join(out))


class Handler(BaseHTTPRequestHandler):
    store: Store = None            # set on the class before serving

    def log_message(self, *args):  # keep the console for the run, not for every asset
        pass

    def _send(self, payload: bytes, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        parts = [urllib.parse.unquote(p) for p in parsed.path.strip("/").split("/") if p]
        query = urllib.parse.parse_qs(parsed.query)
        try:
            if not parts:
                return self._send(view_index(self.store))
            if parts[0] == "agent" and len(parts) == 2:
                return self._send(view_agent(self.store, parts[1]))
            if parts[0] == "node" and len(parts) == 2:
                return self._send(view_node(self.store, parts[1]))
            if parts[0] == "node" and len(parts) == 4 and parts[2] == "call":
                return self._send(view_call(self.store, parts[1], int(parts[3])))
            if parts[0] == "jump":
                node = (query.get("node") or [""])[0]
                index = max(1, int((query.get("i") or ["1"])[0] or 1)) - 1
                self.send_response(302)
                self.send_header(
                    "Location", f"/node/{url_of(node)}/call/{index}")
                self.end_headers()
                return
            self._send(page("not found", "<h1>Not found</h1><p><a href='/'>start over</a></p>"),
                       404)
        except Exception as error:                      # a broken view must not kill the server
            self._send(page("error", f"<h1>Error</h1><pre>{e(error)}</pre>"), 500)


def find_stores(*roots: str) -> List[str]:
    """Every run's store, oldest first. Searches any output directory, not just `out/`."""
    found: List[str] = []
    for root in roots:
        found += glob.glob(os.path.join(root, "run-*", "calls.duckdb"))
        found += glob.glob(os.path.join(root, "*", "run-*", "calls.duckdb"))
    return sorted(set(found), key=os.path.getmtime)


def resolve_store(given: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Turn whatever the user passed into a store path, plus the alternatives.

    `--store` accepts the file, the run directory, or an output directory - guessing which of
    those someone meant is cheaper than making them look it up.
    """
    if given:
        if os.path.isfile(given):
            return given, []
        candidates = find_stores(given)
        return (candidates[-1] if candidates else None), candidates[:-1]
    candidates = find_stores(HERE)
    return (candidates[-1] if candidates else None), candidates[:-1]


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=None, help="calls.duckdb (default: latest run)")
    parser.add_argument("--port", type=int, default=8900)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    path, others = resolve_store(args.store)
    if not path or not os.path.exists(path):
        print("no store found - run `python run.py --csv <file>` first")
        return 1

    Handler.store = Store(path)
    # 127.0.0.1, never 0.0.0.0: gateway logs are the most sensitive thing an estate has, and
    # this serves them unauthenticated
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}/"
    print(f"serving {os.path.relpath(path, HERE)}")
    for other in reversed(others[-4:]):
        print(f"  other run: --store {os.path.relpath(other, HERE)}")
    print(f"  {url}   (ctrl-c to stop)")
    if not args.no_open:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
