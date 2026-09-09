"""Open a callsite and read what is actually in it.

The profile says "these 60 calls are one callsite, these lines are template, those are payload".
This is how you check that claim instead of taking it.

    python inspect_calls.py                            list every callsite
    python inspect_calls.py --node app_rag:2a02ef88    the callsite: regions, static, dynamic
    python inspect_calls.py --node app_rag:2a02ef88 --call 3
                                                       one real request, static lines dimmed
                                                       and payload marked, so you can see the
                                                       split applied to actual text
    python inspect_calls.py --node ... --all           EVERY request in the group IN FULL,
                                                       each line marked = shared / ~ differs.
                                                       The view for deciding by eye that these
                                                       really are one callsite.
    python inspect_calls.py --node ... --all --brief   the same, differing lines only
    python inspect_calls.py --node ... --export dir/   every request as its own file, to read
                                                       or diff outside the terminal
    python inspect_calls.py --sql "SELECT ..."         anything else

Reads the DuckDB file a run writes. Node ids may be given as a unique prefix.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, List, Optional

import duckdb

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

# Which region of the profile a message role belongs to. Shared by every view here, and it must
# match `report_html.ROLE_REGION`: a line looked up under one region and filed under another
# comes back "unknown" and the whole static/dynamic marking silently degrades.
ROLE_REGION = {"system": "instructions", "developer": "instructions",
               "user": "the user turn", "assistant": "conversation history",
               "tool": "conversation history"}

DIM = "\033[2m"
BOLD = "\033[1m"
YELLOW = "\033[33m"
GREEN = "\033[32m"
RESET = "\033[0m"


def _colour(enabled: bool):
    if enabled:
        return DIM, BOLD, YELLOW, GREEN, RESET
    return "", "", "", "", ""


def latest_store(out_dir: str) -> Optional[str]:
    found = sorted(glob.glob(os.path.join(out_dir, "run-*", "calls.duckdb")))
    return found[-1] if found else None


def resolve_node(con: Any, prefix: str) -> Optional[str]:
    rows = con.execute("SELECT node_id FROM nodes WHERE node_id LIKE ?",
                       [f"{prefix}%"]).fetchall()
    if not rows:
        return None
    if len(rows) > 1:
        print(f"'{prefix}' matches {len(rows)} callsites:")
        for (nid,) in rows[:10]:
            print(f"  {nid}")
        return None
    return rows[0][0]


def list_nodes(con: Any) -> None:
    rows = con.execute(
        "SELECT app_id, node_id, name, calls, avg_request_tokens, recoverable, "
        "quality, judge_verdict FROM nodes ORDER BY app_id, calls DESC").fetchall()
    print(f"{'agent':<16}{'callsite':<26}{'name':<30}{'calls':>7}{'avg tok':>9}"
          f"{'recover':>9}  quality / judge")
    print("-" * 118)
    for app, nid, name, calls, avg, recover, quality, judge in rows:
        print(f"{app:<16}{nid:<26}{(name or '')[:29]:<30}{calls:>7,}{avg:>9,}"
              f"{recover or 0:>9,}  {quality} / {judge or 'n/a'}")
    unassigned = con.execute(
        "SELECT app_id, COUNT(*) FROM calls WHERE node_id IS NULL GROUP BY app_id").fetchall()
    if unassigned:
        print("\ncalls below the size floor, in no callsite:")
        for app, n in unassigned:
            print(f"  {app:<16}{n:>7,}")


def show_node(con: Any, node_id: str, colour: bool, limit: int) -> None:
    dim, bold, yellow, green, reset = _colour(colour)
    row = con.execute(
        "SELECT app_id, name, purpose, calls, avg_request_tokens, cacheable_now, recoverable, "
        "distinct_prompts, quality, quality_margin, judge_verdict, description, findings, "
        "fingerprint FROM nodes WHERE node_id = ?", [node_id]).fetchone()
    if not row:
        print(f"no such callsite: {node_id}")
        return
    (app, name, purpose, calls, avg, cacheable, recover, distinct, quality, margin,
     judge, description, findings, fingerprint) = row

    print("=" * 100)
    print(f"{bold}{name or node_id}{reset}   {dim}{node_id}{reset}")
    print("=" * 100)
    if purpose:
        print(f"  {purpose}")
    if description:
        print(f"\n  {description}")
    print(f"\n  {calls:,} calls, {avg:,} avg tokens, {distinct:,} distinct prompts")
    print(f"  cacheable now {cacheable:,}, recoverable by reorder {recover:,}")
    print(f"  grouping quality: {quality} (margin {margin:.2f}), judge: {judge or 'not run'}")
    for finding in json.loads(findings or "[]"):
        print(f"    - {finding}")

    fp = json.loads(fingerprint or "{}")
    if fp:
        print(f"\n{bold}FINGERPRINT{reset}  {dim}what the five views saw across these "
              f"{calls:,} calls{reset}")
        shapes = ", ".join(f"{s['shape']} x{s['calls']}" for s in (fp.get("shapes") or [])[:5])
        print(f"  system prompt present  {fp.get('with_system_prompt', 0):>6,} of {calls:,}")
        print(f"  user template present  {fp.get('with_user_template', 0):>6,} of {calls:,}")
        print(f"  thin prompts           {fp.get('weak_prompts', 0):>6,} "
              f"{dim}(too few lines to identify anything alone){reset}")
        print(f"  median prompt lines    {fp.get('median_prompt_lines', 0):>6,}")
        print(f"  distinct tool sets     {fp.get('distinct_tool_sets', 0):>6,}")
        print(f"  distinct formats       {fp.get('distinct_formats', 0):>6,}")
        print(f"  message shapes         {shapes or 'none'}")
        if fp.get("distinct_shapes", 1) > 1:
            print(f"    {dim}several shapes at one callsite is normal for an agent loop - "
                  f"the same code at different depths{reset}")

    print(f"\n{bold}REGIONS{reset}")
    regions = con.execute(
        "SELECT region, mean_tokens, share, p10, p90, static_share, varies "
        "FROM regions WHERE node_id = ? ORDER BY mean_tokens DESC", [node_id]).fetchall()
    print(f"  {'region':<22}{'mean tok':>10}{'share':>8}{'p10-p90':>16}{'static':>9}   verdict")
    for region, mean, share, p10, p90, static, varies in regions:
        mark = f"{yellow}VARIES{reset}" if varies else f"{green}identical{reset}"
        print(f"  {region:<22}{mean:>10,}{share:>7.0%}{f'{p10:,}-{p90:,}':>16}"
              f"{static:>8.0%}   {mark}")

    payloads = sorted(r[0] for r in con.execute(
        "SELECT dynamic_tokens FROM calls WHERE node_id = ? AND dynamic_tokens IS NOT NULL",
        [node_id]).fetchall())
    if payloads:
        median = payloads[len(payloads) // 2]
        spread = payloads[-1] / max(payloads[0], 1)
        note = (f"{yellow}the largest carries {spread:,.0f}x the smallest{reset}"
                if spread >= 10 else f"{green}uniform across the group{reset}")
        print(f"\n{bold}PAYLOAD PER REQUEST{reset}  min {payloads[0]:,}  median {median:,}  "
              f"max {payloads[-1]:,} tokens  -  {note}")

    for region, *_ in regions:
        static = con.execute(
            "SELECT line, doc_frequency, calls FROM region_lines WHERE node_id = ? "
            "AND region = ? AND kind = 'static' ORDER BY doc_frequency DESC",
            [node_id, region]).fetchall()
        dynamic = con.execute(
            "SELECT line, doc_frequency, calls FROM region_lines WHERE node_id = ? "
            "AND region = ? AND kind = 'dynamic' ORDER BY doc_frequency DESC",
            [node_id, region]).fetchall()
        if not static and not dynamic:
            continue
        print(f"\n{bold}{region.upper()}{reset}  "
              f"{green}{len(static)} static{reset} / {yellow}{len(dynamic)} dynamic{reset} lines")
        for line, df, total in static[:limit]:
            print(f"  {green}[{df}/{total}]{reset} {line[:150]}")
        if len(static) > limit:
            print(f"  {dim}... {len(static) - limit} more static lines{reset}")
        for line, df, total in dynamic[:limit]:
            print(f"  {yellow}[{df}/{total}]{reset} {dim}{line[:150]}{reset}")
        if len(dynamic) > limit:
            print(f"  {dim}... {len(dynamic) - limit} more dynamic lines{reset}")

    members = con.execute(
        "SELECT request_id, ts, prompt_tokens, completion_tokens, cached_tokens, "
        "finish_reason, role_shape, dynamic_tokens FROM calls WHERE node_id = ? ORDER BY ts",
        [node_id]).fetchall()
    shown = members if limit <= 0 else members[:limit]
    print(f"\n{bold}MEMBER REQUESTS{reset}  ({len(members)} in this group; "
          f"--call N opens one, --all reads them all)")
    for i, (rid, ts, pt, ct, cached, fr, shape, dyn) in enumerate(shown):
        print(f"  [{i}] {rid:<24}{str(ts)[:19]}  {pt or 0:>7,} in / {ct or 0:>5,} out  "
              f"payload {dyn or 0:>7,}  cached {cached or 0:>6,}  {fr or '':<12}{shape}")
    if len(members) > len(shown):
        print(f"  {dim}... {len(members) - len(shown)} more (--limit 0 for all){reset}")


def show_call(con: Any, node_id: str, index: int, colour: bool) -> None:
    """One real request, with the static/dynamic verdict applied line by line."""
    from profiler.segment import lines_of

    dim, bold, yellow, green, reset = _colour(colour)
    rows = con.execute(
        "SELECT request_id, request_json, response_json FROM calls WHERE node_id = ? "
        "ORDER BY ts", [node_id]).fetchall()
    if not rows:
        print("no calls stored for that callsite")
        return
    if index >= len(rows):
        print(f"only {len(rows)} calls; index {index} is out of range")
        return
    request_id, request_json, response_json = rows[index]
    request = json.loads(request_json)
    response = json.loads(response_json)

    # The store truncates a line at 4,000 characters, so the lookup key must be truncated the
    # same way - a 30,000-character retrieved passage otherwise matched nothing and every line
    # of it came back "unknown".
    verdict = {}
    for region, kind, line in con.execute(
            "SELECT region, kind, line FROM region_lines WHERE node_id = ?",
            [node_id]).fetchall():
        verdict[(region, line[:4000])] = kind

    print("=" * 100)
    print(f"{bold}{request_id}{reset}   call {index + 1} of {len(rows)}   {dim}{node_id}{reset}")
    print(f"  {green}={reset} template, appears in most calls    "
          f"{yellow}~{reset} varies per call    {dim}?{reset} not seen in the split")
    print("=" * 100)

    for tool in request.get("tools") or []:
        fn = (tool.get("function") or {})
        print(f"  {dim}[tool]{reset} {fn.get('name')}: {str(fn.get('description'))[:90]}")
    if request.get("tools"):
        print()

    for message in request.get("messages") or []:
        role = message.get("role")
        region = ROLE_REGION.get(role, "conversation history")
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        print(f"{bold}[{role}]{reset}")
        for line in lines_of(str(content or "")):
            kind = verdict.get((region, line[:4000]))
            # Distinct ASCII markers, not colour alone: with --no-colour, or piped to a file,
            # a green bar and a yellow bar are the same character and the whole point is lost.
            if kind == "static":
                print(f"  {green}={reset} {line[:170]}")
            elif kind == "dynamic":
                print(f"  {yellow}~{reset} {dim}{line[:170]}{reset}")
            else:
                print(f"  {dim}?{reset} {line[:170]}")
        print()

    message = response.get("message") or {}
    print(f"{bold}[response]{reset} finish={response.get('finish_reason')}")
    for line in lines_of(str(message.get("content") or ""))[:20]:
        print(f"    {line[:170]}")
    for tc in message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        print(f"    {dim}-> {fn.get('name')}({str(fn.get('arguments'))[:100]}){reset}")




def _varying_lines(request, verdict):
    """The lines of one request that the profile called dynamic, in wire order."""
    from profiler.segment import lines_of

    out = []
    for message in request.get("messages") or []:
        role = message.get("role")
        region = ROLE_REGION.get(role, "conversation history")
        content = message.get("content")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content if isinstance(p, dict))
        for line in lines_of(str(content or "")):
            kind = verdict.get((region, line[:4000]))
            if kind != "static":
                out.append((role, kind, line))
    return out


def _clip(line: str, width: int, colour: str, reset: str) -> str:
    """One line, cut to the terminal, saying how much was cut rather than hiding it."""
    if len(line) <= width:
        return f"{colour}{line}{reset}" if colour else line
    body = line[:width]
    tail = f" [+{len(line) - width:,} chars]"
    return (f"{colour}{body}{reset}{tail}" if colour else f"{body}{tail}")


def show_all(con, node_id: str, colour: bool, width: int, brief: bool) -> None:
    """Read the entire group.

    The shared template is stated once at the top, then every request is printed IN FULL with
    each line marked: `=` shared by all, `~` differs, `?` not in the split. This is the view for
    deciding by eye whether these really are one callsite.

    Full text is the default rather than a diff. Showing only the differences asks the reader to
    take on trust that the rest is identical - which is precisely the claim being checked. Use
    `--brief` for the compact form once that trust is established.
    """
    dim, bold, yellow, green, reset = _colour(colour)

    verdict, template = {}, {}
    for region, kind, line in con.execute(
            "SELECT region, kind, line FROM region_lines WHERE node_id = ?",
            [node_id]).fetchall():
        verdict[(region, line[:4000])] = kind
        if kind == "static":
            template.setdefault(region, []).append(line)

    rows = con.execute(
        "SELECT request_id, ts, prompt_tokens, request_json, response_json, dynamic_tokens "
        "FROM calls WHERE node_id = ? ORDER BY ts", [node_id]).fetchall()
    if not rows:
        print("no calls stored for that callsite")
        return

    print("=" * 100)
    print(f"{bold}SHARED BY ALL {len(rows)} REQUESTS{reset}   {dim}{node_id}{reset}")
    print("=" * 100)
    for region, lines in template.items():
        print(f"\n{bold}[{region}]{reset}")
        for line in lines:
            print(f"  {green}={reset} {line[:width]}")
    if not template:
        print(f"  {yellow}nothing is shared by every request - that is itself the finding{reset}")

    from profiler.segment import lines_of

    print("\n" + "=" * 100)
    if brief:
        print(f"{bold}WHAT DIFFERS, PER REQUEST{reset}   "
              f"{dim}(everything not shown above is identical){reset}")
    else:
        print(f"{bold}EVERY REQUEST IN FULL{reset}   "
              f"{green}={reset} shared by all   {yellow}~{reset} differs   "
              f"{dim}?{reset} not in the split")
    print("=" * 100)

    for i, (rid, ts, tokens_in, request_json, response_json, dyn) in enumerate(rows):
        request = json.loads(request_json)
        response = json.loads(response_json)
        reply = str((response.get("message") or {}).get("content") or "")

        print(f"\n{bold}[{i}]{reset} {rid}  {dim}{str(ts)[:19]}  "
              f"{tokens_in or 0:,} tokens, {dyn or 0:,} of it payload{reset}")

        if brief:
            varying = _varying_lines(request, verdict)
            if not varying:
                print(f"  {dim}(identical to the template - nothing varies){reset}")
            for role, kind, line in varying[:12]:
                glyph = f"{yellow}~{reset}" if kind == "dynamic" else f"{dim}?{reset}"
                print(f"  {glyph} {dim}[{role}]{reset} {_clip(line, width, dim, reset)}")
            if len(varying) > 12:
                print(f"  {dim}... {len(varying) - 12} more varying lines{reset}")
        else:
            # The whole request, with the shared parts marked so the differences stand out
            # against them. Seeing only the differences asks the reader to trust that the rest
            # really is identical, which is the very thing being checked.
            for tool in request.get("tools") or []:
                fn = tool.get("function") or {}
                print(f"  {dim}[tool] {fn.get('name')}: "
                      f"{str(fn.get('description'))[:width - 20]}{reset}")
            for message in request.get("messages") or []:
                role = message.get("role")
                region = ROLE_REGION.get(role, "conversation history")
                content = message.get("content")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict))
                print(f"  {bold}[{role}]{reset}")
                for line in lines_of(str(content or "")):
                    kind = verdict.get((region, line[:4000]))
                    if kind == "static":
                        print(f"    {green}={reset} {dim}{_clip(line, width, '', '')}{reset}")
                    elif kind == "dynamic":
                        print(f"    {yellow}~{reset} {_clip(line, width, yellow, reset)}")
                    else:
                        print(f"    {dim}?{reset} {_clip(line, width, '', '')}")
                for tc in message.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    print(f"    {dim}-> {fn.get('name')}"
                          f"({str(fn.get('arguments'))[:80]}){reset}")
        if reply:
            print(f"  {bold}[response]{reset} {dim}{_clip(reply, width, '', '')}{reset}")

    print(f"\n{dim}{len(rows)} requests. Lines marked = are in every one of them.{reset}")


def export_group(con, node_id: str, out_dir: str) -> None:
    """Write every request in the group to its own text file, for reading or diffing."""
    os.makedirs(out_dir, exist_ok=True)
    rows = con.execute(
        "SELECT request_id, request_json, response_json FROM calls WHERE node_id = ? "
        "ORDER BY ts", [node_id]).fetchall()
    for i, (rid, request_json, response_json) in enumerate(rows):
        request = json.loads(request_json)
        path = os.path.join(out_dir, f"{i:04d}-{rid}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            for tool in request.get("tools") or []:
                fn = tool.get("function") or {}
                handle.write(f"[tool] {fn.get('name')}: {fn.get('description')}\n")
            for message in request.get("messages") or []:
                content = message.get("content")
                if isinstance(content, list):
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict))
                handle.write(f"\n[{message.get('role')}]\n{content}\n")
            handle.write(f"\n[response]\n{json.loads(response_json)}\n")
    print(f"wrote {len(rows)} requests to {out_dir}")
    print("  every file is one member of this group - diff any two to see what changes")


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--store", default=None, help="calls.duckdb (default: latest run)")
    parser.add_argument("--node", default=None, help="callsite id, or a unique prefix")
    parser.add_argument("--call", type=int, default=None, help="open member request N")
    parser.add_argument("--sql", default=None, help="run any query against the store")
    parser.add_argument("--limit", type=int, default=25, help="lines shown per region")
    parser.add_argument("--all", action="store_true",
                        help="read EVERY request in the group: the shared template once, "
                             "then what differs in each")
    parser.add_argument("--brief", action="store_true",
                        help="with --all, show only the lines that differ instead of the "
                             "whole request")
    parser.add_argument("--export", default=None, metavar="DIR",
                        help="write every request in the group to its own file")
    parser.add_argument("--width", type=int, default=150, help="characters per line")
    parser.add_argument("--no-colour", action="store_true")
    args = parser.parse_args(argv)

    store = args.store or latest_store(os.path.join(HERE, "out"))
    if not store or not os.path.exists(store):
        print("no store found - run `python run.py --csv <file>` first")
        return 1
    print(f"{DIM}{os.path.relpath(store, HERE)}{RESET}\n")

    con = duckdb.connect(store, read_only=True)
    colour = not args.no_colour and sys.stdout.isatty()

    if args.sql:
        for row in con.execute(args.sql).fetchall():
            print(row)
        return 0
    if not args.node:
        list_nodes(con)
        return 0

    node_id = resolve_node(con, args.node)
    if node_id is None:
        return 1
    if args.export:
        export_group(con, node_id, args.export)
    elif args.all:
        show_all(con, node_id, colour, args.width, args.brief)
    elif args.call is not None:
        show_call(con, node_id, args.call, colour)
    else:
        show_node(con, node_id, colour, args.limit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
