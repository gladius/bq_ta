"""Every grouped request, written to DuckDB so the profile can be checked rather than believed.

Until now a run produced a *summary*: `report.json` listed each callsite's `request_ids` but not
the requests, so there was no way to open a group and see what was actually in it. A profile
nobody can inspect has to be taken on trust, which is the opposite of the point.

One file per run, `calls.duckdb`, holding four tables:

    calls         every request with the callsite it was assigned to, and the raw payloads
    nodes         one row per callsite: what it is, how big, how confident we are
    regions       per callsite per region - instructions, tool schemas, user turn, history -
                  mean size, share, and how much of it is static
    region_lines  **the static/dynamic answer**: every line of every region, marked static or
                  dynamic, with the document frequency behind the call

`region_lines` is the table that matters for trust. "This line appears in 60 of 60 requests, so
we called it template; that one appears in 1, so we called it payload" is a claim anyone can
check with a `SELECT`, and disagree with.

Deterministic and offline. Nothing here feeds back into grouping or metrics - it records what
was already decided.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence

import duckdb

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id VARCHAR, created_at TIMESTAMP, csv_path VARCHAR, calls INTEGER, apps INTEGER,
    callsites INTEGER, gate_passed BOOLEAN, gate_reason VARCHAR
);
CREATE TABLE IF NOT EXISTS nodes (
    node_id VARCHAR, app_id VARCHAR, name VARCHAR, purpose VARCHAR,
    calls INTEGER, avg_request_tokens INTEGER, cacheable_now INTEGER, recoverable INTEGER,
    reported_cached_mean DOUBLE, distinct_prompts INTEGER, confidence VARCHAR,
    quality VARCHAR, quality_margin DOUBLE, judge_verdict VARCHAR,
    tools_declared INTEGER, tools_never_called VARCHAR, description VARCHAR,
    template VARCHAR, findings VARCHAR, fingerprint VARCHAR
);
CREATE TABLE IF NOT EXISTS calls (
    request_id VARCHAR, app_id VARCHAR, node_id VARCHAR, ts TIMESTAMP, model VARCHAR,
    prompt_tokens INTEGER, completion_tokens INTEGER, cached_tokens INTEGER,
    finish_reason VARCHAR, n_messages INTEGER, n_tools INTEGER, role_shape VARCHAR,
    static_tokens INTEGER, dynamic_tokens INTEGER, dynamic_share DOUBLE,
    run_id_chain VARCHAR, step INTEGER,
    request_json VARCHAR, response_json VARCHAR
);
CREATE TABLE IF NOT EXISTS regions (
    node_id VARCHAR, app_id VARCHAR, region VARCHAR, mean_tokens INTEGER, share DOUBLE,
    p10 INTEGER, p90 INTEGER, static_share DOUBLE, varies BOOLEAN
);
CREATE TABLE IF NOT EXISTS region_lines (
    node_id VARCHAR, app_id VARCHAR, region VARCHAR, kind VARCHAR,
    doc_frequency INTEGER, calls INTEGER, line VARCHAR
);
"""

REGIONS = ("instructions", "tool schemas", "the user turn", "conversation history")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def write_store(result: Any, path: str, calls_by_app: Dict[str, List[Any]],
                csv_path: str = "") -> str:
    """Write one DuckDB file for this run. Returns the path."""
    from profiler.metrics import per_call_split, region_split
    from profiler.fingerprint import role_shape

    if os.path.exists(path):
        os.remove(path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    con = duckdb.connect(path)
    con.execute(SCHEMA)

    total_calls = sum(a.calls for a in result.apps)
    callsites = sum(len(a.nodes) for a in result.apps)
    con.execute(
        "INSERT INTO runs VALUES (?, now(), ?, ?, ?, ?, ?, ?)",
        [result.run_id, csv_path, total_calls, len(result.apps), callsites,
         result.gate_passed, result.gate_reason])

    node_rows: List[tuple] = []
    call_rows: List[tuple] = []
    region_rows: List[tuple] = []
    line_rows: List[tuple] = []

    for app in result.apps:
        by_id = {c.request_id: c for c in calls_by_app.get(app.app_id, [])}
        # which run each call belongs to, if the chain could be walked at all
        chain_of: Dict[str, tuple] = {}
        if app.chain and app.chain.verdict not in ("unusable",):
            for run in (app.chain.longest or []):
                pass    # longest is a summary; per-call chain ids are not retained

        for node in app.nodes:
            label = node.label or {}
            m, q = node.metrics, node.node.quality
            node_rows.append((
                node.node.node_id, app.app_id, label.get("name"), label.get("purpose"),
                m.calls, m.avg_request_tokens, m.cacheable_now, m.recoverable,
                m.reported_cached_mean, node.node.distinct_prompts, node.node.confidence,
                q.verdict, q.margin,
                node.verdict.verdict if node.verdict else None,
                m.tools_declared, _json(m.tools_never_called),
                m.anatomy.description if m.anatomy else None,
                _json(node.node.template), _json(m.findings),
                _json(node.node.signature)))

            members = [by_id[r] for r in node.node.request_ids if r in by_id]
            # how much of EACH request is payload, not just the group average
            static_by_region = {r.name: set(r.static_lines)
                                for r in (m.anatomy.regions if m.anatomy else [])}
            for call in members:
                fr = call.finish_reason
                stat, dyn = per_call_split(call, static_by_region)
                call_rows.append((
                    call.request_id, app.app_id, node.node.node_id, call.ts, call.model,
                    call.prompt_tokens, call.completion_tokens, call.cached_tokens, fr,
                    len(call.messages), len(call.tools), role_shape(call),
                    stat, dyn, round(dyn / max(stat + dyn, 1), 4),
                    chain_of.get(call.request_id, (None, None))[0],
                    chain_of.get(call.request_id, (None, None))[1],
                    _json({"messages": call.messages, "tools": call.tools,
                           "response_format": call.response_format}),
                    _json({"message": call.response_message, "finish_reason": fr,
                           "usage": call.usage})))

            if m.anatomy:
                for r in m.anatomy.regions:
                    region_rows.append((node.node.node_id, app.app_id, r.name, r.mean_tokens,
                                        r.share, r.p10, r.p90, r.static_share, r.varies))

            # the static/dynamic answer, per region, line by line - reusing the split the
            # anatomy already computed rather than repeating a document frequency
            split = {r.name: (r.static_lines, r.dynamic_lines)
                     for r in (m.anatomy.regions if m.anatomy else [])}
            for region in REGIONS:
                if not members:
                    continue
                static, dynamic = split.get(region, ([], []))
                if not static and not dynamic:
                    static, dynamic, _ = region_split(members, region)
                if not static and not dynamic:
                    continue
                counts = _line_counts(members, region)
                for line in static:
                    line_rows.append((node.node.node_id, app.app_id, region, "static",
                                      counts.get(line, 0), len(members), line[:4000]))
                for line in dynamic[:400]:
                    line_rows.append((node.node.node_id, app.app_id, region, "dynamic",
                                      counts.get(line, 0), len(members), line[:4000]))

        # calls that were never assigned to a callsite still belong in the store
        assigned = {r for n in app.nodes for r in n.node.request_ids}
        for call in calls_by_app.get(app.app_id, []):
            if call.request_id in assigned:
                continue
            call_rows.append((
                call.request_id, app.app_id, None, call.ts, call.model, call.prompt_tokens,
                call.completion_tokens, call.cached_tokens, call.finish_reason,
                len(call.messages), len(call.tools), role_shape(call),
                None, None, None, None, None,
                _json({"messages": call.messages, "tools": call.tools,
                       "response_format": call.response_format}),
                _json({"message": call.response_message, "usage": call.usage})))

    _insert(con, "nodes", node_rows)
    _insert(con, "calls", call_rows)
    _insert(con, "regions", region_rows)
    _insert(con, "region_lines", line_rows)
    con.close()
    return path


def _line_counts(calls: Sequence[Any], region: str) -> Dict[str, int]:
    from collections import Counter

    from profiler.metrics import region_text
    from profiler.segment import lines_of

    counts: Counter = Counter()
    for call in calls:
        counts.update(set(lines_of(region_text(call, region))))
    return dict(counts)


def _insert(con: "duckdb.DuckDBPyConnection", table: str, rows: List[tuple]) -> None:
    """Bulk insert. executemany row by row dominated the run at 25 MB/s in an earlier build."""
    if not rows:
        return
    import pandas as pd

    columns = [d[0] for d in con.execute(f"SELECT * FROM {table} LIMIT 0").description]
    frame = pd.DataFrame(rows, columns=columns)
    con.register("_incoming", frame)
    con.execute(f"INSERT INTO {table} SELECT * FROM _incoming")
    con.unregister("_incoming")
