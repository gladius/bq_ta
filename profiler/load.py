"""CSV export -> normalised calls.

DuckDB reads the CSV (chunked, so files larger than memory are fine) and extracts the JSON
paths; Python does the per-row validation. Rows that cannot be parsed are counted and skipped,
never fatal - a production export always contains some.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterator, List, Optional, Tuple

import duckdb
import xxhash

from profiler.config import Mapping


@dataclass
class Call:
    """One chat completion, normalised. Everything downstream reads this, not the CSV."""

    request_id: str
    app_id: str
    ts: datetime
    model: str
    messages: List[Dict[str, Any]]
    tools: List[Dict[str, Any]]
    response_message: Dict[str, Any]
    finish_reason: Optional[str]
    usage: Dict[str, Any]
    response_format: Optional[Dict[str, Any]] = None

    @property
    def prompt_tokens(self) -> Optional[int]:
        return self.usage.get("prompt_tokens")

    @property
    def cached_tokens(self) -> Optional[int]:
        details = self.usage.get("prompt_tokens_details") or {}
        value = details.get("cached_tokens")
        return value if isinstance(value, int) else None

    @property
    def completion_tokens(self) -> Optional[int]:
        return self.usage.get("completion_tokens")


@dataclass
class LoadReport:
    """What the loader saw. Surfaced in the report so silent data loss is impossible."""

    total_rows: int = 0
    loaded: int = 0
    skipped_unparseable: int = 0
    skipped_non_chat: int = 0
    skipped_no_messages: int = 0
    skipped_no_app_id: int = 0
    missing_usage: int = 0
    synthesised_request_ids: int = 0
    missing_columns: List[str] = field(default_factory=list)
    app_id_disagreements: int = 0
    app_id_source_counts: Dict[str, int] = field(default_factory=dict)
    per_app_counts: Dict[str, int] = field(default_factory=dict)
    capped_apps: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "loaded": self.loaded,
            "skipped": {
                "unparseable_json": self.skipped_unparseable,
                "non_chat": self.skipped_non_chat,
                "no_messages": self.skipped_no_messages,
                "no_app_id": self.skipped_no_app_id,
            },
            "missing_usage": self.missing_usage,
            "synthesised_request_ids": self.synthesised_request_ids,
            "missing_columns": self.missing_columns,
            "app_id_disagreements": self.app_id_disagreements,
            "app_id_source_counts": self.app_id_source_counts,
            "apps": len(self.per_app_counts),
            "per_app_counts": dict(sorted(self.per_app_counts.items(),
                                          key=lambda kv: -kv[1])),
            "capped_apps": self.capped_apps,
        }


def _dig(obj: Any, path: List[str]) -> Any:
    for key in path:
        if not isinstance(obj, dict):
            return None
        obj = obj.get(key)
    return obj


def _parse_json(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def resolve_app_id(
    row: Dict[str, Any], parsed: Dict[str, Optional[Dict[str, Any]]], paths: List[str]
) -> Tuple[Optional[str], Optional[str], bool]:
    """First non-null path wins. Returns (app_id, which_path, disagreed).

    Disagreement between two paths that both resolve is reported rather than silently
    preferred: the app id is the outer partition, so a high rate would invalidate everything.
    """
    found: List[Tuple[str, str]] = []
    for path in paths:
        head, *rest = path.split(".")
        source = parsed.get(head)
        if source is None:
            source = _parse_json(row.get(head))
        value = _dig(source, rest) if source is not None else None
        if isinstance(value, str) and value.strip():
            found.append((path, value.strip()))
    if not found:
        return None, None, False
    disagreed = len({v for _, v in found}) > 1
    return found[0][1], found[0][0], disagreed


def _extract_tools(request: Dict[str, Any]) -> List[Dict[str, Any]]:
    tools = request.get("tools")
    return [t for t in tools if isinstance(t, dict)] if isinstance(tools, list) else []


def load_calls(
    csv_path: str, mapping: Mapping, con: Optional[duckdb.DuckDBPyConnection] = None,
    chunk: int = 2000,
) -> Tuple[List[Call], LoadReport]:
    """Read the export and return validated chat calls plus a report of what was dropped."""
    owns = con is None
    con = con or duckdb.connect()
    report = LoadReport()

    # Select only the columns this export actually has. Exports differ between environments,
    # and an app-id path pointing at a column that is not present must degrade to "try the
    # next path", not abort the run before a single row is read.
    absolute = os.path.abspath(csv_path)
    available = {
        d[0] for d in con.execute(
            "SELECT * FROM read_csv_auto(?, header=true, sample_size=1000, "
            "ignore_errors=true, all_varchar=true) LIMIT 0", [absolute]).description
    }
    wanted = mapping.source_columns
    present = [c for c in wanted if c in available]
    report.missing_columns = sorted(set(wanted) - available)

    required = [mapping.request, mapping.response]
    missing_required = [c for c in required if c not in available]
    if missing_required:
        raise ValueError(
            f"{csv_path} has no column(s) {missing_required}. Found: {sorted(available)}. "
            f"Point mapping.yaml at the right columns.")

    columns = ", ".join(f'"{c}"' for c in present)
    cursor = con.execute(
        f"SELECT {columns} FROM read_csv_auto(?, header=true, sample_size=-1, "
        f"ignore_errors=true, all_varchar=true)",
        [absolute],
    )
    names = [d[0] for d in cursor.description]

    calls: List[Call] = []
    per_app: Dict[str, List[Call]] = {}

    while True:
        batch = cursor.fetchmany(chunk)
        if not batch:
            break
        for raw in batch:
            report.total_rows += 1
            row = dict(zip(names, raw))

            if mapping.call_type and mapping.chat_call_types:
                call_type = (row.get(mapping.call_type) or "").strip()
                if call_type and call_type not in mapping.chat_call_types:
                    report.skipped_non_chat += 1
                    continue

            request = _parse_json(row.get(mapping.request))
            response = _parse_json(row.get(mapping.response))
            if request is None:
                report.skipped_unparseable += 1
                continue

            parsed_sources = {mapping.request: request, mapping.response: response}
            app_id, source, disagreed = resolve_app_id(row, parsed_sources,
                                                       mapping.app_id_paths)
            if disagreed:
                report.app_id_disagreements += 1
            if not app_id:
                report.skipped_no_app_id += 1
                continue
            if source:
                report.app_id_source_counts[source] = \
                    report.app_id_source_counts.get(source, 0) + 1

            messages = request.get("messages")
            if not isinstance(messages, list) or not messages:
                report.skipped_no_messages += 1
                continue

            choice = ((response or {}).get("choices") or [{}])[0] or {}
            usage = (response or {}).get("usage") or {}
            if not usage:
                report.missing_usage += 1

            ts = row.get(mapping.timestamp)
            try:
                stamp = datetime.fromisoformat(str(ts).replace("Z", "+00:00").strip())
            except (ValueError, TypeError):
                stamp = datetime.min

            # A request id is not guaranteed. The old fallback was the row counter, which is
            # POSITIONAL: re-export the same window in a different order and every call is
            # renamed. The store, the registry and node membership all key on this, so the
            # fallback has to be derived from the row's own content instead - stable across
            # exports, orderings and re-runs.
            given = str(row.get(mapping.request_id) or "").strip()
            if not given:
                report.synthesised_request_ids += 1
                given = "synth-" + xxhash.xxh64(
                    f"{app_id}|{ts}|{row.get(mapping.request)}".encode()).hexdigest()[:16]

            call = Call(
                request_id=given,
                app_id=app_id,
                ts=stamp.replace(tzinfo=None),
                model=str(row.get(mapping.model) or request.get("model") or "unknown"),
                messages=[m for m in messages if isinstance(m, dict)],
                tools=_extract_tools(request),
                response_message=(choice.get("message") or {}),
                finish_reason=choice.get("finish_reason"),
                usage=usage if isinstance(usage, dict) else {},
                response_format=request.get("response_format")
                if isinstance(request.get("response_format"), dict) else None,
            )
            per_app.setdefault(app_id, []).append(call)

    for app_id, app_calls in per_app.items():
        app_calls.sort(key=lambda c: (c.ts, c.request_id))
        if len(app_calls) > mapping.max_rows_per_app:
            report.capped_apps.append(app_id)
            app_calls = app_calls[: mapping.max_rows_per_app]
        report.per_app_counts[app_id] = len(app_calls)
        calls.extend(app_calls)

    report.loaded = len(calls)
    if owns:
        con.close()
    return calls, report


def group_by_app(calls: List[Call]) -> Dict[str, List[Call]]:
    out: Dict[str, List[Call]] = {}
    for call in calls:
        out.setdefault(call.app_id, []).append(call)
    return out
