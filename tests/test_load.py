"""Loader: JSON paths, malformed rows, and the things a real export gets wrong."""

from __future__ import annotations

import csv
import json
import os

import pytest


from profiler.config import Mapping
from profiler.load import load_calls, resolve_app_id

MAPPING = Mapping(
    request_id="request_id", timestamp="startTime", model="model",
    request="request_payload", response="response_payload", call_type="call_type",
    app_id_paths=["request_payload.labels.vz_key_name", "client_metadata.headers.x-subject"],
    chat_call_types=["acompletion"],
)


def chat_request(app: str = "agent-a", system: str = "You are helpful.") -> str:
    return json.dumps({
        "model": "gpt-4o-mini",
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": "hello"}],
        "labels": {"vz_key_name": app},
    })


def chat_response(cached: int = 0) -> str:
    return json.dumps({
        "choices": [{"message": {"role": "assistant", "content": "hi"},
                     "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5,
                  "prompt_tokens_details": {"cached_tokens": cached}},
    })


def write_csv(tmp_path, rows):
    path = os.path.join(str(tmp_path), "in.csv")
    fields = ["request_id", "startTime", "model", "request_payload", "response_payload",
              "client_metadata", "call_type"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fields})
    return path


def base_row(i: int, **over):
    row = {
        "request_id": f"r{i}", "startTime": "2026-03-02 00:00:00", "model": "gpt-4o-mini",
        "request_payload": chat_request(), "response_payload": chat_response(),
        "client_metadata": json.dumps({"headers": {"x-subject": "agent-a"}}),
        "call_type": "acompletion",
    }
    row.update(over)
    return row


def test_loads_valid_rows(tmp_path):
    path = write_csv(tmp_path, [base_row(i) for i in range(5)])
    calls, report = load_calls(path, MAPPING)
    assert report.loaded == 5
    assert calls[0].app_id == "agent-a"
    assert calls[0].cached_tokens == 0
    assert calls[0].prompt_tokens == 100


def test_malformed_json_is_skipped_not_fatal(tmp_path):
    rows = [base_row(0), base_row(1, request_payload='{"messages": [trunc'), base_row(2)]
    path = write_csv(tmp_path, rows)
    calls, report = load_calls(path, MAPPING)
    assert report.loaded == 2
    assert report.skipped_unparseable == 1


def test_non_chat_rows_are_dropped(tmp_path):
    rows = [base_row(0), base_row(1, call_type="aembedding")]
    path = write_csv(tmp_path, rows)
    calls, report = load_calls(path, MAPPING)
    assert report.loaded == 1
    assert report.skipped_non_chat == 1


def test_missing_usage_is_counted_and_survives(tmp_path):
    no_usage = json.dumps({"choices": [{"message": {"content": "hi"},
                                        "finish_reason": "stop"}]})
    path = write_csv(tmp_path, [base_row(0, response_payload=no_usage)])
    calls, report = load_calls(path, MAPPING)
    assert report.loaded == 1
    assert report.missing_usage == 1
    assert calls[0].cached_tokens is None       # not zero: unknown is not the same as none


def test_app_id_falls_back_to_second_path():
    request = {"messages": [], "labels": {}}
    metadata = {"headers": {"x-subject": "from-header"}}
    app_id, source, disagreed = resolve_app_id(
        {"client_metadata": json.dumps(metadata)}, {"request_payload": request},
        MAPPING.app_id_paths)
    assert app_id == "from-header"
    assert source.endswith("x-subject")
    assert not disagreed


def test_app_id_disagreement_is_reported_not_hidden():
    request = {"labels": {"vz_key_name": "one"}}
    metadata = {"headers": {"x-subject": "two"}}
    app_id, _, disagreed = resolve_app_id(
        {"client_metadata": json.dumps(metadata)}, {"request_payload": request},
        MAPPING.app_id_paths)
    assert app_id == "one"          # first path wins
    assert disagreed is True        # but the conflict is surfaced


def test_rows_without_an_app_id_are_skipped(tmp_path):
    anonymous = json.dumps({"messages": [{"role": "user", "content": "x"}]})
    path = write_csv(tmp_path, [base_row(0, request_payload=anonymous, client_metadata="{}")])
    calls, report = load_calls(path, MAPPING)
    assert report.loaded == 0
    assert report.skipped_no_app_id == 1


def test_per_app_cap_is_applied(tmp_path):
    mapping = Mapping(**{**MAPPING.__dict__, "max_rows_per_app": 3})
    path = write_csv(tmp_path, [base_row(i) for i in range(10)])
    calls, report = load_calls(path, mapping)
    assert report.loaded == 3
    assert "agent-a" in report.capped_apps
