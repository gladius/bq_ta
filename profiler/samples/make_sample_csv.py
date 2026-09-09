"""Export the spike corpus into a CSV shaped like a litellm/BigQuery export.

The spike database holds 13,564 real and synthetic gateway calls whose true callsites we know.
Re-shaping them as a litellm export gives a development fixture with ground truth, so the
production pipeline can be proved to reproduce the spike's numbers before it ever sees real data.

Deliberately included, because real exports have them:
  - request/response stored as JSON *strings*, not structs
  - the agent id in two places that usually agree
  - a few malformed / truncated JSON rows
  - a few non-chat rows (embeddings) that must be filtered out
  - a few rows with no `usage` block, as streamed responses often are
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from typing import Any, Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
SPIKE_DB = os.path.join(HERE, "..", "..", "spike", "data", "sim.duckdb")

COLUMNS = [
    "request_id", "startTime", "endTime", "model", "request_payload", "response_payload",
    "client_metadata", "spend", "total_tokens", "call_type",
]


def build(out_path: str, per_app: int, seed: int, corrupt: int) -> Dict[str, Any]:
    import duckdb

    rng = random.Random(seed)
    con = duckdb.connect(os.path.abspath(SPIKE_DB), read_only=True)

    # a stratified sample: every app, capped, so the file stays a realistic size
    rows = con.execute(
        """
        WITH ranked AS (
          SELECT request_id, ts, app_id, model, request_json, response_json, latency_ms,
                 gt_callsite,
                 row_number() OVER (PARTITION BY app_id ORDER BY ts) AS rn
          FROM gateway_log
        )
        SELECT request_id, ts, app_id, model, request_json, response_json, latency_ms, gt_callsite
        FROM ranked WHERE rn <= ? ORDER BY ts
        """,
        [per_app],
    ).fetchall()
    con.close()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    truth_path = os.path.splitext(out_path)[0] + "_truth.csv"

    stats: Dict[str, Any] = {"rows": 0, "apps": set(), "corrupted": 0, "no_usage": 0,
                             "non_chat": 0, "id_disagree": 0}

    with open(out_path, "w", encoding="utf-8", newline="") as fh, \
            open(truth_path, "w", encoding="utf-8", newline="") as tfh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        truth = csv.DictWriter(tfh, fieldnames=["request_id", "app_id", "gt_callsite"])
        truth.writeheader()

        for i, (rid, ts, app_id, model, req_json, resp_json, latency, callsite) in enumerate(rows):
            request = json.loads(req_json)
            response = json.loads(resp_json)

            # the agent id lives in the request labels, and again in a client metadata header
            request["labels"] = {"vz_key_name": app_id, "environment": "prod"}
            header_id = app_id
            if rng.random() < 0.002:          # rare disagreement, as happens in practice
                header_id = app_id + "-alt"
                stats["id_disagree"] += 1
            client_metadata = {"headers": {"x-subject": header_id,
                                           "user-agent": "litellm/1.0"},
                               "team_id": "team-" + app_id.split("_")[-1]}

            call_type = "acompletion"
            if rng.random() < 0.004:          # embedding rows that must be filtered out
                call_type = "aembedding"
                request = {"model": model, "input": "some text", "labels": request["labels"]}
                response = {"object": "list", "data": [{"embedding": [0.0, 0.1]}]}
                stats["non_chat"] += 1
            elif rng.random() < 0.01:         # streamed rows often carry no usage block
                response = dict(response)
                response.pop("usage", None)
                stats["no_usage"] += 1

            req_text = json.dumps(request, ensure_ascii=False)
            resp_text = json.dumps(response, ensure_ascii=False)

            if stats["corrupted"] < corrupt and rng.random() < 0.01:
                req_text = req_text[: max(20, len(req_text) // 2)]   # truncated JSON
                stats["corrupted"] += 1

            usage = response.get("usage") or {}
            writer.writerow({
                "request_id": rid,
                "startTime": ts.isoformat(sep=" "),
                "endTime": ts.isoformat(sep=" "),
                "model": model,
                "request_payload": req_text,
                "response_payload": resp_text,
                "client_metadata": json.dumps(client_metadata, ensure_ascii=False),
                "spend": round(rng.random() * 0.01, 6),
                "total_tokens": usage.get("total_tokens", ""),
                "call_type": call_type,
            })
            truth.writerow({"request_id": rid, "app_id": app_id, "gt_callsite": callsite})
            stats["rows"] += 1
            stats["apps"].add(app_id)

    stats["apps"] = len(stats["apps"])
    stats["path"] = out_path
    stats["truth_path"] = truth_path
    stats["size_mb"] = round(os.path.getsize(out_path) / 1e6, 1)
    return stats


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(HERE, "sample.csv"))
    parser.add_argument("--per-app", type=int, default=300,
                        help="cap per app; keeps the fixture a realistic export size")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--corrupt", type=int, default=8, help="malformed JSON rows to inject")
    args = parser.parse_args(argv)

    if not os.path.exists(os.path.abspath(SPIKE_DB)):
        print(f"spike database not found at {SPIKE_DB}; run the spike first", file=sys.stderr)
        return 1

    stats = build(args.out, args.per_app, args.seed, args.corrupt)
    print(json.dumps(stats, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
