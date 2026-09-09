"""Agent Profiler - CSV in, a profile of every agent out.

    python run.py --csv samples/hard.csv
    python run.py --csv samples/hard.csv --no-llm       # deterministic only
    python run.py --csv samples/hard.csv --no-registry  # do not remember this run

Profiling only: what callsites each agent has, what each is for, what varies per call, what is
cacheable, and how the calls chain into runs. It proposes no changes.

Each run writes one directory:

    out/run-<id>/index.html      the estate, ranked by tokens   <- start here
    out/run-<id>/<agent>.html    one page per agent
    out/run-<id>/report.md       the same in prose
    out/run-<id>/report.json     everything, for pipelines
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from profiler.audit import Audit
from profiler.config import load_settings
from profiler.pipeline import run_pipeline
from profiler.report import write_json, write_markdown
from profiler.report_html import write_site
from profiler.store import write_store


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, help="litellm/BigQuery CSV export")
    parser.add_argument("--mapping", default=None, help="column mapping YAML")
    parser.add_argument("--no-llm", action="store_true", help="deterministic stages only")
    parser.add_argument("--out", default=None, help="output directory")
    parser.add_argument("--tau", type=float, default=None)
    parser.add_argument("--max-rows-per-app", type=int, default=None)
    parser.add_argument("--no-registry", action="store_true",
                        help="do not record these callsites for future runs")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings(args.mapping, use_llm=not args.no_llm, tau=args.tau,
                             out_dir=args.out)
    if args.max_rows_per_app:
        settings.mapping.max_rows_per_app = args.max_rows_per_app
    if args.no_registry:
        settings.registry_path = ""

    if not settings.llm_enabled and not args.no_llm:
        print("  note: no LLM_KEY found - running deterministic stages only")

    audit = Audit(settings.audit_dir)
    print(f"[profiler] run {audit.run_id}")
    started = time.time()

    result = run_pipeline(args.csv, settings, audit, verbose=not args.quiet)

    run_dir = os.path.join(settings.out_dir, f"run-{audit.run_id}")
    os.makedirs(run_dir, exist_ok=True)
    write_json(result, os.path.join(run_dir, "report.json"))
    write_markdown(result, os.path.join(run_dir, "report.md"))
    index = write_site(result, run_dir)
    store = write_store(result, os.path.join(run_dir, "calls.duckdb"),
                        result.calls_by_app, args.csv)
    audit_md = os.path.join(settings.audit_dir, f"audit-{audit.run_id}.md")
    audit.write_markdown(audit_md, result.load_report)

    nodes = sum(len(a.nodes) for a in result.apps)
    print()
    print(f"  apps {len(result.apps)} · callsites {nodes} · "
          f"LLM calls {result.llm_usage['calls']} (${result.llm_usage['cost_usd']})")
    # "PASSED - unverified" is a contradiction: with no LLM nothing checked anything, so the
    # run is neither passed nor failed.
    state = ("UNVERIFIED" if not result.llm_usage["calls"]
             else "PASSED" if result.gate_passed else "NOT PASSED")
    print(f"  verification: {state} - {result.gate_reason}")
    print(f"  open    {os.path.relpath(index, HERE)}")
    print(f"          + {len(result.apps)} agent pages, report.md, report.json in that folder")
    print(f"  store   {os.path.relpath(store, HERE)}")
    print("  inspect python inspect_calls.py                       # list callsites")
    print("          python inspect_calls.py --node <id>           # static vs dynamic")
    print("          python inspect_calls.py --node <id> --call 0  # read one request")
    print(f"  audit   {os.path.relpath(audit_md, HERE)}")
    print(f"done in {time.time() - started:.1f}s")
    return 0 if result.gate_passed else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
