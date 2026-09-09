"""An adversarial corpus: the agent shapes an enterprise estate actually contains.

Not more volume - harder cases. The earlier fixture was generated from clean templates and then
used to show that template matching recovers templates, which is close to circular. This one is
built from the shapes that are known or suspected to BREAK the method, so that running it
produces a list of failures rather than a score to be pleased with.

Each generator returns (rows, truth) where truth maps request_id -> the callsite that really
produced it. Ambiguous-by-construction cases are marked so they are reported separately instead
of being quietly counted as errors or as passes.

    python samples/hard_corpus.py --out samples/hard.csv

Shapes covered, and why each is here:

  rag_thin            two-line system prompt, 40k of retrieved text in the user turn. The
                      instruction surface carries almost no identity.
  rag_thin_other      same generic prompt, different job. Must not merge with rag_thin.
  tool_loop           growing history, shape varies per step within ONE callsite.
  tool_router         same base prompt as tool_loop, disjoint tools. Must not merge.
  dynamic_tools       MCP-style discovery: the tool list changes every call, one callsite.
                      Must not shatter.
  cached_giant        30k static prefix, tiny dynamic tail. The cache case.
  fewshot_rotating    five examples sampled per call from a pool of twenty.
  structured_json     response_format json_schema.
  multitenant         per-tenant header injected above an identical body.
  guardrail           one-line classifier, very high volume, near-zero identity.
  delegation          a sub-agent under the same app_id.
  fragment_order      system prompt assembled from fragments in varying ORDER.
  no_system           no system message at all.
  multimodal          content as a list of parts rather than a string.
  developer_role      o-series style `developer` role instead of `system`.
  non_ascii           mixed scripts.
  no_usage            streaming rows with no usage block.
  canary              a prompt mid-rollout: 90% v1, 10% v2. AMBIGUOUS by construction.
  compacting          history summarised each step. Chain reconstruction must say "unusable".
  sliding_window      last six messages only. Same.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

T0 = datetime(2026, 6, 1, 9, 0, 0)
RNG = random.Random(20260601)

LOREM = ("The quarterly compliance review examined settlement latency across the clearing "
         "network and found that reconciliation breaks were concentrated in cross-border "
         "instruments where the counterparty reference was populated inconsistently. ")

Row = Dict[str, Any]
Truth = Dict[str, str]


def _text(n_chars: int, seed: int) -> str:
    rng = random.Random(seed)
    out = []
    while sum(len(x) for x in out) < n_chars:
        out.append(LOREM[:rng.randint(60, len(LOREM))])
    return " ".join(out)[:n_chars]


def _tool(name: str, desc: str = "") -> Dict[str, Any]:
    return {"type": "function", "function": {
        "name": name, "description": desc or f"Perform {name}",
        "parameters": {"type": "object",
                       "properties": {"q": {"type": "string"}}, "required": ["q"]}}}


class Builder:
    """Accumulates rows and their ground truth."""

    def __init__(self, app_id: str) -> None:
        self.app_id = app_id
        self.rows: List[Row] = []
        self.truth: Truth = {}
        self.ambiguous: set = set()
        self._n = 0

    def add(self, callsite: str, messages: List[Dict[str, Any]], *,
            tools: List[Dict[str, Any]] = None, response_format: Dict[str, Any] = None,
            reply: str = "ok", tool_calls: List[Dict[str, Any]] = None,
            usage: bool = True, offset_s: int = None, ambiguous: bool = False) -> str:
        self._n += 1
        rid = f"{self.app_id}-{self._n:06d}"
        request: Dict[str, Any] = {"model": "gpt-4.1", "messages": messages,
                                   "labels": {"vz_key_name": self.app_id}}
        if tools:
            request["tools"] = tools
        if response_format:
            request["response_format"] = response_format

        prompt_chars = sum(len(json.dumps(m, default=str)) for m in messages)
        prompt_chars += sum(len(json.dumps(t)) for t in (tools or []))
        message: Dict[str, Any] = {"role": "assistant", "content": reply}
        if tool_calls:
            message["tool_calls"] = tool_calls
            message["content"] = None
        response: Dict[str, Any] = {
            "id": rid, "model": "gpt-4.1",
            "choices": [{"index": 0, "message": message,
                         "finish_reason": "tool_calls" if tool_calls else "stop"}],
        }
        if usage:
            prompt_tokens = prompt_chars // 4
            # a plausible cache hit on the stable head of a long request
            cached = (prompt_tokens // 2 // 128) * 128 if prompt_tokens > 2000 else 0
            response["usage"] = {
                "prompt_tokens": prompt_tokens, "completion_tokens": len(reply) // 4 + 5,
                "total_tokens": prompt_tokens + len(reply) // 4 + 5,
                "prompt_tokens_details": {"cached_tokens": cached}}

        ts = T0 + timedelta(seconds=offset_s if offset_s is not None else self._n * 7)
        self.rows.append({
            "request_id": rid, "startTime": ts.isoformat(), "model": "gpt-4.1",
            "call_type": "acompletion",
            "request_payload": json.dumps(request, ensure_ascii=False),
            "response_payload": json.dumps(response, ensure_ascii=False),
        })
        self.truth[rid] = callsite
        if ambiguous:
            self.ambiguous.add(rid)
        return rid


# ---------------------------------------------------------------------------------------
# app_rag - the shape where the instruction surface carries almost no identity
# ---------------------------------------------------------------------------------------

def app_rag() -> Builder:
    b = Builder("app_rag")
    thin = "You are a helpful assistant."

    for i in range(60):
        b.add("rag_answer", [
            {"role": "system", "content": thin},
            {"role": "user", "content":
                f"Answer the question using only the passages below.\n"
                f"Cite passage numbers.\n"
                f"If the answer is absent, say you do not know.\n\n"
                f"PASSAGES:\n{_text(38000, i)}\n\n"
                f"QUESTION: what drove the settlement break in case {i}?"}],
            reply=f"Passage {i % 7} explains the break.")

    # same generic system prompt, entirely different job
    for i in range(45):
        b.add("rag_extract", [
            {"role": "system", "content": thin},
            {"role": "user", "content":
                f"Extract every counterparty name and LEI from the document.\n"
                f"Return one per line as NAME|LEI.\n"
                f"Do not infer values that are not present.\n\n"
                f"DOCUMENT:\n{_text(30000, 900 + i)}"}],
            reply="ACME|5493001KJTIIGC8Y1R12")

    # no system message at all, and the user turn is the whole instruction
    for i in range(30):
        b.add("rag_rerank", [
            {"role": "user", "content":
                f"Rank these passages by relevance to the query.\n"
                f"Return a JSON array of indices, best first.\n\n"
                f"QUERY: exposure limits for case {i}\n"
                f"PASSAGES:\n{_text(12000, 4000 + i)}"}],
            response_format={"type": "json_object"}, reply="[3,1,0]")
    return b


# ---------------------------------------------------------------------------------------
# app_agent - tool loops, routing, dynamic discovery
# ---------------------------------------------------------------------------------------

AGENT_PROMPT = "\n".join([
    "## Role",
    "You are an operations agent working inside the settlement platform.",
    "Work step by step and use the tools available to you.",
    "## Rules",
    "Never modify a booked trade without an approval reference.",
    "Escalate anything above the desk limit.",
    "Stop as soon as the task is complete.",
])

EXEC_TOOLS = [_tool("search_trades"), _tool("get_trade"), _tool("amend_trade"),
              _tool("post_journal"), _tool("notify_desk")]
ROUTE_TOOLS = [_tool("classify_intent"), _tool("route_to_queue"), _tool("set_priority")]


def app_agent() -> Builder:
    b = Builder("app_agent")

    # a growing tool loop: ONE callsite whose message shape changes every step
    for run in range(25):
        history = [{"role": "system", "content": AGENT_PROMPT},
                   {"role": "user", "content": f"Investigate break {run} on the EUR book."}]
        steps = RNG.randint(3, 9)
        for step in range(steps):
            last = step == steps - 1
            b.add("agent_executor", list(history), tools=EXEC_TOOLS,
                  offset_s=run * 900 + step * 20,
                  tool_calls=None if last else [{
                      "id": f"c{step}", "type": "function",
                      "function": {"name": RNG.choice(["search_trades", "get_trade",
                                                       "amend_trade"]),
                                   "arguments": json.dumps({"q": f"break {run}"})}}],
                  reply="Done." if last else "")
            history = history + [
                {"role": "assistant", "content": f"calling step {step}"},
                {"role": "tool", "content": f"tool output for step {step}: {_text(400, step)}"}]

    # same base prompt, disjoint tools, different job
    for i in range(40):
        b.add("agent_router", [
            {"role": "system", "content": AGENT_PROMPT},
            {"role": "user", "content": f"Route ticket {i}: customer reports a missing "
                                        f"confirmation."}],
            tools=ROUTE_TOOLS, offset_s=40000 + i * 11,
            tool_calls=[{"id": "c0", "type": "function",
                         "function": {"name": "classify_intent",
                                      "arguments": json.dumps({"q": f"ticket {i}"})}}])

    # MCP-style discovery: the tool list is different on every call, but it is ONE callsite
    pool = [_tool(f"mcp_{name}") for name in
            ("read_file", "write_file", "list_dir", "grep", "run_tests", "git_log",
             "git_diff", "open_pr", "fetch_url", "query_db")]
    for i in range(40):
        chosen = RNG.sample(pool, RNG.randint(3, 6))
        b.add("agent_mcp", [
            {"role": "system", "content": "\n".join([
                "You are a coding agent with access to a dynamic tool set.",
                "Discover the tools you need, then use them.",
                "Never write outside the workspace.",
                "Run the tests before you finish."])},
            {"role": "user", "content": f"Fix issue {i} in the billing module."}],
            tools=chosen, offset_s=60000 + i * 13,
            tool_calls=[{"id": "c0", "type": "function",
                         "function": {"name": chosen[0]["function"]["name"],
                                      "arguments": "{}"}}])

    # a sub-agent living under the same app id
    for i in range(30):
        b.add("agent_summariser", [
            {"role": "system", "content": "\n".join([
                "You summarise the work of another agent for an audit log.",
                "Three sentences, past tense, no speculation.",
                "State the approval reference if one was used."])},
            {"role": "user", "content": f"Transcript:\n{_text(3000, 7000 + i)}"}],
            offset_s=80000 + i * 17, reply="The agent amended two trades.")
    return b


# ---------------------------------------------------------------------------------------
# app_platform - caching, tenancy, few-shot, structured output, guardrails
# ---------------------------------------------------------------------------------------

def app_platform() -> Builder:
    b = Builder("app_platform")

    # 30k static prefix, tiny dynamic tail - the cache case done RIGHT
    policy = _text(30000, 11)
    for i in range(50):
        b.add("policy_check", [
            {"role": "system", "content": f"## Policy manual\n{policy}\n## Task\n"
                                          f"Decide whether the request is permitted."},
            {"role": "user", "content": f"Request {i}: transfer to an unverified account."}],
            offset_s=i * 30, reply="DENY")

    # the same content with the dynamic part at the FRONT - cache broken by construction
    for i in range(50):
        b.add("policy_check_bad", [
            {"role": "system", "content": f"Request id {i} · tenant {i % 9} · "
                                          f"{(T0 + timedelta(minutes=i)).isoformat()}\n"
                                          f"## Policy manual\n{policy}\n## Task\n"
                                          f"Decide whether the request is permitted."},
            {"role": "user", "content": f"Request {i}: refund above the desk limit."}],
            offset_s=20000 + i * 30, reply="ESCALATE")

    # per-tenant header above an identical body: ONE callsite
    body = "\n".join(["## Role", "You classify inbound documents.",
                      "## Classes", "invoice, contract, statement, other",
                      "Return the class name only."])
    for i in range(60):
        b.add("doc_classify", [
            {"role": "system", "content":
                f"Tenant: acme-{i % 12} | Region: {'EU' if i % 2 else 'US'} | "
                f"Date: 2026-06-{i % 28 + 1:02d}\n{body}"},
            {"role": "user", "content": f"Document:\n{_text(1500, 12000 + i)}"}],
            offset_s=40000 + i * 12, reply="invoice")

    # few-shot examples rotating per call: ONE callsite
    pool = [f"Q: example question {k}\nA: example answer {k}" for k in range(20)]
    for i in range(45):
        shots = "\n\n".join(RNG.sample(pool, 5))
        b.add("fewshot_qa", [
            {"role": "system", "content":
                f"Answer in the style of the examples.\nBe terse.\n\n{shots}"},
            {"role": "user", "content": f"Q: real question {i}"}],
            offset_s=60000 + i * 14, reply="A: an answer")

    # structured output
    schema = {"type": "json_schema", "json_schema": {
        "name": "extraction", "schema": {"type": "object",
                                         "properties": {"amount": {"type": "number"}}}}}
    for i in range(35):
        b.add("amount_extract", [
            {"role": "system", "content": "Extract the total amount. Return JSON."},
            {"role": "user", "content": f"Invoice text:\n{_text(900, 14000 + i)}"}],
            response_format=schema, offset_s=80000 + i * 9, reply='{"amount": 12.5}')

    # a one-line guardrail at very high volume - almost no identity to work with
    for i in range(200):
        b.add("guardrail", [
            {"role": "system", "content": "Is this text safe? Answer SAFE or UNSAFE."},
            {"role": "user", "content": f"user message {i}: {_text(200, 15000 + i)}"}],
            offset_s=100000 + i * 3, reply="SAFE")
    return b


# ---------------------------------------------------------------------------------------
# app_awkward - the encodings and roles that break parsers
# ---------------------------------------------------------------------------------------

def app_awkward() -> Builder:
    b = Builder("app_awkward")

    # content as a list of parts
    for i in range(30):
        b.add("multimodal", [
            {"role": "system", "content": [
                {"type": "text", "text": "You describe screenshots for an accessibility log."},
                {"type": "text", "text": "One paragraph. Mention any visible error message."}]},
            {"role": "user", "content": [
                {"type": "text", "text": f"Screenshot {i} from the settlement console."},
                {"type": "image_url", "image_url": {"url": f"https://x/img{i}.png"}}]}],
            offset_s=i * 20, reply="A dialog is shown.")

    # o-series style developer role
    for i in range(30):
        b.add("developer_role", [
            {"role": "developer", "content": "\n".join([
                "Reason carefully before answering.",
                "You are checking arithmetic in filed reports.",
                "Flag any figure that does not reconcile."])},
            {"role": "user", "content": f"Report {i}:\n{_text(1200, 16000 + i)}"}],
            offset_s=20000 + i * 20, reply="Row 4 does not reconcile.")

    # mixed scripts
    for i in range(30):
        b.add("non_ascii", [
            {"role": "system", "content": "\n".join([
                "あなたは決済業務のアシスタントです。",
                "Répondez toujours en français.",
                "Не раскрывайте персональные данные."])},
            {"role": "user", "content": f"取引 {i} の状況を教えてください。"}],
            offset_s=40000 + i * 20, reply="Le règlement est en attente.")

    # streaming rows with no usage block at all
    for i in range(25):
        b.add("no_usage", [
            {"role": "system", "content": "\n".join([
                "You stream a live commentary of the settlement queue.",
                "Emit one sentence per event.",
                "Never repeat an event id."])},
            {"role": "user", "content": f"Events: {_text(600, 17000 + i)}"}],
            usage=False, offset_s=60000 + i * 20, reply="Queue drained.")

    # system prompt assembled from fragments in a VARYING ORDER - one callsite
    fragments = ["## Role\nYou reconcile custody positions.",
                 "## Constraints\nNever fabricate a position id.",
                 "## Output\nReturn a markdown table.",
                 "## Escalation\nFlag any break above 1,000,000."]
    for i in range(40):
        order = RNG.sample(fragments, len(fragments))
        b.add("fragment_order", [
            {"role": "system", "content": "\n".join(order)},
            {"role": "user", "content": f"Positions:\n{_text(800, 18000 + i)}"}],
            offset_s=80000 + i * 15, reply="| id | break |")
    return b


# ---------------------------------------------------------------------------------------
# app_rollout - a prompt mid-canary. AMBIGUOUS by construction, reported separately.
# ---------------------------------------------------------------------------------------

def app_rollout() -> Builder:
    b = Builder("app_rollout")
    v1 = "\n".join(["## Role", "You draft customer replies about payment delays.",
                    "Apologise once.", "Offer a concrete next step.", "Sign off as the team."])
    v2 = "\n".join(["## Role", "You draft customer replies about payment delays.",
                    "Apologise once.", "Offer a concrete next step.",
                    "Include the expected settlement date.",     # the edit
                    "Sign off as the team."])
    for i in range(90):
        b.add("reply_draft", [{"role": "system", "content": v1},
                              {"role": "user", "content": f"Case {i}: payment delayed."}],
              offset_s=i * 25, reply="We are sorry.", ambiguous=True)
    for i in range(10):
        b.add("reply_draft", [{"role": "system", "content": v2},
                              {"role": "user", "content": f"Case {900 + i}: payment delayed."}],
              offset_s=50000 + i * 25, reply="We are sorry.", ambiguous=True)
    return b


# ---------------------------------------------------------------------------------------
# app_lossy - agents whose history CANNOT be walked. Chain must say so.
# ---------------------------------------------------------------------------------------

def app_lossy() -> Builder:
    b = Builder("app_lossy")
    prompt = "\n".join(["## Role", "You are a long-running research agent.",
                        "Summarise your progress whenever the context grows.",
                        "Cite every source you used."])

    # compaction: each step replaces history with a summary of it
    for run in range(12):
        for step in range(8):
            b.add("research_compacting", [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Goal: investigate topic {run}."},
                {"role": "assistant", "content":
                    f"Summary of steps 0..{step}: {_text(500, run * 100 + step)}"},
                {"role": "tool", "content": f"latest finding {step}"}],
                offset_s=run * 1200 + step * 30, reply=f"continuing {step}")

    # sliding window: only the last six messages survive
    for run in range(12):
        history = [{"role": "system", "content": prompt},
                   {"role": "user", "content": f"Goal: monitor feed {run}."}]
        for step in range(10):
            b.add("feed_windowed", history[-6:], offset_s=30000 + run * 1200 + step * 30,
                  reply=f"tick {step}")
            history = history + [{"role": "assistant", "content": f"observed {step}"},
                                 {"role": "tool", "content": f"feed row {step}"}]
    return b


BUILDERS = (app_rag, app_agent, app_platform, app_awkward, app_rollout, app_lossy)


def build() -> Tuple[List[Row], Truth, set]:
    rows: List[Row] = []
    truth: Truth = {}
    ambiguous: set = set()
    for make in BUILDERS:
        b = make()
        rows.extend(b.rows)
        truth.update(b.truth)
        ambiguous |= b.ambiguous
    rows.sort(key=lambda r: r["startTime"])
    return rows, truth, ambiguous


def main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=os.path.join(HERE, "samples", "hard.csv"))
    args = parser.parse_args(argv)

    rows, truth, ambiguous = build()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    truth_path = os.path.splitext(args.out)[0] + ".truth.json"
    with open(truth_path, "w", encoding="utf-8") as handle:
        json.dump({"callsite_of": truth, "ambiguous": sorted(ambiguous)}, handle, indent=1)

    apps = sorted({r["request_id"].rsplit("-", 1)[0] for r in rows})
    print(f"{len(rows):,} rows · {len(apps)} apps · {len(set(truth.values()))} true callsites")
    print(f"  {os.path.relpath(args.out, HERE)}")
    print(f"  {os.path.relpath(truth_path, HERE)}")
    size = os.path.getsize(args.out) / 1e6
    print(f"  {size:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
