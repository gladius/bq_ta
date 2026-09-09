"""Run reconstruction: it must work where history is replayed, and SAY SO where it is not.

The second half matters more. A confident run tree built from an agent that compacts its
history is a fabrication, and the spike measured exactly how badly prefix walking fails there
(0.344 compacting, 0.054 windowing). So the tests below check the alarm as carefully as the
happy path.
"""

from __future__ import annotations

from datetime import datetime, timedelta


from profiler.chain import reconstruct
from profiler.load import Call

T0 = datetime(2026, 3, 1, 12, 0, 0)


def call(rid, messages, offset_s=0):
    return Call(request_id=rid, app_id="app", ts=T0 + timedelta(seconds=offset_s),
                model="gpt-4.1", messages=messages, tools=[],
                response_message={"role": "assistant", "content": "ok"},
                finish_reason="stop", usage={"prompt_tokens": 100, "completion_tokens": 10})


def loop(run, steps, start):
    """One agent run that replays its whole history each step."""
    messages = [{"role": "system", "content": "You are an agent."},
                {"role": "user", "content": f"task {run}"}]
    calls = []
    for step in range(steps):
        calls.append(call(f"{run}-{step}", list(messages), start + step * 10))
        messages = messages + [
            {"role": "assistant", "content": f"step {step}"},
            {"role": "tool", "content": f"result {step}"},
        ]
    return calls


def test_a_replayed_loop_is_reconstructed_as_one_run():
    report = reconstruct(loop("r1", 4, 0))
    assert report.runs == 1
    assert report.max_steps == 4
    assert report.verdict == "reliable"
    assert report.unlinked_calls == 1          # only the opening call has no parent


def test_separate_runs_do_not_merge():
    report = reconstruct(loop("r1", 3, 0) + loop("r2", 3, 500))
    assert report.runs == 2
    assert report.mean_steps == 3.0


def test_single_step_traffic_is_named_not_faked():
    """Every call a first call: there are no runs to find, and it must say that."""
    calls = [call(f"c{i}", [{"role": "system", "content": "S"},
                            {"role": "user", "content": f"q{i}"}], i * 10)
             for i in range(10)]
    report = reconstruct(calls)
    assert report.verdict == "single_step"
    assert report.runs == 10


def test_a_compacting_agent_is_reported_as_unusable():
    """The failure that cannot be engineered away - it must be declared, not papered over."""
    calls = []
    for i in range(10):
        # each step summarises what came before, so no prefix survives to link against
        calls.append(call(f"c{i}", [
            {"role": "system", "content": "You are an agent."},
            {"role": "user", "content": f"summary of steps 0..{i}"},
            {"role": "assistant", "content": f"prior work {i}"},
            {"role": "tool", "content": f"compacted {i}"},
        ], i * 10))
    report = reconstruct(calls)
    assert report.verdict == "unusable"
    assert "cannot be recovered from logs" in report.note
    assert report.orphan_continuations >= 8


def test_a_retry_is_not_counted_as_a_step():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "q"}]
    calls = [call("a", list(messages), 0), call("b", list(messages), 2)]
    report = reconstruct(calls)
    assert report.retries == 1
    assert report.runs == 1


def test_the_node_path_names_the_callsites_a_run_visited():
    calls = loop("r1", 3, 0)
    node_of = {"r1-0": "planner", "r1-1": "executor", "r1-2": "executor"}
    report = reconstruct(calls, node_of)
    assert report.top_paths[0]["path"] == "planner -> executor -> executor"


def test_tokens_are_summed_per_run_not_per_call():
    report = reconstruct(loop("r1", 4, 0))
    assert report.longest[0]["prompt_tokens"] == 400
    assert report.mean_run_tokens == 440
