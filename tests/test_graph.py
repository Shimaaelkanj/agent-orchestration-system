from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import orchestrator.graph as g
import orchestrator.memory.service as svc
from orchestrator import config
from orchestrator.hitl import runner
from orchestrator.hitl.queue import ReviewQueue
from orchestrator.memory.embeddings import HashEmbedder
from orchestrator.memory.long_term import LongTermMemory
from orchestrator.memory.working import WorkingMemory, _MemoryBackend
from orchestrator.schemas import (Complexity, ExecutionPlan, ReviewVerdict, Specialist,
                                  SpecialistReport, Subtask)
from orchestrator.tools import ApprovalRequired


def st(i, deps):
    return Subtask(id=i, description=f"do {i}", specialist=Specialist.RESEARCH, depends_on=deps,
                   required_inputs=[], expected_output="text", complexity=Complexity.LOW)


def mk_plan(conf=0.9, ids=("t1", "t2")):
    subs = [st("t1", [])] + ([st("t2", ["t1"])] if "t2" in ids else [])
    return ExecutionPlan(rationale="r", confidence=conf, subtasks=subs)


class FakeRegistry:
    def __init__(self, calls):
        self.calls = calls

    def invoke(self, tool, agent, args, approved=False):
        self.calls["tools"].append((tool, agent, args, approved))
        return '{"deleted": true}'


@pytest.fixture
def env(monkeypatch, tmp_path):
    calls = {"specialist": [], "review": [], "plan_ctx": [], "tools": [], "notified": []}
    knobs = {"plan": mk_plan(), "conf": 0.9,
             "approve": lambda sid, n: True,        # n = how many times this subtask was reviewed
             "crash": lambda sid, n: False,         # n = how many times this subtask was attempted
             "sensitive": lambda sid, n: False}

    def fake_specialist(subtask, deps, feedback, registry):
        calls["specialist"].append((subtask.id, dict(deps), feedback))
        n = sum(1 for c in calls["specialist"] if c[0] == subtask.id)
        if knobs["sensitive"](subtask.id, n):
            raise ApprovalRequired("delete_file", "research", {"path": "old.txt"})
        if knobs["crash"](subtask.id, n):
            raise RuntimeError("boom")
        return SpecialistReport(answer=f"out-{subtask.id}", confidence=knobs["conf"]), "evidence"

    def fake_review(subtask, report, evidence=""):
        calls["review"].append(subtask.id)
        ok = knobs["approve"](subtask.id, calls["review"].count(subtask.id))
        return ReviewVerdict(approved=ok, feedback="" if ok else "fix it", confidence=0.9)

    def fake_plan(task, err=None, memory_context="", human_feedback=""):
        calls["plan_ctx"].append((memory_context, human_feedback))
        return knobs["plan"]

    ltm = LongTermMemory(HashEmbedder(), tmp_path / "chroma")
    wm = WorkingMemory(_MemoryBackend())
    queue = ReviewQueue(tmp_path / "q.db")

    monkeypatch.setattr(config, "MEMORY_MIN_SIMILARITY", 0.05)
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path)
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.6)
    monkeypatch.setattr(config, "NOTIFY_BELOW", 0.8)
    monkeypatch.setattr(config, "PLAN_CONFIDENCE_THRESHOLD", 0.5)
    monkeypatch.setattr(g, "get_ltm", lambda: ltm)
    monkeypatch.setattr(g, "get_wm", lambda: wm)
    monkeypatch.setattr(g, "get_queue", lambda: queue)
    monkeypatch.setattr(g, "notify", lambda item: calls["notified"].append(item["review_id"]))
    monkeypatch.setattr(g, "registry", FakeRegistry(calls))
    monkeypatch.setattr(svc, "llm_extract", lambda s: None)          # no LLM during tests
    monkeypatch.setattr(g, "create_plan", fake_plan)
    monkeypatch.setattr(g, "run_specialist", fake_specialist)
    monkeypatch.setattr(g, "review_output", fake_review)
    monkeypatch.setattr(g, "synthesize", lambda task, outputs: "FINAL:" + ",".join(sorted(outputs)))
    return SimpleNamespace(calls=calls, knobs=knobs, ltm=ltm, wm=wm, queue=queue)


@pytest.fixture
def graph(env):
    return g.build_graph(MemorySaver())


def begin(graph, review=False, tid="task1", task="find prime numbers"):
    cfg = {"configurable": {"thread_id": tid}}
    graph.invoke({"task": task, "user_id": "alice", "task_id": tid, "review_requested": review}, cfg)
    return cfg


def state_of(graph, cfg):
    return runner._outcome(graph, cfg["configurable"]["thread_id"])


def answer(graph, cfg, decision):
    graph.invoke(Command(resume=decision), cfg)
    return state_of(graph, cfg)


def t1_calls(env):
    return [c for c in env.calls["specialist"] if c[0] == "t1"]


# ---------------- normal flow ----------------
def test_happy_path_needs_no_human(env, graph):
    out = state_of(graph, begin(graph))
    assert out["state"] == "done"
    assert out["values"]["final_answer"] == "FINAL:t1,t2"
    assert env.calls["specialist"][1][1] == {"t1": "out-t1"}      # t2 received t1's output
    assert env.queue.list_reviews() == []


def test_reject_then_retry_with_feedback(env, graph):
    env.knobs["approve"] = lambda sid, n: not (sid == "t1" and n == 1)
    out = state_of(graph, begin(graph))
    assert out["state"] == "done"
    assert out["values"]["results"]["t1"]["attempts"] == 2
    assert t1_calls(env)[1][2] == "fix it"                         # reviewer feedback reached the retry


def test_crash_once_then_recover(env, graph):
    env.knobs["crash"] = lambda sid, n: sid == "t1" and n == 1
    out = state_of(graph, begin(graph))
    assert out["state"] == "done"
    assert "different approach" in t1_calls(env)[1][2]


def test_invalid_plan_three_times_escalates_when_the_fallback_is_off(env, graph, monkeypatch):
    monkeypatch.setenv("PLAN_FALLBACK", "0")
    env.knobs["plan"] = ExecutionPlan(rationale="r", confidence=0.9, subtasks=[st("t1", []), st("t1", [])])
    out = state_of(graph, begin(graph))
    assert out["state"] == "needs_human"
    assert "valid plan" in out["values"]["escalation_reason"]


def test_invalid_plan_three_times_falls_back_to_a_plan_a_human_approves(env, graph, monkeypatch):
    monkeypatch.delenv("PLAN_FALLBACK", raising=False)
    monkeypatch.delenv("PLAN_FALLBACK_CONFIDENCE", raising=False)
    env.knobs["plan"] = ExecutionPlan(rationale="r", confidence=0.9, subtasks=[st("t1", []), st("t1", [])])
    cfg = begin(graph)
    out = state_of(graph, cfg)
    assert out["state"] == "waiting"                              # paused for a human, nothing has run yet
    item = env.queue.get(out["review_id"])
    assert (item["level"], item["trigger"]) == ("approve_plan", "low_plan_confidence")
    assert len(item["payload"]["plan"]["subtasks"]) == 1          # the single-subtask fallback plan
    assert env.calls["specialist"] == []
    assert answer(graph, cfg, {"action": "approve"})["state"] == "done"


# ---------------- Approve plan ----------------
def test_low_plan_confidence_pauses_before_any_work(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    out = state_of(graph, cfg)
    assert out["state"] == "waiting" and env.calls["specialist"] == []
    item = env.queue.get(out["review_id"])
    assert (item["level"], item["trigger"], item["status"]) == ("approve_plan", "low_plan_confidence", "pending")
    assert out["review_id"] in env.calls["notified"]                # the human was notified
    assert answer(graph, cfg, {"action": "approve"})["state"] == "done"


def test_user_requested_review_pauses_even_when_confident(env, graph):
    cfg = begin(graph, review=True)
    out = state_of(graph, cfg)
    assert out["state"] == "waiting"
    assert env.queue.get(out["review_id"])["trigger"] == "user_requested"


def test_plan_reject_aborts_and_is_remembered(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    out = answer(graph, cfg, {"action": "reject", "note": "too risky"})
    assert out["state"] == "rejected" and env.calls["specialist"] == []
    assert "too risky" in out["values"]["escalation_reason"]
    assert any(m.kind == "decision" for m in env.ltm.fetch("alice"))


def test_plan_modify_with_edited_plan(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    edited = mk_plan(conf=0.9, ids=("t1",)).model_dump(mode="json")
    out = answer(graph, cfg, {"action": "modify", "plan": edited})
    assert out["state"] == "done"
    assert [c[0] for c in env.calls["specialist"]] == ["t1"]
    assert out["values"]["final_answer"] == "FINAL:t1"


def test_plan_modify_with_note_triggers_replan(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    env.knobs["plan"] = mk_plan(conf=0.9)
    out = answer(graph, cfg, {"action": "modify", "note": "use fewer steps"})
    assert out["state"] == "done"
    assert env.calls["plan_ctx"][1][1] == "use fewer steps"        # the supervisor saw the human's request


def test_plan_take_over_agents_stand_down(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    out = answer(graph, cfg, {"action": "take_over", "output": "my own answer"})
    assert out["state"] == "done" and out["values"]["final_answer"] == "my own answer"
    assert env.calls["specialist"] == []


# ---------------- Approve action ----------------
def test_sensitive_action_waits_for_approval_then_runs_once(env, graph):
    env.knobs["sensitive"] = lambda sid, n: sid == "t1" and n == 1
    cfg = begin(graph)
    out = state_of(graph, cfg)
    assert out["state"] == "waiting" and env.calls["tools"] == []   # nothing executed yet
    item = env.queue.get(out["review_id"])
    assert (item["level"], item["trigger"]) == ("approve_action", "sensitive_action")
    assert item["payload"]["proposed"]["tool"] == "delete_file"
    out = answer(graph, cfg, {"action": "approve"})
    assert out["state"] == "done"
    assert env.calls["tools"] == [("delete_file", "research", {"path": "old.txt"}, True)]
    assert "ALREADY executed" in t1_calls(env)[1][2]


def test_sensitive_action_modify_arguments(env, graph):
    env.knobs["sensitive"] = lambda sid, n: sid == "t1" and n == 1
    cfg = begin(graph)
    answer(graph, cfg, {"action": "modify", "args": {"path": "safe.txt"}})
    assert env.calls["tools"] == [("delete_file", "research", {"path": "safe.txt"}, True)]


def test_sensitive_action_reject_never_runs(env, graph):
    env.knobs["sensitive"] = lambda sid, n: sid == "t1" and n == 1
    cfg = begin(graph)
    out = answer(graph, cfg, {"action": "reject", "note": "no"})
    assert out["state"] == "rejected" and env.calls["tools"] == []


def test_low_quality_asks_to_confirm_the_deliverable(env, graph):
    env.knobs["plan"] = mk_plan(ids=("t1",))
    env.knobs["conf"] = 0.1
    cfg = begin(graph)
    out = state_of(graph, cfg)
    item = env.queue.get(out["review_id"])
    assert (out["state"], item["level"], item["trigger"]) == ("waiting", "approve_action", "low_quality")
    assert env.wm.get_output("task1", "t1") == "out-t1"             # working memory kept while waiting
    assert answer(graph, cfg, {"action": "approve"})["state"] == "done"


def test_borderline_quality_only_notifies(env, graph):
    env.knobs["plan"] = mk_plan(ids=("t1",))
    env.knobs["conf"] = 0.7
    out = state_of(graph, begin(graph))
    assert out["state"] == "done"                                   # no pause
    notes = env.queue.list_reviews("notified")
    assert len(notes) == 1 and notes[0]["level"] == "notify"


# ---------------- Take over ----------------
def test_failed_twice_human_takes_over_and_flow_continues(env, graph):
    env.knobs["crash"] = lambda sid, n: sid == "t1"
    cfg = begin(graph)
    out = state_of(graph, cfg)
    item = env.queue.get(out["review_id"])
    assert (item["level"], item["trigger"]) == ("take_over", "failed_twice")
    out = answer(graph, cfg, {"action": "take_over", "output": "manual result"})
    assert out["state"] == "done"
    assert out["values"]["results"]["t1"]["by_human"] is True
    assert [c for c in env.calls["specialist"] if c[0] == "t2"][0][1] == {"t1": "manual result"}


def test_failed_twice_modify_with_guidance_retries_with_fresh_attempts(env, graph):
    env.knobs["crash"] = lambda sid, n: sid == "t1" and n <= 2
    cfg = begin(graph)
    assert state_of(graph, cfg)["state"] == "waiting"
    out = answer(graph, cfg, {"action": "modify", "note": "try method B"})
    assert out["state"] == "done"
    assert t1_calls(env)[2][2] == "Human reviewer guidance: try method B"
    assert out["values"]["results"]["t1"]["attempts"] == 1


def test_failed_twice_reject_aborts(env, graph):
    env.knobs["crash"] = lambda sid, n: sid == "t1"
    cfg = begin(graph)
    assert answer(graph, cfg, {"action": "reject"})["state"] == "rejected"


# ---------------- queue + resume ----------------
def test_resume_uses_the_decision_stored_in_the_queue(env, graph):
    env.knobs["plan"] = mk_plan(conf=0.2)
    cfg = begin(graph)
    rid = state_of(graph, cfg)["review_id"]
    with pytest.raises(ValueError):
        runner.resume(rid, queue=env.queue, graph=graph)            # not decided yet
    assert env.queue.decide(rid, {"action": "approve"})
    assert runner.resume(rid, queue=env.queue, graph=graph)["state"] == "done"
    assert env.queue.get(rid)["status"] == "resolved"
    with pytest.raises(ValueError):
        runner.resume(rid, queue=env.queue, graph=graph)            # cannot be resumed twice


# ---------------- Phase 2 memory behavior ----------------
def test_success_is_memorized_and_working_memory_cleared(env, graph):
    out = state_of(graph, begin(graph))
    assert out["values"]["memory_report"]["episode"] == 1
    assert "episode" in [m.kind for m in env.ltm.fetch("alice")]
    snap = env.wm.snapshot("task1")
    assert snap["outputs"] == {} and snap["plan"] is None


def test_second_run_gets_memory_injected_into_planning(env, graph):
    begin(graph, tid="a")
    begin(graph, tid="b")
    assert env.calls["plan_ctx"][0][0] == ""
    assert "find prime numbers" in env.calls["plan_ctx"][1][0]


def test_rejection_clears_working_memory(env, graph):
    env.knobs["plan"] = mk_plan(ids=("t1",))
    env.knobs["conf"] = 0.1
    cfg = begin(graph)
    answer(graph, cfg, {"action": "reject"})
    assert env.wm.snapshot("task1")["outputs"] == {}


def test_memory_is_per_user(env, graph):
    begin(graph)
    assert env.ltm.fetch("bob") == []