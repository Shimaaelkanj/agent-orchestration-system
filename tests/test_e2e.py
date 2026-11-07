
import json
import uuid
from types import SimpleNamespace

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

import orchestrator.graph as g
import orchestrator.memory.service as svc
from orchestrator import config
from orchestrator.demo import scripted
from orchestrator.hitl import runner
from orchestrator.hitl.checkpoint import make_checkpointer
from orchestrator.hitl.queue import ReviewQueue
from orchestrator.memory.embeddings import HashEmbedder
from orchestrator.memory.long_term import LongTermMemory
from orchestrator.memory.working import WorkingMemory, _MemoryBackend
from orchestrator.obs import decisions, tracing
from orchestrator.schemas import (Complexity, ExecutionPlan, Specialist, SpecialistReport, Subtask,
                                  validate_plan)

TASK_A = ("Research the pros and cons of vector databases for a small startup team, "
          "analyze the benchmark data, and write a recommendation.")
TASK_B = TASK_A.replace("startup", "mobile app")


@pytest.fixture
def e2e(monkeypatch, tmp_path):
    ltm = LongTermMemory(HashEmbedder(), tmp_path / "chroma")
    wm = WorkingMemory(_MemoryBackend())
    queue = ReviewQueue(tmp_path / "q.db")
    for name, value in [("DEMO_MODE", True), ("STRIP_FINAL_WRITING", False), ("MEMORY_MIN_SIMILARITY", 0.05),
                        ("MAX_PARALLEL", 2), ("LOGS_DIR", tmp_path), ("CONFIDENCE_THRESHOLD", 0.6),
                        ("NOTIFY_BELOW", 0.8), ("PLAN_CONFIDENCE_THRESHOLD", 0.5)]:
        monkeypatch.setattr(config, name, value)
    monkeypatch.setattr(decisions, "_create_plan", scripted.create_plan)
    monkeypatch.setattr(decisions, "_run_specialist", scripted.run_specialist)
    monkeypatch.setattr(decisions, "_review_output", scripted.review_output)
    monkeypatch.setattr(decisions, "_synthesize", scripted.synthesize)
    monkeypatch.setattr(svc, "llm_extract", lambda s: None)
    monkeypatch.setattr(g, "get_ltm", lambda: ltm)
    monkeypatch.setattr(g, "get_wm", lambda: wm)
    monkeypatch.setattr(g, "get_queue", lambda: queue)
    monkeypatch.setattr(g, "notify", lambda item: None)
    g.registry._calls.clear()                       # per-tool rate limits are per process
    return SimpleNamespace(ltm=ltm, wm=wm, queue=queue)


def start(e, task=TASK_A, user="alice", tid=None, graph=None, **flags):
    graph = graph or g.build_graph(MemorySaver())
    tid = tid or uuid.uuid4().hex[:8]
    cfg = {"configurable": {"thread_id": tid}}
    graph.invoke({"task": task, "user_id": user, "task_id": tid, **flags}, cfg)
    return graph, cfg, tid


def state(graph, tid):
    return runner._outcome(graph, tid)


def decide(graph, cfg, tid, decision):
    graph.invoke(Command(resume=decision), cfg)
    return state(graph, tid)


def decs(tid, kind, key=None):
    return [d for d in tracing.get_store().decisions(tid) if d["kind"] == kind and (key is None or d["key"] == key)]


def planning_span(tid):
    return next(s for s in tracing.get_store().spans(tid) if s["name"] == "node.planning")


def sub(i, desc, spec=Specialist.CODE_EXECUTION, deps=None):
    return Subtask(id=i, description=desc, specialist=spec, depends_on=deps or [], required_inputs=[],
                   expected_output="text", complexity=Complexity.LOW)


def use_plan(monkeypatch, *subtasks):
    plan = ExecutionPlan(rationale="r", confidence=0.9, subtasks=list(subtasks))
    monkeypatch.setattr(decisions, "_create_plan", lambda *a, **k: plan)


# ---------------- decomposition and tools ----------------
def test_decomposition_produces_a_valid_plan_and_research_runs_in_one_wave(e2e):
    graph, cfg, tid = start(e2e)
    assert state(graph, tid)["state"] == "done"
    plan = ExecutionPlan.model_validate(decs(tid, "plan")[0]["output"])
    validate_plan(plan)
    assert [(s.id, s.specialist.value, s.depends_on) for s in plan.subtasks] == [
        ("t1", "research", []), ("t2", "research", []), ("t3", "data_analysis", ["t2"]),
        ("t4", "writing", ["t1", "t3"])]
    waves = [json.loads(s["attrs"]["execute.ready"]) for s in tracing.get_store().spans(tid)
             if s["name"] == "node.execute"]
    assert waves[0] == ["t1", "t2"] and waves[1] == ["t3"]


def test_specialists_use_their_own_tools_and_report_real_results(e2e):
    graph, cfg, tid = start(e2e)
    calls = [(s["attrs"]["tool.name"], s["attrs"]["tool.agent"])
             for s in tracing.get_store().spans(tid) if s["name"] == "tool.call"]
    assert calls.count(("web_search", "research")) == 2
    assert ("run_python", "data_analysis") in calls
    assert not any(agent == "writing" for _, agent in calls)           # the writer has no tools
    analysis = decs(tid, "specialist", "t3")[0]["output"]["report"]["answer"]
    assert "avg_qps: 1833.3" in analysis and "best_qps: qdrant" in analysis   # computed by real code


# ---------------- reviewer ----------------
def test_reviewer_catches_a_bad_deliverable_and_the_specialist_fixes_it(e2e):
    graph, cfg, tid = start(e2e)
    reviews = decs(tid, "review", "t4")
    assert [r["output"]["approved"] for r in reviews] == [False, True]
    assert "citations" in reviews[0]["output"]["feedback"]
    assert len(decs(tid, "specialist", "t4")) == 2
    assert "[1]" in state(graph, tid)["values"]["final_answer"]


def test_reviewer_rejects_fabricated_or_unsupported_outputs():
    def review(spec, answer, evidence=""):
        return scripted.review_output(sub("t", "x", spec), SpecialistReport(answer=answer, confidence=0.9), evidence)

    assert not review(Specialist.RESEARCH, "made up facts").approved                        # no search was run
    assert not review(Specialist.DATA_ANALYSIS, "avg_qps: 9999", "TOOL run_python ...").approved   # number not in evidence
    assert not review(Specialist.WRITING, "claims without sources").approved
    assert review(Specialist.WRITING, "claim [1] and claim [2]").approved


# ---------------- memory ----------------
def test_memory_improves_planning_for_a_repeated_task(e2e):
    _, _, a = start(e2e, TASK_A, user="alice")
    _, _, b = start(e2e, TASK_B, user="alice")
    _, _, c = start(e2e, TASK_B, user="bob")                           # another user has no memory
    assert planning_span(a)["attrs"].get("plan.used_memory") is not True
    assert planning_span(b)["attrs"]["plan.used_memory"] is True
    assert len(decs(b, "specialist", "t4")) == 1                       # asked for citations up front
    assert all(r["output"]["approved"] for r in decs(b, "review"))
    assert len(decs(c, "specialist", "t4")) == 2                       # bob repeats alice's first mistake


# ---------------- human escalation ----------------
def test_low_plan_confidence_pauses_before_any_work(e2e, monkeypatch):
    monkeypatch.setattr(decisions, "_create_plan",
                        lambda *a, **k: scripted.create_plan(*a, **k).model_copy(update={"confidence": 0.3}))
    graph, cfg, tid = start(e2e)
    out = state(graph, tid)
    item = e2e.queue.get(out["review_id"])
    assert out["state"] == "waiting" and (item["level"], item["trigger"]) == ("approve_plan", "low_plan_confidence")
    assert decs(tid, "specialist") == []
    assert decide(graph, cfg, tid, {"action": "approve"})["state"] == "done"


def test_the_final_deliverable_waits_for_a_human(e2e):
    graph, cfg, tid = start(e2e, approve_final=True)
    out = state(graph, tid)
    item = e2e.queue.get(out["review_id"])
    assert (out["state"], item["trigger"], item["level"]) == ("waiting", "final_approval", "approve_action")
    assert item["payload"]["proposed"]["output"].startswith("Recommendation")
    assert e2e.ltm.fetch("alice") == []                                # nothing is memorized before approval
    done = decide(graph, cfg, tid, {"action": "approve", "by": "qa"})
    assert done["state"] == "done" and e2e.ltm.fetch("alice")


def test_a_human_can_edit_or_reject_the_final_deliverable(e2e):
    graph, cfg, tid = start(e2e, approve_final=True)
    out = decide(graph, cfg, tid, {"action": "modify", "output": "EDITED ANSWER"})
    assert out["state"] == "done" and out["values"]["final_answer"] == "EDITED ANSWER"
    graph2, cfg2, tid2 = start(e2e, approve_final=True)
    out2 = decide(graph2, cfg2, tid2, {"action": "reject", "note": "not good enough"})
    assert out2["state"] == "rejected" and "not good enough" in out2["values"]["escalation_reason"]


def test_a_sensitive_tool_never_runs_before_a_human_approves(e2e, monkeypatch):
    victim = config.WORKSPACE / f"e2e_{uuid.uuid4().hex[:6]}.txt"
    victim.write_text("old report")
    try:
        use_plan(monkeypatch, sub("t1", f"DELETE: {victim.name}"))
        graph, cfg, tid = start(e2e)
        out = state(graph, tid)
        item = e2e.queue.get(out["review_id"])
        assert item["trigger"] == "sensitive_action" and victim.exists()
        assert item["payload"]["proposed"]["args"] == {"path": victim.name}
        assert decide(graph, cfg, tid, {"action": "approve"})["state"] == "done"
        assert not victim.exists()                                      # executed once, after approval
    finally:
        victim.unlink(missing_ok=True)


def test_a_subtask_that_fails_twice_goes_to_a_human_who_takes_over(e2e, monkeypatch):
    use_plan(monkeypatch, sub("t1", "CRASH please", Specialist.RESEARCH), sub("t2", "Summarize", deps=["t1"]))
    graph, cfg, tid = start(e2e)
    out = state(graph, tid)
    item = e2e.queue.get(out["review_id"])
    assert (item["level"], item["trigger"]) == ("take_over", "failed_twice")
    done = decide(graph, cfg, tid, {"action": "take_over", "output": "manual result"})
    assert done["state"] == "done" and done["values"]["results"]["t1"]["by_human"] is True


# ---------------- failure recovery ----------------
def test_the_system_recovers_from_a_one_off_agent_failure(e2e, monkeypatch):
    use_plan(monkeypatch, sub("t1", "FLAKY step"))
    graph, cfg, tid = start(e2e)
    out = state(graph, tid)
    assert out["state"] == "done" and out["values"]["results"]["t1"]["attempts"] == 2
    assert e2e.queue.list_reviews() == []                               # no human was needed


def test_a_paused_run_survives_a_restart(e2e, tmp_path):
    cp = tmp_path / "checkpoints.db"
    g1 = g.build_graph(make_checkpointer(cp))
    _, _, tid = start(e2e, graph=g1, tid="restart1", approve_final=True)
    out = state(g1, tid)
    assert out["state"] == "waiting"
    assert e2e.queue.decide(out["review_id"], {"action": "approve", "by": "qa"})
    g2 = g.build_graph(make_checkpointer(cp))                           # a new "process": same file, new objects
    assert runner.resume(out["review_id"], queue=e2e.queue, graph=g2)["state"] == "done"