
import threading
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langgraph.checkpoint.memory import MemorySaver
from langgraph.errors import GraphInterrupt
from langgraph.types import Command

import orchestrator.graph as g
import orchestrator.memory.service as svc
from orchestrator import config
from orchestrator.hitl.queue import ReviewQueue
from orchestrator.memory.embeddings import HashEmbedder
from orchestrator.memory.long_term import LongTermMemory
from orchestrator.memory.working import WorkingMemory, _MemoryBackend
from orchestrator.obs import decisions, metrics, replay, tracing, tree
from orchestrator.schemas import (Complexity, ExecutionPlan, ReviewVerdict, Specialist,
                                  SpecialistReport, Subtask)


def spans_of(task):
    return tracing.get_store().spans(task)


# ---------------- tracing basics ----------------
def test_node_spans_nest_and_carry_task_and_agent():
    with tracing.node_span("node.a", "T1"):
        with tracing.agent_scope("reviewer"), tracing.span("child", x=1):
            pass
    s = {x["name"]: x for x in spans_of("T1")}
    assert s["child"]["parent_id"] == s["node.a"]["span_id"]
    assert s["child"]["attrs"]["agent"] == "reviewer" and s["child"]["attrs"]["task.id"] == "T1"
    assert s["node.a"]["status"] == "success"
    assert s["node.a"]["trace_id"] == format(tracing.trace_id_for("T1"), "032x")
    assert s["node.a"]["parent_id"] == format(tracing.root_span_id_for("T1"), "016x")


def test_exceptions_and_interrupts_set_the_status():
    with pytest.raises(ValueError):
        with tracing.node_span("node.fail", "T2"):
            raise ValueError("boom")
    with pytest.raises(GraphInterrupt):
        with tracing.node_span("node.pause", "T2"):
            raise GraphInterrupt()
    with tracing.node_span("node.warn", "T2"):
        tracing.mark("warning")
    status = {x["name"]: x["status"] for x in spans_of("T2")}
    assert status == {"node.fail": "failure", "node.pause": "escalated", "node.warn": "warning"}
    assert [x for x in spans_of("T2") if x["name"] == "node.fail"][0]["attrs"]["error.message"] == "boom"


def test_propagate_keeps_the_parent_inside_worker_threads():
    with tracing.node_span("node.exec", "T3"):
        def work():
            with tracing.span("sub"):
                pass
        t = threading.Thread(target=tracing.propagate(work))
        t.start()
        t.join()
    s = {x["name"]: x for x in spans_of("T3")}
    assert s["sub"]["parent_id"] == s["node.exec"]["span_id"] and s["sub"]["attrs"]["task.id"] == "T3"


def test_llm_callback_records_prompt_response_tokens_and_cost(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(config, "PRICING_FILE", tmp_path / "none.json")
    cb = tracing.ObsCallback("specialist", "openai", "gpt-4o-mini")
    with tracing.node_span("node.exec", "T4"), tracing.agent_scope("specialist:research"):
        cb.on_chat_model_start({}, [[HumanMessage("hi there")]], run_id="r1")
        msg = AIMessage(content="hello", usage_metadata={"input_tokens": 1000, "output_tokens": 500,
                                                         "total_tokens": 1500})
        cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id="r1")
    a = next(x for x in spans_of("T4") if x["name"] == "llm.call")["attrs"]
    assert a["llm.response"] == "hello" and a["llm.total_tokens"] == 1500
    assert a["agent"] == "specialist:research" and "hi there" in a["llm.prompt"]
    assert a["llm.cost_usd"] == pytest.approx((1000 * 0.15 + 500 * 0.60) / 1_000_000)


def test_local_models_are_free_and_unknown_models_are_flagged(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PRICING_FILE", tmp_path / "none.json")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")

    def call(cb, rid):
        msg = AIMessage(content="x", usage_metadata={"input_tokens": 10, "output_tokens": 10, "total_tokens": 20})
        cb.on_chat_model_start({}, [[HumanMessage("q")]], run_id=rid)
        cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id=rid)

    with tracing.node_span("node.exec", "T5"):
        call(tracing.ObsCallback("specialist", "openai", "llama3.2"), "a")
        monkeypatch.delenv("OPENAI_BASE_URL")
        call(tracing.ObsCallback("specialist", "openai", "mystery-model"), "b")
    attrs = [x["attrs"] for x in spans_of("T5") if x["name"] == "llm.call"]
    assert attrs[0]["llm.cost_usd"] == 0 and attrs[0]["llm.priced"] is True
    assert attrs[1]["llm.cost_usd"] == 0 and attrs[1]["llm.priced"] is False


def test_task_list_shows_the_outcome():
    with tracing.node_span("node.intake", "T6"):
        tracing.annotate(**{"task.text": "do x", "user.id": "alice"})
    with tracing.node_span("node.delivery", "T6"):
        tracing.annotate(**{"result.status": "done", "result.final_answer": "ANSWER"})
    t = tracing.get_store().tasks()[0]
    assert (t["task_id"], t["outcome"], t["task_text"]) == ("T6", "done", "do x")
    assert tracing.get_store().task_result("T6")["final_answer"] == "ANSWER"


# ---------------- metrics ----------------
def mk(name, start, dur, attrs=None, status="success", task="S"):
    return {"span_id": f"{name}{start}{task}", "parent_id": None, "trace_id": "t", "task_id": task, "name": name,
            "start": start, "end": start + dur, "duration_s": dur, "status": status,
            "attrs": attrs or {}, "events": []}


def llm_attrs(agent, cost):
    return {"agent": agent, "llm.model": "m1", "llm.input_tokens": 100, "llm.output_tokens": 50,
            "llm.cost_usd": cost, "llm.priced": True}


def test_task_summary_aggregates_tokens_cost_tools_and_human_time():
    spans = [mk("node.intake", 0, 1, {"task.text": "t", "user.id": "u"}),
             mk("node.planning", 1, 2, {"plan.task_type": "code_execution+writing"}),
             mk("llm.call", 1, 1, llm_attrs("supervisor", 0.01)),
             mk("llm.call", 3, 1, llm_attrs("reviewer", 0.02)),
             mk("tool.call", 4, 0.5, {"tool.name": "run_python", "tool.success": False, "tool.latency_ms": 500},
                status="warning"),
             mk("node.delivery", 10, 1)]
    reviews = [{"level": "approve_plan", "status": "decided", "created_at": 100.0, "decided_at": 190.0},
               {"level": "notify", "status": "notified", "created_at": 1.0, "decided_at": None}]
    s = metrics.task_summary(spans, reviews)
    assert (s["llm_calls"], s["input_tokens"], s["output_tokens"]) == (2, 200, 100)
    assert s["cost_usd"] == pytest.approx(0.03)
    assert (s["tool_calls"], s["tool_failures"]) == (1, 1)
    assert s["human_wait_s"] == 90 and s["escalated"] is True        # the notify item is ignored
    assert (s["outcome"], s["task_type"]) == ("done", "code_execution+writing")
    assert {a["agent"] for a in s["by_agent"]} == {"supervisor", "reviewer"}
    assert s["wall_s"] == 11 and s["active_s"] == 1 + 2 + 1


def test_aggregate_groups_by_type_agent_tool_and_day():
    day = 86400

    def task(tid, t0, ttype, cost, escalated):
        spans = [mk("node.intake", t0, 1, task=tid), mk("node.planning", t0 + 1, 1, {"plan.task_type": ttype}, task=tid),
                 mk("llm.call", t0 + 1, 1, llm_attrs("supervisor", cost), task=tid),
                 mk("tool.call", t0 + 2, 1, {"tool.name": "run_python", "tool.success": True,
                                             "tool.latency_ms": 100}, task=tid),
                 mk("node.delivery", t0 + 3, 1, task=tid)]
        rv = [{"level": "approve_plan", "status": "decided", "created_at": 0, "decided_at": 5}] if escalated else []
        return metrics.task_summary(spans, rv)

    agg = metrics.aggregate([task("a", 1_700_000_000, "code_execution", 0.10, False),
                             task("b", 1_700_000_000 + 2 * day, "code_execution", 0.30, True),
                             task("c", 1_700_000_000 + 2 * day, "research+writing", 0.05, True)])
    by_type = {r["task_type"]: r for r in agg["by_type"]}
    assert by_type["code_execution"]["tasks"] == 2 and by_type["code_execution"]["cost_usd"] == pytest.approx(0.40)
    assert by_type["code_execution"]["avg_cost_usd"] == pytest.approx(0.20)
    assert agg["by_agent"][0]["agent"] == "supervisor" and agg["by_agent"][0]["cost_usd"] == pytest.approx(0.45)
    assert agg["tools"][0]["tool"] == "run_python" and agg["tools"][0]["calls"] == 3
    assert [round(d["escalation_rate"], 2) for d in agg["trend"]] == [0.0, 1.0]
    assert agg["totals"]["tasks"] == 3 and agg["totals"]["escalation_rate"] == pytest.approx(0.667, abs=0.001)


def test_tree_rollup_and_dot_use_status_colors():
    with tracing.node_span("node.a", "TR"):
        with tracing.span("decision.review") as sp:
            tracing.set_status(sp, "warning")
        with tracing.span("tool.call") as sp2:
            tracing.set_status(sp2, "failure")
    order = tree.build_tree(spans_of("TR"))
    assert (order[0][0], order[0][1]["name"]) == (0, "node.a")
    rolled = tree.rollup(order)
    assert rolled[order[0][1]["span_id"]]["tools"] == 1
    full = tree.to_dot(order, rolled)
    assert tree.COLORS["warning"] in full and tree.COLORS["failure"] in full
    assert full.count("->") > tree.to_dot(order, rolled, show_leaf=False).count("->")


# ---------------- a real graph run, recorded and replayed ----------------
def st_(i, deps):
    return Subtask(id=i, description=f"do {i}", specialist=Specialist.RESEARCH, depends_on=deps,
                   required_inputs=[], expected_output="text", complexity=Complexity.LOW)


@pytest.fixture
def run_env(monkeypatch, tmp_path):
    calls = {"plan": 0, "spec": 0, "review": 0}
    plan = ExecutionPlan(rationale="r", confidence=0.9, subtasks=[st_("t1", []), st_("t2", ["t1"])])

    def fake_plan(task, err=None, memory_context="", human_feedback=""):
        calls["plan"] += 1
        return plan

    def fake_spec(subtask, deps, feedback, registry):
        calls["spec"] += 1
        return SpecialistReport(answer=f"out-{subtask.id}", confidence=0.9), f"TOOL x {subtask.id}"

    def fake_review(subtask, report, evidence=""):
        calls["review"] += 1
        return ReviewVerdict(approved=True, feedback="", confidence=0.9)

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
    monkeypatch.setattr(g, "notify", lambda item: None)
    monkeypatch.setattr(svc, "llm_extract", lambda s: None)
    monkeypatch.setattr(decisions, "_create_plan", fake_plan)
    monkeypatch.setattr(decisions, "_run_specialist", fake_spec)
    monkeypatch.setattr(decisions, "_review_output", fake_review)
    monkeypatch.setattr(decisions, "_synthesize", lambda task, outputs: "FINAL:" + ",".join(sorted(outputs)))
    monkeypatch.setattr(decisions, "_recall", lambda ltm_, uid, task, max_chars=2500: "MEMCTX")
    return SimpleNamespace(calls=calls, ltm=ltm, wm=wm, queue=queue)


def run(graph, tid="task1", review=False):
    cfg = {"configurable": {"thread_id": tid}}
    graph.invoke({"task": "find primes", "user_id": "alice", "task_id": tid, "review_requested": review}, cfg)
    return cfg


def do_replay(env, overrides=None, **kw):
    return replay.run_replay("task1", overrides or [], store=tracing.get_store(), queue=env.queue,
                             graph=g.build_graph(MemorySaver()), **kw)


def test_a_run_produces_a_trace_with_every_decision(run_env):
    run(g.build_graph(MemorySaver()))
    store = tracing.get_store()
    assert [(d["kind"], d["key"], d["index"]) for d in store.decisions("task1")] == [
        ("memory", "", 0), ("plan", "", 0), ("specialist", "t1", 0), ("review", "t1", 0),
        ("specialist", "t2", 0), ("review", "t2", 0), ("synthesize", "", 0)]
    names = {s["name"] for s in store.spans("task1")}
    assert {"node.intake", "node.planning", "node.plan_gate", "node.execute", "node.review", "node.synthesis",
            "node.memorize", "node.delivery"} <= names
    by_id = {s["span_id"]: s for s in store.spans("task1")}
    for s in store.spans("task1"):
        if s["name"].startswith("decision."):               # every decision hangs under a graph node
            assert by_id[s["parent_id"]]["name"].startswith("node.")
    assert store.task_result("task1")["final_answer"] == "FINAL:t1,t2"


def test_faithful_replay_reuses_every_recording(run_env):
    run(g.build_graph(MemorySaver()))
    before = dict(run_env.calls)
    rep = do_replay(run_env)
    assert {s["mode"] for s in rep["steps"]} == {"replayed"} and len(rep["steps"]) == 7
    assert run_env.calls == before                               # the models were not called
    assert rep["replayed"]["final_answer"] == "FINAL:t1,t2" and rep["diverged_at"] is None
    assert rep["diff"] == [] and rep["answer_changed"] is False


def test_replay_with_a_replaced_output_diverges_then_converges(run_env):
    run(g.build_graph(MemorySaver()))
    spec_calls = run_env.calls["spec"]
    ov = [{"kind": "review", "key": "t1", "index": 0,
           "output": {"approved": False, "feedback": "fix it", "confidence": 0.9}}]
    rep = do_replay(run_env, ov)
    modes = {(s["kind"], s["key"], s["index"]): s["mode"] for s in rep["steps"]}
    assert modes[("review", "t1", 0)] == "override-output"
    assert modes[("specialist", "t1", 1)] == "live"              # the retry did not exist in the recording
    assert modes[("review", "t1", 1)] == "live"
    assert modes[("specialist", "t2", 0)] == "replayed"          # t2's input is unchanged: it converges again
    assert run_env.calls["spec"] == spec_calls + 1 and rep["diverged_at"] is not None
    assert rep["replayed"]["status"] == "done"


def test_replay_with_an_edited_input_runs_that_step_live(run_env):
    run(g.build_graph(MemorySaver()))
    spec_calls = run_env.calls["spec"]
    rep = do_replay(run_env, [{"kind": "specialist", "key": "t1", "index": 0,
                               "input_patch": {"feedback": "use method B"}}])
    step = next(s for s in rep["steps"] if (s["kind"], s["key"], s["index"]) == ("specialist", "t1", 0))
    assert step["mode"] == "live-edited" and step["changed"] is False     # same answer: the change did not matter
    assert run_env.calls["spec"] == spec_calls + 1
    assert next(s for s in rep["steps"] if (s["kind"], s["key"], s["index"]) == ("review", "t1", 0))["mode"] == "replayed"


def test_replay_never_touches_memory_or_the_real_queue(run_env):
    run(g.build_graph(MemorySaver()))
    memories, reviews = len(run_env.ltm.fetch("alice")), len(run_env.queue.list_reviews())
    rep = do_replay(run_env)
    assert len(run_env.ltm.fetch("alice")) == memories and len(run_env.queue.list_reviews()) == reviews
    assert tracing.get_store().task_result(rep["replay_id"])["status"] == "done"   # but it has its own trace


def test_replay_reuses_the_original_human_decision_or_your_override(run_env):
    graph = g.build_graph(MemorySaver())
    cfg = run(graph, review=True)                                  # pauses at the plan review
    assert tracing.get_store().task_result("task1")["status"] == "waiting"
    run_env.queue.decide("task1-plan-0", {"action": "approve", "by": "qa"})
    graph.invoke(Command(resume=run_env.queue.get("task1-plan-0")["decision"]), cfg)
    reviews = len(run_env.queue.list_reviews())

    same = do_replay(run_env)
    assert same["human"][0]["source"] == "original" and same["replayed"]["status"] == "done"
    other = do_replay(run_env, [{"kind": "human", "key": "plan-0", "index": 0,
                                 "output": {"action": "reject", "note": "no"}}])
    assert other["human"][0]["source"] == "override" and other["replayed"]["status"] == "rejected"
    assert len(run_env.queue.list_reviews()) == reviews


def test_replay_of_an_unknown_task_is_refused(run_env):
    with pytest.raises(ValueError):
        replay.run_replay("nope", [], store=tracing.get_store(), queue=run_env.queue)


def test_decision_counters_continue_after_a_restart(run_env):
    run(g.build_graph(MemorySaver()))
    decisions.reset()                                              # simulates a new process resuming the task
    with tracing.node_span("node.review", "task1"):
        decisions.review_output(st_("t1", []), SpecialistReport(answer="a", confidence=0.9), "")
    idx = [d["index"] for d in tracing.get_store().decisions("task1") if d["kind"] == "review" and d["key"] == "t1"]
    assert sorted(idx) == [0, 1]


def test_planning_retries_after_malformed_model_output(run_env, monkeypatch):
    good, attempts = decisions._create_plan, {"n": 0}

    def flaky(task, err=None, memory_context="", human_feedback=""):
        attempts["n"] += 1
        if attempts["n"] == 1:
            ExecutionPlan.model_validate({"rationale": "r", "confidence": 0.9, "subtasks": 5})   # raises
        return good(task, err, memory_context, human_feedback)

    monkeypatch.setattr(decisions, "_create_plan", flaky)
    run(g.build_graph(MemorySaver()))
    assert attempts["n"] == 2 and tracing.get_store().task_result("task1")["status"] == "done"


def test_planning_gives_up_after_three_malformed_attempts(run_env, monkeypatch):
    monkeypatch.setenv("PLAN_FALLBACK", "0")                      # without the fallback, planning escalates

    def broken(task, err=None, memory_context="", human_feedback=""):
        ExecutionPlan.model_validate({"rationale": "r", "subtasks": 5})

    monkeypatch.setattr(decisions, "_create_plan", broken)
    run(g.build_graph(MemorySaver()))
    assert tracing.get_store().task_result("task1")["status"] == "needs_human"

def test_planning_retries_when_the_model_returns_no_structured_output(run_env, monkeypatch):
    good, attempts = decisions._create_plan, {"n": 0}

    def sometimes_none(task, err=None, memory_context="", human_feedback=""):
        attempts["n"] += 1
        return None if attempts["n"] == 1 else good(task, err, memory_context, human_feedback)

    monkeypatch.setattr(decisions, "_create_plan", sometimes_none)
    run(g.build_graph(MemorySaver()))
    assert attempts["n"] == 2 and tracing.get_store().task_result("task1")["status"] == "done"


def test_a_reviewer_that_returns_nothing_is_treated_as_a_rejection(run_env, monkeypatch):
    good, state = decisions._review_output, {"n": 0}

    def flaky_review(subtask, report, evidence=""):
        state["n"] += 1
        return None if state["n"] == 1 else good(subtask, report, evidence)

    monkeypatch.setattr(decisions, "_review_output", flaky_review)
    run(g.build_graph(MemorySaver()))
    assert tracing.get_store().task_result("task1")["status"] == "done"
    assert run_env.calls["spec"] == 3                  # t1 was run again after the failed review, then t2    

def test_planning_falls_back_to_a_single_subtask_plan(run_env, monkeypatch):
    def broken(task, err=None, memory_context="", human_feedback=""):
        raise ValueError("no plan")

    monkeypatch.setattr(decisions, "_create_plan", broken)
    monkeypatch.setenv("PLAN_FALLBACK_CONFIDENCE", "0.9")        # confident enough to skip the human gate
    run(g.build_graph(MemorySaver()))
    store = tracing.get_store()
    assert store.task_result("task1")["status"] == "done"
    node = next(s for s in store.spans("task1") if s["name"] == "node.planning")
    assert node["attrs"]["plan.fallback"] is True and node["attrs"]["plan.n_subtasks"] == 1


def test_the_default_fallback_plan_asks_a_human_to_approve(run_env, monkeypatch):
    def broken(task, err=None, memory_context="", human_feedback=""):
        raise ValueError("no plan")

    monkeypatch.setattr(decisions, "_create_plan", broken)
    monkeypatch.delenv("PLAN_FALLBACK_CONFIDENCE", raising=False)
    run(g.build_graph(MemorySaver()))
    assert tracing.get_store().task_result("task1")["status"] == "waiting"
    assert run_env.queue.get("task1-plan-0")["trigger"] == "low_plan_confidence"