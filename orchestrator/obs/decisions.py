
import json
import threading

from opentelemetry import context as otel_context

from .. import config
from ..memory.service import recall_for_planning as _recall
from ..schemas import ExecutionPlan, ReviewVerdict, SpecialistReport, Subtask
from . import replay_core
from .replay_core import Kind, encode_exc
from .tracing import agent_scope, get_store, set_status, span

if config.DEMO_MODE:      # deterministic showcase agents (the tools, graph, memory and queue stay real)
    from ..demo.scripted import create_plan as _create_plan
    from ..demo.scripted import review_output as _review_output
    from ..demo.scripted import run_specialist as _run_specialist
    from ..demo.scripted import synthesize as _synthesize
else:
    from ..agents import create_plan as _create_plan
    from ..agents import review_output as _review_output
    from ..agents import run_specialist as _run_specialist
    from ..agents import synthesize as _synthesize

_counters: dict = {}
_lock = threading.Lock()


def reset() -> None:
    with _lock:
        _counters.clear()


def _next_index(task_id: str, kind: str, key: str) -> int:
    """Occurrence number of this decision in the task. It is seeded from the store, so it keeps counting
    correctly after the task is resumed in a different process."""
    k = (task_id, kind, key)
    with _lock:
        if k not in _counters:
            _counters[k] = get_store().count_decisions(task_id, kind, key) if task_id else 0
        i = _counters[k]
        _counters[k] += 1
        return i


def _clip(text, n: int = 120) -> str:
    t = " ".join(str(text).split())
    return t if len(t) <= n else t[: n - 3] + "..."


PLAN = Kind("plan", lambda p: p.model_dump(mode="json"), ExecutionPlan.model_validate,
            lambda p: "warning" if p.confidence < config.PLAN_CONFIDENCE_THRESHOLD else "success",
            lambda p: f"confidence {p.confidence:.2f}, {len(p.subtasks)} subtasks")
SPECIALIST = Kind("specialist",
                  lambda r: {"report": r[0].model_dump(), "evidence": r[1]},
                  lambda d: (SpecialistReport.model_validate(d["report"]), d["evidence"]),
                  lambda r: "warning" if r[0].confidence < config.CONFIDENCE_THRESHOLD else "success",
                  lambda r: _clip(r[0].answer))
REVIEW = Kind("review", lambda v: v.model_dump(), ReviewVerdict.model_validate,
              lambda v: "success" if v.approved else "warning",
              lambda v: "approved" if v.approved else "rejected: " + _clip(v.feedback, 80))
SYNTH = Kind("synthesize", lambda s: str(s), lambda s: s, lambda s: "success", _clip)
MEMORY = Kind("memory", lambda s: s, lambda s: s, lambda s: "success", lambda s: f"{len(s)} chars retrieved")


def _decide(kind: Kind, key: str, inputs: dict, live, agent: str):
    ctrl = replay_core.current()
    if ctrl is not None:                                  # replay mode: the controller decides
        return ctrl.handle(kind, key, inputs, live, agent)
    task_id = otel_context.get_value("obs.task") or ""
    idx = _next_index(task_id, kind.name, key)
    with agent_scope(agent), span(f"decision.{kind.name}", **{"decision.kind": kind.name,
                                                               "decision.key": key,
                                                               "decision.index": idx}) as sp:
        sp.set_attribute("decision.input", json.dumps(inputs, default=str)[: config.MAX_DECISION_CHARS])
        try:
            result = live(inputs)
            if result is None:      # small models sometimes answer in plain text instead of calling the schema tool
                raise ValueError(f"The model returned no structured {kind.name} output")
        except Exception as e:                            # failures and approval requests are decisions too
            sp.set_attribute("decision.output", json.dumps(encode_exc(e), default=str))
            raise
        sp.set_attribute("decision.output", json.dumps(kind.encode(result), default=str)[: config.MAX_DECISION_CHARS])
        sp.set_attribute("decision.summary", kind.summary(result)[:200])
        set_status(sp, kind.status(result))
        return result


# The graph imports these wrappers instead of the raw agent functions. Signatures are identical.
def create_plan(task, previous_error=None, memory_context="", human_feedback=""):
    inputs = {"task": task, "previous_error": previous_error,
              "memory_context": memory_context, "human_feedback": human_feedback}
    return _decide(PLAN, "", inputs,
                   lambda i: _create_plan(i["task"], i["previous_error"], i["memory_context"], i["human_feedback"]),
                   "supervisor")


def run_specialist(subtask, dep_outputs, feedback, registry):
    inputs = {"subtask": subtask.model_dump(mode="json"), "dep_outputs": dict(dep_outputs), "feedback": feedback}
    agent = f"specialist:{subtask.specialist.value}"

    def live(i):
        if config.ASYNC_SPECIALISTS:                      # run on a Celery worker (a different process)
            from ..worker.remote import run_remote
            return run_remote(i)
        return _run_specialist(Subtask.model_validate(i["subtask"]), i["dep_outputs"], i["feedback"], registry)

    return _decide(SPECIALIST, subtask.id, inputs, live, agent)


def review_output(subtask, report, evidence=""):
    inputs = {"subtask": subtask.model_dump(mode="json"), "answer": report.answer,
              "confidence": report.confidence, "evidence": evidence}
    return _decide(REVIEW, subtask.id, inputs,
                   lambda i: _review_output(Subtask.model_validate(i["subtask"]),
                                            SpecialistReport(answer=i["answer"], confidence=i["confidence"]),
                                            i["evidence"]),
                   "reviewer")


def synthesize(task, outputs):
    return _decide(SYNTH, "", {"task": task, "outputs": dict(outputs)},
                   lambda i: _synthesize(i["task"], i["outputs"]), "supervisor")


def recall_for_planning(ltm, user_id, task, max_chars=2500):
    return _decide(MEMORY, "", {"user_id": user_id, "task": task},
                   lambda i: _recall(ltm, i["user_id"], i["task"], max_chars), "memory")