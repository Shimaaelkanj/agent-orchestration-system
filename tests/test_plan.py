import pytest
import json

from orchestrator.schemas import (Complexity, ExecutionPlan, Specialist, Subtask,
                                  drop_trailing_writing, validate_plan)
from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.agents.fallback import guess_specialist
from orchestrator.llm import _extract_json, _WithFallback
from orchestrator.schemas import ReviewVerdict


def test_extract_json_handles_fences_and_prose():
    assert _extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert _extract_json('here: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        _extract_json("no json here")


class _Plain:
    def invoke(self, messages):
        return AIMessage(content='```json\n{"approved": true, "feedback": "ok", "confidence": 0.8}\n```')


class _Nothing:
    def invoke(self, messages):
        return None


def test_structured_fallback_reads_plain_json_when_the_primary_returns_nothing():
    v = _WithFallback(_Plain(), ReviewVerdict, _Nothing()).invoke([HumanMessage("x")])
    assert v.approved is True and v.confidence == 0.8


def test_fallback_specialist_is_guessed_from_the_request():
    assert guess_specialist("Write a Python function that checks primes") == Specialist.CODE_EXECUTION
    assert guess_specialist("Research the latest news about LangGraph") == Specialist.RESEARCH
    assert guess_specialist("Greet my grandmother") == Specialist.WRITING

def st(i, deps):
    return Subtask(id=i, description="x", specialist=Specialist.RESEARCH, depends_on=deps,
                   required_inputs=[], expected_output="text", complexity=Complexity.LOW)


def plan(subtasks):
    return ExecutionPlan(rationale="r", confidence=0.9, subtasks=subtasks)


def test_valid_plan():
    validate_plan(plan([st("t1", []), st("t2", ["t1"])]))


def test_cycle_rejected():
    with pytest.raises(ValueError):
        validate_plan(plan([st("t1", ["t2"]), st("t2", ["t1"])]))


def test_unknown_dep_rejected():
    with pytest.raises(ValueError):
        validate_plan(plan([st("t1", ["zzz"])]))


def test_drop_trailing_writing():
    w = st("t3", ["t2"]).model_copy(update={"specialist": Specialist.WRITING})
    p = plan([st("t1", []), st("t2", ["t1"]), w])
    assert [s.id for s in drop_trailing_writing(p).subtasks] == ["t1", "t2"]

def test_plan_accepts_subtasks_sent_as_a_json_string():
    raw = {"rationale": "r", "subtasks": json.dumps([{"description": "x", "specialist": "Code_Execution"}])}
    p = ExecutionPlan.model_validate(raw)
    assert p.subtasks[0].id == "t1" and p.subtasks[0].specialist == Specialist.CODE_EXECUTION
    assert p.confidence == 0.5 and p.subtasks[0].depends_on == []


def test_plan_still_rejects_nonsense():
    with pytest.raises(ValueError):
        ExecutionPlan.model_validate({"rationale": "r", "subtasks": 5})
    with pytest.raises(ValueError):
        ExecutionPlan.model_validate({"rationale": "r", "subtasks": [{"description": "x", "specialist": "wizard"}]})    