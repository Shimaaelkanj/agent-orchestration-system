
import pytest
from pydantic import BaseModel

import orchestrator.tools.registry as reg_mod
from orchestrator import config
from orchestrator.hitl import policy
from orchestrator.hitl.policy import Level, Trigger
from orchestrator.hitl.queue import ReviewQueue
from orchestrator.tools import ApprovalRequired, ToolRegistry, ToolSpec


def test_plan_triggers(monkeypatch):
    monkeypatch.setattr(config, "PLAN_CONFIDENCE_THRESHOLD", 0.5)
    assert policy.plan_triggers(0.2, False) == [Trigger.LOW_PLAN_CONFIDENCE]
    assert policy.plan_triggers(0.9, True) == [Trigger.USER_REQUESTED]
    assert policy.plan_triggers(0.9, False) == []


def test_quality_triggers(monkeypatch):
    monkeypatch.setattr(config, "CONFIDENCE_THRESHOLD", 0.6)
    monkeypatch.setattr(config, "NOTIFY_BELOW", 0.8)
    assert policy.quality_trigger(0.3) == Trigger.LOW_QUALITY
    assert policy.quality_trigger(0.7) == Trigger.BORDERLINE_QUALITY
    assert policy.quality_trigger(0.95) is None


def test_level_mapping():
    assert policy.level_for(Trigger.LOW_PLAN_CONFIDENCE) == Level.APPROVE_PLAN
    assert policy.level_for(Trigger.USER_REQUESTED) == Level.APPROVE_PLAN
    assert policy.level_for(Trigger.SENSITIVE_ACTION) == Level.APPROVE_ACTION
    assert policy.level_for(Trigger.LOW_QUALITY) == Level.APPROVE_ACTION
    assert policy.level_for(Trigger.FAILED_TWICE) == Level.TAKE_OVER
    assert policy.level_for(Trigger.BORDERLINE_QUALITY) == Level.NOTIFY


def test_queue_lifecycle(tmp_path):
    q = ReviewQueue(tmp_path / "q.db")
    assert q.submit("r1", "t1", "alice", "approve_plan", "low_plan_confidence", {"x": 1}) is True
    assert q.submit("r1", "t1", "alice", "approve_plan", "low_plan_confidence", {"x": 1}) is False   # idempotent
    assert [i["review_id"] for i in q.list_reviews("pending")] == ["r1"]
    assert q.decide("r1", {"action": "approve"}) is True
    assert q.decide("r1", {"action": "reject"}) is False            # a review is decided only once
    item = q.get("r1")
    assert item["status"] == "decided" and item["decision"]["action"] == "approve"
    assert item["trigger"] == "low_plan_confidence"
    q.mark_resolved("r1")
    assert q.get("r1")["status"] == "resolved"
    q.add_chat("r1", "human", "why?")
    assert q.chat("r1")[0]["content"] == "why?"


class In(BaseModel):
    path: str


class Out(BaseModel):
    ok: bool


def make_registry(monkeypatch, tmp_path, sensitive):
    monkeypatch.setattr(reg_mod, "LOGS_DIR", tmp_path)
    ran = []
    reg = ToolRegistry()
    reg.register(ToolSpec("danger", "d", In, Out, lambda i: (ran.append(i.path) or {"ok": True}),
                          {"writing"}, sensitive=sensitive))
    return reg, ran


def test_sensitive_tool_needs_approval(monkeypatch, tmp_path):
    reg, ran = make_registry(monkeypatch, tmp_path, True)
    with pytest.raises(ApprovalRequired) as e:
        reg.invoke("danger", "writing", {"path": "a.txt"})
    assert ran == [] and e.value.tool == "danger"                    # nothing ran
    assert reg.invoke("danger", "writing", {"path": "a.txt"}, approved=True) == '{"ok": true}'
    assert ran == ["a.txt"]


def test_sensitivity_can_depend_on_the_arguments(monkeypatch, tmp_path):
    reg, ran = make_registry(monkeypatch, tmp_path, lambda i: i.path.endswith(".sql"))
    assert reg.invoke("danger", "writing", {"path": "ok.txt"}) == '{"ok": true}'
    with pytest.raises(ApprovalRequired):
        reg.invoke("danger", "writing", {"path": "drop.sql"})
    assert ran == ["ok.txt"]