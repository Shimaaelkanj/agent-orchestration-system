
from enum import Enum

from .. import config


class Level(str, Enum):
    NOTIFY = "notify"                    # proceed, but inform the human
    APPROVE_ACTION = "approve_action"    # human confirms the next step
    APPROVE_PLAN = "approve_plan"        # human reviews the whole plan before any work
    TAKE_OVER = "take_over"              # human provides the output; agents stand down


class Trigger(str, Enum):
    LOW_PLAN_CONFIDENCE = "low_plan_confidence"
    USER_REQUESTED = "user_requested"
    SENSITIVE_ACTION = "sensitive_action"
    LOW_QUALITY = "low_quality"
    FAILED_TWICE = "failed_twice"
    BORDERLINE_QUALITY = "borderline_quality"
    FINAL_APPROVAL = "final_approval"


# The whole escalation policy is this table: easy to read, review and change.
LEVEL_FOR = {
    Trigger.LOW_PLAN_CONFIDENCE: Level.APPROVE_PLAN,
    Trigger.USER_REQUESTED: Level.APPROVE_PLAN,
    Trigger.SENSITIVE_ACTION: Level.APPROVE_ACTION,
    Trigger.LOW_QUALITY: Level.APPROVE_ACTION,
    Trigger.FAILED_TWICE: Level.TAKE_OVER,
    Trigger.BORDERLINE_QUALITY: Level.NOTIFY,
    Trigger.FINAL_APPROVAL: Level.APPROVE_ACTION,
}


def level_for(trigger: Trigger) -> Level:
    return LEVEL_FOR[trigger]


def plan_triggers(plan_confidence: float, user_requested: bool) -> list[Trigger]:
    out: list[Trigger] = []
    if user_requested:
        out.append(Trigger.USER_REQUESTED)
    if plan_confidence < config.PLAN_CONFIDENCE_THRESHOLD:
        out.append(Trigger.LOW_PLAN_CONFIDENCE)
    return out


def quality_trigger(confidence: float) -> Trigger | None:
    if confidence < config.CONFIDENCE_THRESHOLD:
        return Trigger.LOW_QUALITY
    if confidence < config.NOTIFY_BELOW:
        return Trigger.BORDERLINE_QUALITY
    return None