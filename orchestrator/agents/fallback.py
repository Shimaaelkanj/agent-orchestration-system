
import os
import re

from ..schemas import Complexity, ExecutionPlan, Specialist, Subtask

_RULES = [
    (Specialist.CODE_EXECUTION, r"\b(code|python|function|script|program|algorithm|test it|run)\b"),
    (Specialist.DATA_ANALYSIS, r"\b(csv|dataset|sql|database|statistic|analy[sz]e data)\b"),
    (Specialist.RESEARCH, r"\b(search|research|find out|compare|latest|news|look up|sources?)\b"),
]


def enabled() -> bool:
    return os.getenv("PLAN_FALLBACK", "1") == "1"


def confidence() -> float:
    return float(os.getenv("PLAN_FALLBACK_CONFIDENCE", "0.4"))     # below 0.5: a human approves it


def guess_specialist(task: str) -> Specialist:
    low = task.lower()
    for specialist, pattern in _RULES:
        if re.search(pattern, low):
            return specialist
    return Specialist.WRITING


def fallback_plan(task: str) -> ExecutionPlan:
    return ExecutionPlan(
        rationale="Fallback plan: the supervisor model could not produce a structured plan, so the whole "
                  "request was assigned to one specialist.",
        confidence=confidence(),
        subtasks=[Subtask(id="t1", description=task, specialist=guess_specialist(task), depends_on=[],
                          required_inputs=[], expected_output="a complete answer with concrete results",
                          complexity=Complexity.MEDIUM)])