import json
from enum import Enum

from pydantic import BaseModel, Field, model_validator


class Specialist(str, Enum):
    RESEARCH = "research"
    DATA_ANALYSIS = "data_analysis"
    WRITING = "writing"
    CODE_EXECUTION = "code_execution"


class Complexity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def _as_list(value):
    """Small models often send a list as a JSON string, as one bare string, or as null."""
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except ValueError:
            return [text]
        return parsed if isinstance(parsed, list) else [parsed]
    return value


# NOTE: no defaults or constraints on the fields: some providers reject them in strict schema mode.
# Tolerance lives in the 'before' validators instead, so the schema sent to the model stays strict.
class Subtask(BaseModel):
    id: str = Field(description="Short unique id such as t1, t2")
    description: str = Field(description="What must be done, self-contained")
    specialist: Specialist
    depends_on: list[str] = Field(description="Ids of subtasks whose output this one needs (empty list if none)")
    required_inputs: list[str] = Field(description="Inputs needed to start")
    expected_output: str = Field(description="Format/content of the expected result")
    complexity: Complexity

    @model_validator(mode="before")
    @classmethod
    def _tolerate(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        d["depends_on"] = [str(x) for x in _as_list(d.get("depends_on"))]
        d["required_inputs"] = [str(x) for x in _as_list(d.get("required_inputs"))]
        if not d.get("complexity"):
            d["complexity"] = "medium"
        if not d.get("expected_output"):
            d["expected_output"] = "the result of this subtask"
        for k in ("specialist", "complexity"):
            if isinstance(d.get(k), str):
                d[k] = d[k].strip().lower()
        if d.get("id") is not None:
            d["id"] = str(d["id"])
        return d


class ExecutionPlan(BaseModel):
    rationale: str
    confidence: float = Field(description="Your confidence from 0 to 1 that this plan fully accomplishes the request")
    subtasks: list[Subtask]

    @model_validator(mode="before")
    @classmethod
    def _tolerate(cls, data):
        if not isinstance(data, dict):
            return data
        d = dict(data)
        subs = _as_list(d.get("subtasks"))
        if isinstance(subs, list):
            subs = [({**s, "id": f"t{i}"} if isinstance(s, dict) and not s.get("id") else s)
                    for i, s in enumerate(subs, 1)]
        d["subtasks"] = subs
        if d.get("confidence") is None:
            d["confidence"] = 0.5          # a missing score counts as "neutral", not as certainty
        if d.get("rationale") is None:
            d["rationale"] = ""
        return d


class SpecialistReport(BaseModel):
    answer: str
    confidence: float = Field(description="Honest self-assessed confidence from 0 to 1")


class ReviewVerdict(BaseModel):
    approved: bool
    feedback: str = Field(description="Concrete fixes if rejected, otherwise a short note")
    confidence: float = Field(description="Reviewer confidence from 0 to 1")


def validate_plan(plan: ExecutionPlan) -> None:
    """Raise ValueError if the plan is empty, has unknown deps, duplicate ids, or cycles."""
    if not plan.subtasks or len(plan.subtasks) > 8:
        raise ValueError("Plan must contain 1-8 subtasks")
    ids = [s.id for s in plan.subtasks]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate subtask ids")
    known = set(ids)
    for s in plan.subtasks:
        for d in s.depends_on:
            if d not in known:
                raise ValueError(f"{s.id} depends on unknown subtask {d}")
            if d == s.id:
                raise ValueError(f"{s.id} depends on itself")
    # cycle detection (Kahn's algorithm)
    indeg = {s.id: len(s.depends_on) for s in plan.subtasks}
    queue = [i for i, n in indeg.items() if n == 0]
    seen = 0
    while queue:
        cur = queue.pop()
        seen += 1
        for s in plan.subtasks:
            if cur in s.depends_on:
                indeg[s.id] -= 1
                if indeg[s.id] == 0:
                    queue.append(s.id)
    if seen != len(plan.subtasks):
        raise ValueError("Plan contains a dependency cycle")


def drop_trailing_writing(plan: ExecutionPlan) -> ExecutionPlan:
    """Drop 'writing' subtasks nobody depends on (the synthesis node writes the final answer),
    as long as at least one other subtask remains."""
    needed = {d for s in plan.subtasks for d in s.depends_on}
    keep = [s for s in plan.subtasks
            if not (s.specialist == Specialist.WRITING and s.id not in needed)]
    if not keep or len(keep) == len(plan.subtasks):
        return plan
    return plan.model_copy(update={"subtasks": keep})