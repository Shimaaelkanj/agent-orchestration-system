
from pydantic import BaseModel, Field

from ..tools.registry import ToolRegistry, ToolSpec
from .working import CURRENT_TASK

ALL_AGENTS = {"research", "data_analysis", "writing", "code_execution"}


class NoteWriteIn(BaseModel):
    key: str = Field(description="Short name, e.g. 'prime_list'")
    value: str = Field(description="Text to save")


class NoteWriteOut(BaseModel):
    saved: bool


class NoteReadIn(BaseModel):
    key: str


class NoteReadOut(BaseModel):
    found: bool
    value: str


def _task_id() -> str:
    tid = CURRENT_TASK.get()
    if not tid:
        raise RuntimeError("No active task")
    return tid


def register_memory_tools(registry: ToolRegistry, get_wm) -> None:
    def note_write(inp: NoteWriteIn) -> dict:
        get_wm().note_write(_task_id(), inp.key, inp.value[:4000])
        return {"saved": True}

    def note_read(inp: NoteReadIn) -> dict:
        v = get_wm().note_read(_task_id(), inp.key)
        return {"found": v is not None, "value": v or ""}

    registry.register(ToolSpec("note_write", "Save an intermediate result to the shared task notes.",
                               NoteWriteIn, NoteWriteOut, note_write, ALL_AGENTS, rate_limit_per_min=60))
    registry.register(ToolSpec("note_read", "Read an intermediate result saved earlier in this task.",
                               NoteReadIn, NoteReadOut, note_read, ALL_AGENTS, rate_limit_per_min=60))