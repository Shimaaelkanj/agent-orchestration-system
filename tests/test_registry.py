
from pydantic import BaseModel
from orchestrator.tools import ToolRegistry, ToolSpec


class In(BaseModel):
    x: int

class Out(BaseModel):
    y: int


def make():
    reg = ToolRegistry()
    reg.register(ToolSpec("double", "doubles", In, Out, lambda i: {"y": i.x * 2}, {"research"}, rate_limit_per_min=2))
    return reg


def test_allowed_agent_works():
    assert make().invoke("double", "research", {"x": 2}) == '{"y": 4}'


def test_disallowed_agent_blocked():
    assert make().invoke("double", "writing", {"x": 2}).startswith("ERROR")


def test_rate_limit():
    reg = make()
    reg.invoke("double", "research", {"x": 1})
    reg.invoke("double", "research", {"x": 1})
    assert "Rate limit" in reg.invoke("double", "research", {"x": 1})