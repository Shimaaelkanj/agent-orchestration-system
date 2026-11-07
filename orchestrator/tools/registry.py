import json
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Callable

from langchain_core.tools import StructuredTool
from pydantic import BaseModel

from ..config import LOGS_DIR
from ..obs.tracing import set_status, span

log = logging.getLogger("orchestrator.tools")


class ApprovalRequired(Exception):
    """Raised BEFORE a sensitive tool runs. Nothing has happened yet; a human must approve first."""

    obs_status = "escalated"       # lets the tracing layer color this span as 'escalated', not 'failure'

    def __init__(self, tool: str, agent: str, args: dict) -> None:
        super().__init__(f"Human approval required for tool '{tool}'")
        self.tool, self.agent = tool, agent
        self.tool_args = args      # NOT `self.args`: BaseException.args is special and turns a dict into a tuple of keys
        self.reasoning = ""


@dataclass
class ToolSpec:
    name: str
    description: str
    input_schema: type[BaseModel]
    output_schema: type[BaseModel]
    func: Callable[[Any], Any]          # receives a validated input_schema instance
    allowed_agents: set[str]
    rate_limit_per_min: int = 30
    sensitive: Any = False              # bool, or callable(parsed_input) -> bool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}
        self._calls: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._log_file = LOGS_DIR / "tool_calls.jsonl"

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def _rate_ok(self, spec: ToolSpec) -> bool:
        now = time.monotonic()
        with self._lock:
            q = self._calls[spec.name]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= spec.rate_limit_per_min:
                return False
            q.append(now)
            return True

    @staticmethod
    def _is_sensitive(spec: ToolSpec, parsed: BaseModel) -> bool:
        s = spec.sensitive
        return bool(s(parsed)) if callable(s) else bool(s)

    def _write_log(self, record: dict) -> None:
        with self._lock, self._log_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def invoke(self, name: str, agent: str, args: dict, approved: bool = False) -> str:
        spec = self._tools[name]
        start = time.perf_counter()
        output, error = None, None
        with span("tool.call", **{"tool.name": name, "tool.agent": agent, "tool.args": args,
                                  "tool.approved": approved}) as sp:
            try:
                if agent not in spec.allowed_agents:
                    raise PermissionError(f"Agent '{agent}' may not use tool '{name}'")
                parsed = spec.input_schema(**args)
                if not approved and self._is_sensitive(spec, parsed):
                    raise ApprovalRequired(name, agent, args)      # nothing has run yet
                if not self._rate_ok(spec):
                    raise RuntimeError(f"Rate limit exceeded for '{name}'")
                raw = spec.func(parsed)
                output = spec.output_schema.model_validate(raw).model_dump()
            except ApprovalRequired:
                self._write_log({"ts": time.time(), "tool": name, "agent": agent, "input": args, "output": None,
                                 "success": False, "error": "approval_required", "approved": False,
                                 "latency_ms": round((time.perf_counter() - start) * 1000, 1)})
                log.warning("tool=%s agent=%s needs human approval", name, agent)
                sp.set_attribute("tool.success", False)
                raise
            except Exception as e:  # tools must never crash the agent loop
                error = f"{type(e).__name__}: {e}"

            latency_ms = round((time.perf_counter() - start) * 1000, 1)
            self._write_log({"ts": time.time(), "tool": name, "agent": agent, "input": args,
                             "output": output, "success": error is None, "error": error,
                             "approved": approved, "latency_ms": latency_ms})
            log.info("tool=%s agent=%s ok=%s%s %.0fms", name, agent, error is None,
                     " (human-approved)" if approved else "", latency_ms)
            sp.set_attribute("tool.success", error is None)
            sp.set_attribute("tool.latency_ms", latency_ms)
            if error is None:
                sp.set_attribute("tool.output", json.dumps(output, default=str)[:20000])
            else:
                sp.set_attribute("error.message", error[:2000])
                set_status(sp, "warning")      # the agent can see the error and recover
            return json.dumps(output) if error is None else f"ERROR: {error}"

    def tools_for(self, agent: str) -> list[StructuredTool]:
        return [self._as_langchain(s, agent) for s in self._tools.values() if agent in s.allowed_agents]

    def _as_langchain(self, spec: ToolSpec, agent: str) -> StructuredTool:
        def _run(**kwargs: Any) -> str:
            return self.invoke(spec.name, agent, kwargs)
        return StructuredTool.from_function(
            func=_run, name=spec.name, description=spec.description, args_schema=spec.input_schema
        )