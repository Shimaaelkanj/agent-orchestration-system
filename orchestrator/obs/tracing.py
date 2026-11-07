import contextlib
import functools
import hashlib
import json
import logging
import os
import time

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import messages_to_dict
from opentelemetry import context as otel_context
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

from .. import config
from . import pricing
from .store import SpanStore, SQLiteExporter

log = logging.getLogger("orchestrator.obs")

_state: dict = {"provider": None, "tracer": None, "store": None}
_RANK = {"success": 0, "warning": 1, "escalated": 2, "failure": 3}
_INTERRUPTS = {"GraphInterrupt", "NodeInterrupt"}


# ---------------- setup ----------------
class _TagProcessor(SpanProcessor):
    """Copies the task id and the acting agent (stored in the OpenTelemetry context) onto every span."""

    def on_start(self, span, parent_context=None):
        tid = otel_context.get_value("obs.task", parent_context)
        agent = otel_context.get_value("obs.agent", parent_context)
        if tid:
            span.set_attribute("task.id", tid)
        if agent:
            span.set_attribute("agent", agent)

    def on_end(self, span):
        pass

    def shutdown(self):
        pass

    def force_flush(self, timeout_millis: int = 30000):
        return True


def configure(db_path=None, otlp: bool = True) -> SpanStore:
    if db_path is None and os.getenv("PYTEST_CURRENT_TEST"):
        raise RuntimeError("Tracing tried to use the REAL database during a test. "
                           "tests/conftest.py must call tracing.configure(tmp_path / 'traces.db').")
    store = SpanStore(db_path)
    provider = TracerProvider(resource=Resource.create({"service.name": "agent-orchestrator"}))
    provider.add_span_processor(_TagProcessor())
    # Simple (synchronous) export: a process that pauses for a human may exit right after, and nothing may be lost.
    provider.add_span_processor(SimpleSpanProcessor(SQLiteExporter(store)))
    if otlp and config.OTLP_ENDPOINT:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        except Exception as e:
            log.warning("OTLP export disabled: %s", str(e)[:120])
    _state.update(provider=provider, tracer=provider.get_tracer("orchestrator"), store=store)
    return store


def reset() -> None:
    _state.update(provider=None, tracer=None, store=None)


def tracer():
    if _state["tracer"] is None:
        configure()
    return _state["tracer"]


def get_store() -> SpanStore:
    if _state["store"] is None:
        configure()
    return _state["store"]


# ---------------- deterministic ids: one trace per task, even across processes ----------------
def trace_id_for(task_id: str) -> int:
    return int(hashlib.sha256(f"trace:{task_id}".encode()).hexdigest()[:32], 16) or 1


def root_span_id_for(task_id: str) -> int:
    return int(hashlib.sha256(f"root:{task_id}".encode()).hexdigest()[:16], 16) or 1


def task_context(task_id: str):
    """Context whose current span is the task's (unrecorded) root span and which carries the task id."""
    sc = SpanContext(trace_id=trace_id_for(task_id), span_id=root_span_id_for(task_id), is_remote=True,
                     trace_flags=TraceFlags(TraceFlags.SAMPLED))
    ctx = trace.set_span_in_context(NonRecordingSpan(sc))
    return otel_context.set_value("obs.task", task_id, ctx)


# ---------------- helpers ----------------
def clean(attrs: dict, limit: int | None = None) -> dict:
    """OpenTelemetry attributes must be primitives: everything else becomes a JSON string."""
    limit = limit or config.MAX_ATTR_CHARS
    out = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float)):
            out[k] = v
        elif isinstance(v, str):
            out[k] = v[:limit]
        else:
            out[k] = json.dumps(v, default=str)[:limit]
    return out


def set_status(sp, status: str) -> None:
    """Colors: success < warning < escalated < failure. A span never goes back to a milder status."""
    cur = None
    try:
        cur = (sp.attributes or {}).get("obs.status")
    except Exception:
        pass
    if cur is None or _RANK.get(status, 0) >= _RANK.get(cur, 0):
        sp.set_attribute("obs.status", status)


@contextlib.contextmanager
def span(name: str, task: str | None = None, **attrs):
    # With a task id, the task context is ATTACHED (made current) first. Passing it as `context=` to
    # start_as_current_span would only choose the parent: children would then never see the task id.
    token = otel_context.attach(task_context(task)) if task else None
    try:
        with tracer().start_as_current_span(name, attributes=clean(attrs), record_exception=False,
                                            set_status_on_exception=False) as sp:
            try:
                yield sp
            except BaseException as e:
                status = getattr(e, "obs_status", None) or (
                    "escalated" if type(e).__name__ in _INTERRUPTS else "failure")
                set_status(sp, status)
                if status == "escalated":
                    sp.set_attribute("obs.waiting_for_human", True)
                else:
                    sp.set_attribute("error.type", type(e).__name__)
                    sp.set_attribute("error.message", str(e)[:2000])
                raise
            else:
                set_status(sp, "success")
    finally:
        if token is not None:
            otel_context.detach(token)


def node_span(name: str, task_id: str, **attrs):
    return span(name, task=task_id, **attrs)


def annotate(**attrs) -> None:
    sp = trace.get_current_span()
    if sp.is_recording():
        for k, v in clean(attrs).items():
            sp.set_attribute(k, v)


def event(name: str, **attrs) -> None:
    sp = trace.get_current_span()
    if sp.is_recording():
        sp.add_event(name, attributes=clean(attrs))


def mark(status: str) -> None:
    sp = trace.get_current_span()
    if sp.is_recording():
        set_status(sp, status)


@contextlib.contextmanager
def agent_scope(name: str):
    """Everything started inside is attributed to this agent (for the explorer and the cost report)."""
    token = otel_context.attach(otel_context.set_value("obs.agent", name))
    try:
        yield
    finally:
        otel_context.detach(token)


def propagate(fn):
    """Run fn in another thread with the CURRENT trace context (threads do not inherit it by themselves)."""
    ctx = otel_context.get_current()

    @functools.wraps(fn)
    def wrapper(*a, **k):
        token = otel_context.attach(ctx)
        try:
            return fn(*a, **k)
        finally:
            otel_context.detach(token)
    return wrapper


# ---------------- LLM calls: prompt, response, tokens, cost ----------------
def _as_text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


class ObsCallback(BaseCallbackHandler):
    """Attached to every model in llm.py: records one 'llm.call' span per model call."""

    def __init__(self, role: str, provider: str, model: str) -> None:
        self.role, self.provider, self.model = role, provider, model
        self._open: dict = {}

    def _is_local(self) -> bool:
        base = os.getenv("OPENAI_BASE_URL") or ""
        return self.provider == "openai" and any(h in base for h in ("localhost", "127.0.0.1"))

    def _start(self, run_id, prompt_json: str) -> None:
        sp = tracer().start_span("llm.call", attributes=clean({
            "llm.role": self.role, "llm.provider": self.provider, "llm.model": self.model,
            "llm.prompt": prompt_json}))
        self._open[run_id] = (sp, time.perf_counter())

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        try:
            prompt = json.dumps(messages_to_dict(messages[0]), default=str)
        except Exception:
            prompt = str(messages)
        self._start(run_id, prompt)

    def on_llm_start(self, serialized, prompts, *, run_id, **kwargs):
        self._start(run_id, json.dumps([{"type": "human", "data": {"content": p}} for p in prompts]))

    def on_llm_end(self, response, *, run_id, **kwargs):
        item = self._open.pop(run_id, None)
        if not item:
            return
        sp, t0 = item
        text, calls, usage = "", [], {}
        try:
            gen = response.generations[0][0]
            msg = getattr(gen, "message", None)
            text = _as_text(msg.content) if msg is not None else (getattr(gen, "text", "") or "")
            calls = list(getattr(msg, "tool_calls", None) or [])
            usage = dict(getattr(msg, "usage_metadata", None) or {})
            if not usage:
                tu = (response.llm_output or {}).get("token_usage") or {}
                usage = {"input_tokens": tu.get("prompt_tokens", 0), "output_tokens": tu.get("completion_tokens", 0)}
        except Exception:
            pass
        tin, tout = int(usage.get("input_tokens") or 0), int(usage.get("output_tokens") or 0)
        estimated = False
        if tin + tout == 0:      # some local servers return no usage: estimate (about 4 characters per token)
            prompt = str((sp.attributes or {}).get("llm.prompt", ""))
            tin, tout, estimated = len(prompt) // 4, len(text) // 4, True
        cost, priced = pricing.cost(self.model, tin, tout, local=self._is_local())
        for k, v in clean({"llm.response": text, "llm.tool_calls": calls or None, "llm.input_tokens": tin,
                           "llm.output_tokens": tout, "llm.total_tokens": tin + tout,
                           "llm.tokens_estimated": estimated, "llm.cost_usd": cost, "llm.priced": priced,
                           "llm.latency_s": round(time.perf_counter() - t0, 3)}).items():
            sp.set_attribute(k, v)
        set_status(sp, "success")
        sp.end()

    def on_llm_error(self, error, *, run_id, **kwargs):
        item = self._open.pop(run_id, None)
        if not item:
            return
        sp, _ = item
        set_status(sp, "failure")
        sp.set_attribute("error.type", type(error).__name__)
        sp.set_attribute("error.message", str(error)[:2000])
        sp.end()




def export_context() -> dict:
    """The current span and task/agent, as plain JSON, so another process can continue the same trace."""
    sc = trace.get_current_span().get_span_context()
    return {"trace_id": sc.trace_id, "span_id": sc.span_id,
            "task_id": otel_context.get_value("obs.task") or "", "agent": otel_context.get_value("obs.agent") or ""}


@contextlib.contextmanager
def import_context(d: dict):
    sc = SpanContext(trace_id=int(d["trace_id"]), span_id=int(d["span_id"]), is_remote=True,
                     trace_flags=TraceFlags(TraceFlags.SAMPLED))
    ctx = trace.set_span_in_context(NonRecordingSpan(sc))
    if d.get("task_id"):
        ctx = otel_context.set_value("obs.task", d["task_id"], ctx)
    if d.get("agent"):
        ctx = otel_context.set_value("obs.agent", d["agent"], ctx)
    token = otel_context.attach(ctx)
    try:
        yield
    finally:
        otel_context.detach(token)        