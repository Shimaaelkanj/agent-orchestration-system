
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from streamlit.testing.v1 import AppTest

from orchestrator import config
from orchestrator.obs import tracing, tree

UI = str(Path(__file__).resolve().parents[1] / "orchestrator" / "obs" / "ui.py")


@pytest.fixture
def seeded(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REVIEW_DB", tmp_path / "q.db")
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replays")
    monkeypatch.setattr(config, "PRICING_FILE", tmp_path / "none.json")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    with tracing.node_span("node.intake", "UI1"):
        tracing.annotate(**{"task.text": "find primes", "user.id": "alice"})
    with tracing.node_span("node.execute", "UI1"):
        cb = tracing.ObsCallback("specialist", "openai", "gpt-4o-mini")
        with tracing.agent_scope("specialist:code_execution"):
            cb.on_chat_model_start({}, [[HumanMessage("PROMPT-TEXT")]], run_id="r1")
            msg = AIMessage(content="RESPONSE-TEXT", usage_metadata={"input_tokens": 10, "output_tokens": 5,
                                                                      "total_tokens": 15})
            cb.on_llm_end(LLMResult(generations=[[ChatGeneration(message=msg)]]), run_id="r1")
        with tracing.span("tool.call", **{"tool.name": "run_python", "tool.success": False}) as sp:
            tracing.set_status(sp, "warning")
    with tracing.node_span("node.delivery", "UI1"):
        tracing.annotate(**{"result.status": "done", "result.final_answer": "ANSWER"})


def open_ui():
    at = AppTest.from_file(UI, default_timeout=30).run()
    assert not at.exception, at.exception
    return at


def test_empty_store_shows_a_message(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "REVIEW_DB", tmp_path / "q.db")
    at = open_ui()
    assert any("No traces yet" in i.value for i in at.info)


def test_ui_renders_the_three_tabs_and_the_summary(seeded):
    at = open_ui()
    assert {"Trace explorer", "Cost and performance", "Replay"} <= {t.label for t in at.tabs}
    assert len(at.metric) >= 6


def test_selecting_an_llm_span_shows_prompt_and_response(seeded):
    at = open_ui()
    order = tree.build_tree(tracing.get_store().spans("UI1"))
    i = next(i for i, (_, s) in enumerate(order) if s["name"] == "llm.call")
    at.selectbox(key="span_pick").select(i).run()
    assert not at.exception, at.exception
    code = " ".join(c.value for c in at.code)
    assert "PROMPT-TEXT" in code and "RESPONSE-TEXT" in code


def test_selecting_a_failed_tool_span_works(seeded):
    at = open_ui()
    order = tree.build_tree(tracing.get_store().spans("UI1"))
    i = next(i for i, (_, s) in enumerate(order) if s["name"] == "tool.call")
    at.selectbox(key="span_pick").select(i).run()
    assert not at.exception, at.exception