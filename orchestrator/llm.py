import json
import os
import re

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from .config import ROLE_MODELS


def get_llm(role: str) -> BaseChatModel:
    provider, model = ROLE_MODELS[role]
    from .obs.tracing import ObsCallback
    callbacks = [ObsCallback(role, provider, model)]
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        base_url = os.getenv("OPENAI_BASE_URL") or None   # None = real OpenAI
        return ChatOpenAI(model=model, temperature=0, base_url=base_url, max_retries=5,
                          timeout=300, callbacks=callbacks)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        # no temperature: newer Claude models reject non-default values
        return ChatAnthropic(model=model, max_tokens=4096, callbacks=callbacks)
    raise ValueError(f"Unknown provider: {provider}")


def _text(content) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content or "")


def _extract_json(text: str):
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("the model's answer contains no JSON object")
    return json.loads(text[start:end + 1])


class _WithFallback:
    """Structured output for weak local models: constrained decoding first, then plain JSON in text."""

    def __init__(self, llm, schema, primary) -> None:
        self.llm, self.schema, self.primary = llm, schema, primary

    def invoke(self, messages, *args, **kwargs):
        try:
            result = self.primary.invoke(messages, *args, **kwargs)
            if result is not None:
                return result
        except Exception:
            pass
        ask = HumanMessage("Reply with ONLY one JSON object that matches this JSON schema. "
                           "No prose, no markdown.\n" + json.dumps(self.schema.model_json_schema()))
        text = _text(self.llm.invoke(list(messages) + [ask]).content)
        return self.schema.model_validate(_extract_json(text))


def structured(llm: BaseChatModel, role: str, schema: type[BaseModel]):
    provider, _ = ROLE_MODELS[role]
    if provider == "anthropic":
        return llm.with_structured_output(schema, method="json_schema")
    if os.getenv("OPENAI_BASE_URL"):                       # local server such as Ollama
        primary = llm.with_structured_output(schema, method="json_schema")
        return _WithFallback(llm, schema, primary)
    return llm.with_structured_output(schema)