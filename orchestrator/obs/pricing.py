
import json

from .. import config

# USD per 1 million tokens: (input, output). EXAMPLE VALUES ONLY: verify them on your provider's pricing page.
# Override or extend them in data/pricing.json, for example {"my-model": [1.0, 4.0]}.
DEFAULT_PRICES = {"gpt-4o-mini": (0.15, 0.60), "gpt-4o": (2.50, 10.00)}


def _table() -> dict:
    table = dict(DEFAULT_PRICES)
    try:
        if config.PRICING_FILE.exists():
            for name, v in json.loads(config.PRICING_FILE.read_text(encoding="utf-8")).items():
                table[name] = (float(v[0]), float(v[1]))
    except Exception:
        pass
    return table


def cost(model: str, tokens_in: int, tokens_out: int, local: bool = False) -> tuple[float, bool]:
    """Returns (cost in USD, priced). Local models (Ollama) are free; unknown models are flagged unpriced."""
    if local:
        return 0.0, True
    table = _table()
    best = max((k for k in table if model.startswith(k)), key=len, default=None)
    if best is None:
        return 0.0, False
    p_in, p_out = table[best]
    return (tokens_in * p_in + tokens_out * p_out) / 1_000_000, True