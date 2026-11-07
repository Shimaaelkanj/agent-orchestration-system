
import hashlib
import math
import re
from functools import lru_cache

import httpx

from .. import config


class Embedder:
    name = "base"

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class OllamaEmbedder(Embedder):
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.name = "ollama-" + re.sub(r"[^a-zA-Z0-9._-]", "-", model)

    def embed(self, texts):
        try:
            r = httpx.post(f"{self.base_url}/api/embed",
                           json={"model": self.model, "input": texts}, timeout=120)
            r.raise_for_status()
            return r.json()["embeddings"]
        except Exception as e:
            raise RuntimeError(f"Embedding failed ({e}). Run `ollama pull {self.model}` and make sure "
                               f"Ollama is running, or set EMBED_PROVIDER=hash.") from e


class HashEmbedder(Embedder):
    """Deterministic bag-of-words embedder. Offline and free, but only matches shared WORDS.
    Used in tests; fine as a fallback, weak for real semantic search."""

    name = "hash256"
    DIM = 256

    def embed(self, texts):
        out = []
        for t in texts:
            v = [0.0] * self.DIM
            for tok in re.findall(r"[a-z0-9]+", t.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                v[h % self.DIM] += 1.0 if (h >> 8) & 1 else -1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    if config.EMBED_PROVIDER == "hash":
        return HashEmbedder()
    return OllamaEmbedder(config.EMBED_MODEL, config.EMBED_BASE_URL)