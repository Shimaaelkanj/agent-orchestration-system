
import logging
import time
from typing import Callable

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage

from ..llm import get_llm
from .long_term import LongTermMemory, importance

log = logging.getLogger("orchestrator.memory.management")
EXPIRE_BELOW = 0.15


def expire(ltm: LongTermMemory, user_id: str | None = None,
           threshold: float = EXPIRE_BELOW, now: float | None = None) -> list[str]:
    """Delete memories whose decayed importance fell below the threshold."""
    now = now if now is not None else time.time()
    gone = [m.id for m in ltm.fetch(user_id) if importance(m.meta, now) < threshold]
    ltm.delete_ids(gone)
    return gone


def llm_summarizer(kind: str, texts: list[str]) -> str:
    try:
        out = get_llm("supervisor").invoke([
            SystemMessage("Merge these notes into ONE concise note (max 3 sentences). Keep every distinct, "
                          "useful detail. Do not invent anything."),
            HumanMessage("\n---\n".join(texts)),
        ]).content
        if isinstance(out, list):
            out = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in out)
        if out and out.strip():
            return out.strip()
    except Exception as e:
        log.warning("summarizer failed, falling back to concatenation: %s", str(e)[:120])
    return " | ".join(dict.fromkeys(t.strip() for t in texts))[:1500]


def consolidate(ltm: LongTermMemory, user_id: str,
                summarizer: Callable[[str, list[str]], str] | None = None,
                threshold: float = 0.8, min_cluster: int = 3) -> int:
    """Merge clusters of similar memories into one higher-level 'summary' memory.
    Returns how many original memories were merged away."""
    summarizer = summarizer or llm_summarizer
    merged = 0
    for kind in ("episode", "approach", "failure", "fact"):    # preferences stay verbatim
        mems = ltm.fetch(user_id, [kind], with_embeddings=True)
        if len(mems) < min_cluster:
            continue
        E = np.array([m.embedding for m in mems], dtype=float)
        E = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
        used: set[int] = set()
        for i in range(len(mems)):
            if i in used:
                continue
            group = [i] + [j for j in range(i + 1, len(mems))
                           if j not in used and float(E[i] @ E[j]) >= threshold]
            if len(group) < min_cluster:
                continue
            used.update(group)
            summary = summarizer(kind, [mems[g].text for g in group])
            ltm.add(user_id, "summary", summary,
                    {"source_kind": kind, "merged_count": len(group),
                     "access_count": sum(int(mems[g].meta.get("access_count", 0)) for g in group)},
                    dedup=False)
            ltm.delete_ids([mems[g].id for g in group])
            merged += len(group)
    return merged