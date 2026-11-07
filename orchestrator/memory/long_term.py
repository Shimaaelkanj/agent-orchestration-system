import logging
import math
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import chromadb
import numpy as np
from chromadb.config import Settings

from .. import config
from .embeddings import Embedder, get_embedder

log = logging.getLogger("orchestrator.memory.long_term")

KINDS = ("episode", "approach", "failure", "fact", "preference", "summary", "decision")
BASE_IMPORTANCE = {"episode": 1.0, "approach": 1.5, "failure": 1.2,
                   "fact": 1.2, "preference": 2.0, "summary": 1.8, "decision": 1.5}
HALF_LIFE_DAYS = {"episode": 30, "approach": 90, "failure": 45,
                  "fact": 180, "preference": 365, "summary": 180, "decision": 120}
_USER_RE = re.compile(r"^[A-Za-z0-9_.@-]{1,64}$")


def check_user(user_id: str) -> str:
    if not _USER_RE.match(user_id or ""):
        raise ValueError("user_id must be 1-64 characters: letters, digits, _ . @ -")
    return user_id


def importance(meta: dict, now: float | None = None) -> float:
    """base weight x usage boost x exponential time decay (measured from last access)."""
    now = now if now is not None else time.time()
    age_days = max(0.0, (now - float(meta.get("last_accessed", now))) / 86400)
    kind = meta.get("kind", "episode")
    boost = 1 + math.log1p(float(meta.get("access_count", 0)))
    return BASE_IMPORTANCE.get(kind, 1.0) * boost * 0.5 ** (age_days / HALF_LIFE_DAYS.get(kind, 60))


@dataclass
class Memory:
    id: str
    text: str
    kind: str
    meta: dict
    importance: float
    similarity: float | None = None
    score: float | None = None
    embedding: list | None = None


def _clean_meta(extra: dict | None) -> dict:
    out = {}
    for k, v in (extra or {}).items():
        if v is None:
            continue
        out[k] = v if isinstance(v, (str, int, float, bool)) else str(v)   # Chroma: primitives only
    return out


def _connect(path: Path | str | None):
    """A Chroma client: a server (docker-compose) or a local folder. Always returns a client."""
    settings = Settings(anonymized_telemetry=False)
    if path is None and config.CHROMA_HOST:                  # a Chroma server
        last = None
        for _ in range(30):                                  # the server may still be starting
            try:
                client = chromadb.HttpClient(host=config.CHROMA_HOST, port=config.CHROMA_PORT,
                                             settings=settings)
                client.heartbeat()
                return client
            except Exception as e:
                last = e
                time.sleep(1)
        raise RuntimeError(f"Chroma server {config.CHROMA_HOST}:{config.CHROMA_PORT} is not reachable: {last}")
    folder = Path(path or config.CHROMA_DIR)                 # a local folder
    folder.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(folder), settings=settings)


class LongTermMemory:
    def __init__(self, embedder: Embedder | None = None, path: Path | str | None = None) -> None:
        self.embedder = embedder or get_embedder()
        self._client = _connect(path)
        # One collection per embedder: vectors from different models must never be mixed.
        self._col = self._client.get_or_create_collection(
            name=f"memories_{self.embedder.name}",
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )

    # ---------- helpers ----------
    @staticmethod
    def _where(user_id: str, kinds=None) -> dict:
        conds: list[dict] = [{"user_id": user_id}]
        if kinds:
            conds.append({"kind": {"$in": list(kinds)}})
        return conds[0] if len(conds) == 1 else {"$and": conds}

    def _nearest(self, user_id, kind, vec):
        if self._col.count() == 0:
            return None
        res = self._col.query(query_embeddings=[vec], n_results=1, where=self._where(user_id, [kind]))
        if not res["ids"][0]:
            return None
        return res["ids"][0][0], 1.0 - res["distances"][0][0]

    def _bump(self, ids: list[str]) -> None:
        """Record that memories were used: more accesses means higher importance, fresher decay clock."""
        if not ids:
            return
        got = self._col.get(ids=ids, include=["metadatas"])
        now = time.time()
        metas = []
        for m in got["metadatas"]:
            m = dict(m)
            m["access_count"] = int(m.get("access_count", 0)) + 1
            m["last_accessed"] = now
            metas.append(m)
        self._col.update(ids=got["ids"], metadatas=metas)

    # ---------- write ----------
    def add(self, user_id: str, kind: str, text: str, extra: dict | None = None, dedup: bool = True) -> str:
        check_user(user_id)
        text = text.strip()
        if not text:
            raise ValueError("empty memory")
        vec = self.embedder.embed([text])[0]
        if dedup:
            near = self._nearest(user_id, kind, vec)
            if near and near[1] >= config.DEDUP_SIMILARITY:
                self._bump([near[0]])           # seen again: reinforce instead of duplicating
                return near[0]
        now = time.time()
        meta = {"user_id": user_id, "kind": kind, "created_at": now, "last_accessed": now,
                "access_count": 0, **_clean_meta(extra)}
        mid = uuid.uuid4().hex
        self._col.add(ids=[mid], documents=[text], embeddings=[vec], metadatas=[meta])
        return mid

    # ---------- read ----------
    def search(self, user_id: str, query: str, k: int | None = None, kinds=None,
               min_similarity: float | None = None, touch: bool = True) -> list[Memory]:
        check_user(user_id)
        k = k or config.MEMORY_TOP_K
        min_sim = config.MEMORY_MIN_SIMILARITY if min_similarity is None else min_similarity
        total = self._col.count()
        if total == 0:
            return []
        vec = self.embedder.embed([query])[0]
        res = self._col.query(query_embeddings=[vec], n_results=min(k * 3, total),
                              where=self._where(user_id, kinds))
        mems = []
        for mid, doc, meta, dist in zip(res["ids"][0], res["documents"][0],
                                        res["metadatas"][0], res["distances"][0]):
            sim = 1.0 - dist
            if sim < min_sim:
                continue
            imp = importance(meta)
            mems.append(Memory(mid, doc, meta.get("kind", "episode"), meta, imp, sim,
                               score=0.75 * sim + 0.25 * min(1.0, imp / 3)))
        mems.sort(key=lambda m: m.score, reverse=True)
        mems = mems[:k]
        if touch:
            self._bump([m.id for m in mems])
        return mems

    def fetch(self, user_id: str | None = None, kinds=None, with_embeddings: bool = False) -> list[Memory]:
        """List memories without touching them (dashboard, maintenance)."""
        include = ["documents", "metadatas"] + (["embeddings"] if with_embeddings else [])
        if user_id:
            where = self._where(check_user(user_id), kinds)
        else:
            where = {"kind": {"$in": list(kinds)}} if kinds else None
        got = self._col.get(where=where, include=include)
        embs = got.get("embeddings") if with_embeddings else None
        out = []
        for i, mid in enumerate(got["ids"]):
            meta = got["metadatas"][i]
            out.append(Memory(mid, got["documents"][i], meta.get("kind", "episode"), meta,
                              importance(meta),
                              embedding=np.asarray(embs[i]).tolist() if embs is not None else None))
        out.sort(key=lambda m: m.importance, reverse=True)
        return out

    def top(self, user_id: str, kinds, k: int = 3) -> list[Memory]:
        """Most important memories of given kinds, regardless of the query (e.g. preferences)."""
        mems = self.fetch(user_id, kinds)[:k]
        self._bump([m.id for m in mems])
        return mems

    # ---------- delete ----------
    def delete(self, user_id: str, memory_id: str) -> bool:
        got = self._col.get(ids=[memory_id], include=["metadatas"])
        if not got["ids"] or got["metadatas"][0].get("user_id") != user_id:
            return False
        self._col.delete(ids=[memory_id])
        return True

    def delete_ids(self, ids: list[str]) -> None:
        if ids:
            self._col.delete(ids=ids)

    def delete_user(self, user_id: str) -> int:
        ids = self._col.get(where={"user_id": check_user(user_id)})["ids"]
        if ids:
            self._col.delete(ids=ids)
        return len(ids)

    def users(self) -> list[str]:
        metas = self._col.get(include=["metadatas"])["metadatas"]
        return sorted({m["user_id"] for m in metas})

    def stats(self, user_id: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for m in self.fetch(user_id):
            counts[m.kind] = counts.get(m.kind, 0) + 1
        return counts