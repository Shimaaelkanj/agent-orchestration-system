
import time

import pytest

from orchestrator import config
from orchestrator.memory.embeddings import HashEmbedder
from orchestrator.memory.long_term import LongTermMemory, importance
from orchestrator.memory.management import consolidate, expire
from orchestrator.memory.service import recall_for_planning, remember_task
from orchestrator.memory.working import WorkingMemory, _MemoryBackend


@pytest.fixture
def ltm(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MEMORY_MIN_SIMILARITY", 0.05)
    return LongTermMemory(HashEmbedder(), tmp_path / "chroma")


def test_working_memory_roundtrip_and_clear():
    wm = WorkingMemory(_MemoryBackend())
    wm.set_plan("t", {"a": 1})
    wm.set_output("t", "t1", "hello")
    wm.log_error("t", "t1", "bad")
    wm.note_write("t", "k", "v")
    assert wm.get_plan("t") == {"a": 1}
    assert wm.get_output("t", "t1") == "hello"
    assert wm.errors("t")[0]["error"] == "bad"
    assert wm.note_read("t", "k") == "v"
    wm.set_output("other", "t1", "isolated")
    wm.clear("t")
    assert wm.snapshot("t") == {"plan": None, "outputs": {}, "errors": [], "notes": {}}
    assert wm.get_output("other", "t1") == "isolated"          # other tasks untouched


def test_search_finds_relevant_memory(ltm):
    ltm.add("alice", "fact", "build a python prime checker")
    ltm.add("alice", "fact", "plan a trip to lisbon portugal")
    hits = ltm.search("alice", "python prime number function", k=1)
    assert hits and "prime" in hits[0].text


def test_users_are_isolated(ltm):
    ltm.add("alice", "fact", "alice secret project falcon")
    assert ltm.search("bob", "falcon project") == []


def test_duplicate_is_reinforced_not_duplicated(ltm):
    a = ltm.add("alice", "preference", "user prefers short answers")
    b = ltm.add("alice", "preference", "user prefers short answers")
    mems = ltm.fetch("alice")
    assert a == b and len(mems) == 1 and mems[0].meta["access_count"] == 1


def test_importance_grows_with_use_and_decays_with_age():
    now = time.time()
    fresh = {"kind": "episode", "last_accessed": now, "access_count": 0}
    used = {**fresh, "access_count": 10}
    old = {**fresh, "last_accessed": now - 90 * 86400}
    assert importance(used, now) > importance(fresh, now) > importance(old, now)


def test_expire_removes_stale_but_keeps_durable(ltm):
    ltm.add("alice", "episode", "old build task")
    ltm.add("alice", "preference", "likes bullet points")
    gone = expire(ltm, now=time.time() + 400 * 86400)
    assert len(gone) == 1
    assert [m.kind for m in ltm.fetch("alice")] == ["preference"]


def test_consolidate_merges_similar_memories(ltm):
    for tail in ("one", "two", "three"):
        ltm.add("alice", "episode",
                f"user asked for prime checker python function tested ten numbers {tail}", dedup=False)
    ltm.add("alice", "episode", "zzz qqq www completely unrelated", dedup=False)
    merged = consolidate(ltm, "alice", summarizer=lambda kind, texts: f"SUMMARY of {len(texts)}",
                         threshold=0.5, min_cluster=3)
    assert merged == 3
    assert sorted(m.kind for m in ltm.fetch("alice")) == ["episode", "summary"]


def test_delete_user_erases_everything(ltm):
    ltm.add("alice", "fact", "one thing")
    ltm.add("alice", "preference", "another thing")
    ltm.add("bob", "fact", "bob stays")
    assert ltm.delete_user("alice") == 2
    assert ltm.fetch("alice") == [] and len(ltm.fetch("bob")) == 1


def test_remember_and_recall_roundtrip(ltm):
    state = {"task": "write a prime checker", "task_id": "t",
             "plan": {"subtasks": [{"id": "1", "specialist": "code_execution",
                                    "description": "write and test prime function"}]},
             "results": {"1": {"status": "approved", "attempts": 2, "feedback": "print the results",
                               "error": None, "evidence": "TOOL run_python args={} -> {}", "output": "ok"}},
             "final_answer": "done"}
    counts = remember_task(ltm, "alice", state, outcome="done", use_llm=False)
    assert counts["episode"] == 1 and counts["failure"] == 1
    ctx = recall_for_planning(ltm, "alice", "write a prime checker")
    assert "prime" in ctx and "Things that went wrong" in ctx

def test_long_term_memory_works_through_a_chroma_server(monkeypatch, tmp_path):
    import chromadb
    from chromadb.config import Settings

    server = chromadb.PersistentClient(path=str(tmp_path / "server"), settings=Settings(anonymized_telemetry=False))
    monkeypatch.setattr(config, "CHROMA_HOST", "chroma")
    monkeypatch.setattr(chromadb, "HttpClient", lambda **kw: server)      # stands in for the real server
    ltm = LongTermMemory(HashEmbedder())                                  # no path: the server branch
    ltm.add("alice", "fact", "server mode works")
    assert [m.text for m in ltm.fetch("alice")] == ["server mode works"]    