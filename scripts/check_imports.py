
import importlib
import sys

MODULES = ["orchestrator.graph", "orchestrator.hitl.runner", "orchestrator.hitl.queue", "orchestrator.worker.tasks",
           "orchestrator.worker.remote", "orchestrator.api.main", "orchestrator.memory.api",
           "orchestrator.obs.replay", "orchestrator.obs.metrics", "orchestrator.db", "pandas", "streamlit"]

bad = 0
for name in MODULES:
    try:
        importlib.import_module(name)
        print("OK  ", name)
    except Exception as e:
        bad += 1
        print("FAIL", name, "->", type(e).__name__, e)
sys.exit(1 if bad else 0)