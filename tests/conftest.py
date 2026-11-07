import pytest

from orchestrator import config
from orchestrator.obs import decisions, replay_core, tracing


@pytest.fixture(autouse=True)
def _isolated_observability(tmp_path, monkeypatch):
    """Every test gets its own empty trace database and a clean replay state."""
    monkeypatch.setattr(config, "TRACE_DB", tmp_path / "traces.db")
    tracing.configure(tmp_path / "traces.db", otlp=False)
    decisions.reset()
    replay_core.set_active(None)
    yield
    replay_core.set_active(None)
    decisions.reset()
    tracing.reset()