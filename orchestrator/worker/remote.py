
from .. import config
from ..obs.replay_core import rebuild_exc
from ..obs.tracing import export_context
from ..schemas import SpecialistReport


def run_remote(inputs: dict):
    """Send one specialist attempt to the specialists queue and wait for the result.
    The specialists run in a different worker pool than the orchestration tasks, so waiting cannot deadlock."""
    from .celery_app import app
    result = app.send_task("orchestrator.specialist", args=[inputs, export_context()], queue="specialists")
    out = result.get(timeout=config.SPECIALIST_TIMEOUT_S, disable_sync_subtasks=False)
    if isinstance(out, dict) and "__exc__" in out:
        raise rebuild_exc(out)
    return SpecialistReport.model_validate(out["report"]), out["evidence"]