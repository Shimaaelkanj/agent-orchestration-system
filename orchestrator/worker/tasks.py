
from ..schemas import Subtask
from .celery_app import app


@app.task(name="orchestrator.run_task")
def run_task(task, user_id=None, review=False, approve_final=False, task_id=None):
    """Runs a whole task until it finishes or pauses for a human."""
    from ..hitl import runner
    out = runner.start(task, user_id=user_id, review=review, approve_final=approve_final, task_id=task_id)
    return {"state": out["state"], "task_id": out["task_id"], "review_id": out["review_id"]}


@app.task(name="orchestrator.resume")
def resume_task(review_id):
    """Continues a paused task after a human decided."""
    from ..hitl import runner
    out = runner.resume(review_id)
    return {"state": out["state"], "task_id": out["task_id"], "review_id": out["review_id"]}


@app.task(name="orchestrator.specialist")
def specialist_task(inputs, ctx):
    """One specialist attempt, on a worker of its own. Exceptions travel back as data, not as crashes."""
    from .. import graph
    from ..memory.working import CURRENT_TASK
    from ..obs import decisions
    from ..obs.replay_core import encode_exc
    from ..obs.tracing import import_context

    with import_context(ctx):
        CURRENT_TASK.set(ctx.get("task_id") or None)
        try:
            report, evidence = decisions._run_specialist(Subtask.model_validate(inputs["subtask"]),
                                                         inputs["dep_outputs"], inputs["feedback"], graph.registry)
            return {"report": report.model_dump(), "evidence": evidence}
        except Exception as e:
            return encode_exc(e)