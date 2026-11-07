
import subprocess
import sys
from pathlib import Path

from .. import config


def resume_in_background(review_id: str) -> Path:
    """Continue a decided review without blocking the UI: on a Celery worker when workers exist,
    otherwise in a separate process."""
    config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = config.LOGS_DIR / f"resume_{review_id}.log"
    if config.CELERY_ENABLED:
        from ..worker.celery_app import app
        app.send_task("orchestrator.resume", args=[review_id], queue="orchestration")
        return log_path
    with open(log_path, "ab") as logf:
        subprocess.Popen([sys.executable, "-m", "orchestrator.hitl.runner", "resume", review_id],
                         cwd=str(config.BASE_DIR), stdout=logf, stderr=subprocess.STDOUT)
    return log_path