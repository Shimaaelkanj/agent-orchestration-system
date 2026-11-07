
from celery import Celery

from .. import config

app = Celery("orchestrator", broker=config.CELERY_BROKER_URL, backend=config.CELERY_BROKER_URL,
             include=["orchestrator.worker.tasks"])
app.conf.update(
    task_serializer="json", result_serializer="json", accept_content=["json"], result_expires=3600,
    task_default_queue="orchestration", worker_prefetch_multiplier=1, task_track_started=True,
    broker_connection_retry_on_startup=True,
    task_routes={"orchestrator.run_task": {"queue": "orchestration"},
                 "orchestrator.resume": {"queue": "orchestration"},
                 "orchestrator.specialist": {"queue": "specialists"}},
)