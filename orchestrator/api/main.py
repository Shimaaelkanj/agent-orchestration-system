
import uuid
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from .. import config
from ..hitl.queue import get_queue
from ..obs.tracing import get_store
from ..worker.celery_app import app as celery_app

app = FastAPI(title="Agent orchestration API")


def auth(x_api_key: str | None = Header(default=None)):
    """Placeholder protection. Real authentication is out of scope for this project."""
    if config.ADMIN_API_KEY and x_api_key != config.ADMIN_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class TaskIn(BaseModel):
    task: str
    user_id: str | None = None
    review: bool = False
    approve_final: bool = False


class DecisionIn(BaseModel):
    action: Literal["approve", "modify", "reject", "take_over"]
    note: str = ""
    output: str | None = None
    args: dict | None = None
    plan: dict | None = None
    by: str = "api"


@app.get("/health")
def health():
    status = {"database": get_store().db.kind}
    try:
        import redis
        redis.Redis.from_url(config.REDIS_URL, socket_connect_timeout=2).ping()
        status["redis"] = "ok"
    except Exception as e:
        status["redis"] = f"unreachable: {str(e)[:60]}"
    return status


@app.post("/tasks", status_code=202, dependencies=[Depends(auth)])
def submit(body: TaskIn):
    task_id = uuid.uuid4().hex[:12]
    celery_app.send_task("orchestrator.run_task", task_id=task_id, queue="orchestration",
                         kwargs={"task": body.task, "user_id": body.user_id or config.DEFAULT_USER_ID,
                                 "review": body.review, "approve_final": body.approve_final, "task_id": task_id})
    return {"task_id": task_id}


@app.get("/tasks", dependencies=[Depends(auth)])
def list_tasks(limit: int = 50):
    return get_store().tasks(limit)


@app.get("/tasks/{task_id}", dependencies=[Depends(auth)])
def task_status(task_id: str):
    result = get_store().task_result(task_id)
    pending = [{"review_id": r["review_id"], "level": r["level"], "trigger": r["trigger"]}
               for r in get_queue().list_reviews("pending", limit=500) if r["task_id"] == task_id]
    return {"task_id": task_id, "outcome": result["status"], "final_answer": result["final_answer"],
            "reason": result["reason"], "pending_reviews": pending,
            "worker_state": celery_app.AsyncResult(task_id).state}


@app.get("/reviews", dependencies=[Depends(auth)])
def list_reviews(status: str = "pending"):
    return [{"review_id": r["review_id"], "task_id": r["task_id"], "level": r["level"], "trigger": r["trigger"],
             "reason": r["payload"].get("reason", "")} for r in get_queue().list_reviews(status, limit=200)]


@app.post("/reviews/{review_id}/decision", status_code=202, dependencies=[Depends(auth)])
def decide(review_id: str, body: DecisionIn):
    decision = {k: v for k, v in body.model_dump().items() if v not in (None, "")}
    if not get_queue().decide(review_id, decision):
        raise HTTPException(status_code=409, detail="Review not found or already decided")
    celery_app.send_task("orchestrator.resume", args=[review_id], queue="orchestration")
    return {"queued": True}