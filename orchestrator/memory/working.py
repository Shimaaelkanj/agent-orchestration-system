
import contextvars
import json
import logging
import threading
import time

from .. import config

log = logging.getLogger("orchestrator.memory.working")

# The task the current worker thread is executing; memory tools read it.
CURRENT_TASK: contextvars.ContextVar = contextvars.ContextVar("current_task", default=None)

SECTIONS = ("plan", "outputs", "notes", "errors")


class _MemoryBackend:
    """Process-local fallback with the same interface as Redis."""

    def __init__(self) -> None:
        self._h: dict[str, dict[str, str]] = {}
        self._l: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    def hset(self, key, field, value):
        with self._lock:
            self._h.setdefault(key, {})[field] = value

    def hget(self, key, field):
        with self._lock:
            return self._h.get(key, {}).get(field)

    def hgetall(self, key):
        with self._lock:
            return dict(self._h.get(key, {}))

    def rpush(self, key, value):
        with self._lock:
            self._l.setdefault(key, []).append(value)

    def lrange(self, key):
        with self._lock:
            return list(self._l.get(key, []))

    def delete(self, *keys):
        with self._lock:
            for k in keys:
                self._h.pop(k, None)
                self._l.pop(k, None)


class _RedisBackend:
    def __init__(self, url: str, ttl: int) -> None:
        import redis
        self.r = redis.Redis.from_url(url, decode_responses=True,
                                      socket_connect_timeout=1, socket_timeout=2)
        self.r.ping()          # fail fast if Redis is not running
        self.ttl = ttl

    def hset(self, key, field, value):
        self.r.hset(key, field, value)
        self.r.expire(key, self.ttl)

    def hget(self, key, field):
        return self.r.hget(key, field)

    def hgetall(self, key):
        return self.r.hgetall(key)

    def rpush(self, key, value):
        self.r.rpush(key, value)
        self.r.expire(key, self.ttl)

    def lrange(self, key):
        return self.r.lrange(key, 0, -1)

    def delete(self, *keys):
        self.r.delete(*keys)


def make_backend():
    mode = config.WORKING_MEMORY_BACKEND
    if mode in ("redis", "auto"):
        try:
            return _RedisBackend(config.REDIS_URL, config.WORKING_MEMORY_TTL_S), "redis"
        except Exception as e:
            if mode == "redis":
                raise
            log.warning("Redis unavailable (%s); using in-process working memory", str(e)[:100])
    return _MemoryBackend(), "memory"


class WorkingMemory:
    """Per-task shared store: plan, subtask outputs, scratch notes, error log."""

    def __init__(self, backend=None) -> None:
        if backend is None:
            backend, self.kind = make_backend()
        else:
            self.kind = type(backend).__name__
        self._b = backend

    @staticmethod
    def _k(task_id: str, section: str) -> str:
        return f"wm:{task_id}:{section}"

    # plan
    def set_plan(self, task_id, plan: dict):
        self._b.hset(self._k(task_id, "plan"), "plan", json.dumps(plan))

    def get_plan(self, task_id):
        raw = self._b.hget(self._k(task_id, "plan"), "plan")
        return json.loads(raw) if raw else None

    # subtask outputs
    def set_output(self, task_id, subtask_id, output: str):
        self._b.hset(self._k(task_id, "outputs"), subtask_id, output)

    def get_output(self, task_id, subtask_id):
        return self._b.hget(self._k(task_id, "outputs"), subtask_id)

    def outputs(self, task_id):
        return self._b.hgetall(self._k(task_id, "outputs"))

    # error log
    def log_error(self, task_id, subtask_id, message: str):
        self._b.rpush(self._k(task_id, "errors"),
                      json.dumps({"subtask": subtask_id, "error": message, "ts": time.time()}))

    def errors(self, task_id):
        return [json.loads(x) for x in self._b.lrange(self._k(task_id, "errors"))]

    # scratch notes (written by agents through the note_write tool)
    def note_write(self, task_id, key, value):
        self._b.hset(self._k(task_id, "notes"), key, value)

    def note_read(self, task_id, key):
        return self._b.hget(self._k(task_id, "notes"), key)

    def notes(self, task_id):
        return self._b.hgetall(self._k(task_id, "notes"))

    def snapshot(self, task_id) -> dict:
        return {"plan": self.get_plan(task_id), "outputs": self.outputs(task_id),
                "errors": self.errors(task_id), "notes": self.notes(task_id)}

    def clear(self, task_id):
        self._b.delete(*[self._k(task_id, s) for s in SECTIONS])