import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(override=True)

# ---------------- paths ----------------
BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE = BASE_DIR / "workspace"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))

# ---------------- Phase 1: orchestration ----------------
MAX_ATTEMPTS = 2                                                         # tries per subtask before escalating
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.6"))   # below this -> human review
MAX_PARALLEL = int(os.getenv("MAX_PARALLEL", "1"))                       # subtasks run at the same time
STRIP_FINAL_WRITING = os.getenv("STRIP_FINAL_WRITING", "1") == "1"

ROLE_MODELS = {
    "supervisor": (os.getenv("SUPERVISOR_PROVIDER", "anthropic"), os.getenv("SUPERVISOR_MODEL", "claude-sonnet-5-5")),
    "specialist": (os.getenv("SPECIALIST_PROVIDER", "openai"), os.getenv("SPECIALIST_MODEL", "gpt-4o-mini")),
    "reviewer": (os.getenv("REVIEWER_PROVIDER", "anthropic"), os.getenv("REVIEWER_MODEL", "claude-sonnet-5-5")),
}

# ---------------- Phase 2: memory ----------------
CHROMA_DIR = DATA_DIR / "chroma"

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
WORKING_MEMORY_BACKEND = os.getenv("WORKING_MEMORY_BACKEND", "auto")     # auto | redis | memory
WORKING_MEMORY_TTL_S = int(os.getenv("WORKING_MEMORY_TTL_S", "21600"))   # safety net: 6 hours

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")                   # ollama | hash
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434")

DEFAULT_USER_ID = os.getenv("MEMORY_USER_ID", "default")
ADMIN_API_KEY = os.getenv("ADMIN_API_KEY", "")

MEMORY_TOP_K = 2                                                         # memories retrieved per section
MEMORY_MIN_SIMILARITY = float(os.getenv("MEMORY_MIN_SIMILARITY", "0.35"))   # ignore weak matches
DEDUP_SIMILARITY = 0.92        # at or above this, a new memory reinforces the old one
USE_LLM_EXTRACTION = os.getenv("USE_LLM_EXTRACTION", "1") == "1"

# ---------------- Phase 3: human in the loop ----------------
REVIEW_DB = DATA_DIR / "review_queue.db"
CHECKPOINT_DB = DATA_DIR / "checkpoints.db"
PLAN_CONFIDENCE_THRESHOLD = float(os.getenv("PLAN_CONFIDENCE_THRESHOLD", "0.5"))
NOTIFY_BELOW = float(os.getenv("NOTIFY_BELOW", "0.8"))   # quality in [CONFIDENCE_THRESHOLD, NOTIFY_BELOW) -> notify only
MAX_HUMAN_ROUNDS = int(os.getenv("MAX_HUMAN_ROUNDS", "6"))
NOTIFY_WEBHOOK_URL = os.getenv("NOTIFY_WEBHOOK_URL", "")

# ---------------- Phase 4: observability ----------------
TRACE_DB = DATA_DIR / "traces.db"
REPLAY_DIR = DATA_DIR / "replays"
PRICING_FILE = Path(os.getenv("PRICING_FILE", str(DATA_DIR / "pricing.json")))
OTLP_ENDPOINT = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")   # optional: Jaeger / any OTLP collector
MAX_ATTR_CHARS = int(os.getenv("OBS_MAX_ATTR_CHARS", "20000"))
MAX_DECISION_CHARS = 200_000                                    # decision inputs/outputs must stay parseable

# ---------------- Phase 5: integration ----------------
DATABASE_URL = os.getenv("DATABASE_URL", "")                    # postgresql://... -> PostgreSQL; empty -> SQLite files
CHROMA_HOST = os.getenv("CHROMA_HOST", "")                      # set -> use a Chroma server instead of a local folder
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", REDIS_URL)
CELERY_ENABLED = os.getenv("CELERY_ENABLED", "0") == "1"        # resume runs on workers instead of a subprocess
ASYNC_SPECIALISTS = os.getenv("ASYNC_SPECIALISTS", "0") == "1"  # run specialists on Celery workers
SPECIALIST_TIMEOUT_S = int(os.getenv("SPECIALIST_TIMEOUT_S", "900"))
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"                  # scripted agents + fixture search results parseable                                    # decision inputs/outputs must stay parseable