
import logging
import time

import httpx

from .. import config

log = logging.getLogger("orchestrator.hitl")


def notify(item: dict) -> None:
    """Tell the human a review exists: log, notifications.log, and an optional webhook
    (for example https://ntfy.sh/<your-private-topic>). Best effort: never raises."""
    text = (f"[{item['level']}] {item['trigger']}: review {item['review_id']} "
            f"(task {item['task_id']}) - {str(item.get('reason', ''))[:160]}")
    log.warning("HUMAN REVIEW: %s", text)
    try:
        config.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        with (config.LOGS_DIR / "notifications.log").open("a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        url = config.NOTIFY_WEBHOOK_URL
        if url:
            if "hooks.slack.com" in url:
                httpx.post(url, json={"text": text}, timeout=5)
            else:
                httpx.post(url, content=text.encode("utf-8"), timeout=5)
    except Exception as e:
        log.warning("notification failed: %s", str(e)[:120])