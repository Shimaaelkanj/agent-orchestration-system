
import argparse
import time

from .. import config
from . import get_ltm
from .management import consolidate, expire


def main() -> None:
    ap = argparse.ArgumentParser(prog="memory")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("list", "stats", "forget-user", "consolidate"):
        p = sub.add_parser(name)
        p.add_argument("--user", default=config.DEFAULT_USER_ID)
    p = sub.add_parser("search"); p.add_argument("query"); p.add_argument("--user", default=config.DEFAULT_USER_ID)
    sub.add_parser("expire")
    a = ap.parse_args()
    ltm = get_ltm()

    if a.cmd == "list":
        for m in ltm.fetch(a.user):
            last = time.strftime("%Y-%m-%d", time.localtime(float(m.meta.get("last_accessed", 0))))
            print(f"[{m.kind:10}] imp={m.importance:5.2f} used={int(m.meta.get('access_count', 0)):3} "
                  f"last={last} id={m.id[:8]}\n    {m.text[:200]!r}")
    elif a.cmd == "stats":
        print(ltm.stats(a.user))
    elif a.cmd == "search":
        for m in ltm.search(a.user, a.query, k=5, touch=False):
            print(f"{m.similarity:.2f} [{m.kind}] {m.text[:200]!r}")
    elif a.cmd == "forget-user":
        print("deleted", ltm.delete_user(a.user))
    elif a.cmd == "expire":
        print("expired", len(expire(ltm)))
    elif a.cmd == "consolidate":
        print("merged", consolidate(ltm, a.user))


if __name__ == "__main__":
    main()