
import argparse
import json
import logging

from . import runner
from .queue import ReviewQueue


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(prog="review")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("--status", default="pending")
    sub.add_parser("show").add_argument("review_id")
    sub.add_parser("resume").add_argument("review_id")
    p = sub.add_parser("decide")
    p.add_argument("review_id")
    p.add_argument("action", choices=["approve", "modify", "reject", "take_over"])
    p.add_argument("--note", default="")
    p.add_argument("--output", default="", help="take_over: your output; modify: the edited output")
    p.add_argument("--args", default="", help="modify a sensitive action: edited arguments as JSON")
    p.add_argument("--plan-file", default="", help="modify a plan: path to an edited plan JSON file")
    p.add_argument("--by", default="cli")
    p.add_argument("--resume", action="store_true", help="continue the run right away (in this terminal)")
    a = ap.parse_args()
    q = ReviewQueue()

    if a.cmd == "list":
        for i in q.list_reviews(None if a.status == "all" else a.status):
            print(f"{i['review_id']}  [{i['level']}] {i['trigger']}  status={i['status']}")
    elif a.cmd == "show":
        i = q.get(a.review_id)
        if not i:
            raise SystemExit("unknown review")
        p = i["payload"]
        print(f"Task: {p['task']}\nLevel: {i['level']}   Trigger: {i['trigger']}   Status: {i['status']}")
        print(f"Reason: {p['reason']}\n\nProgress:")
        for s in p["progress"]:
            print(f"  {s['id']} [{s['status']}] {s['description']}")
        print("\nProposed:", json.dumps(p["proposed"], indent=2, default=str)[:2500])
        if p["memories"]:
            print("\nRelevant memories:")
            for m in p["memories"]:
                print(f"  [{m['kind']}] {m['text']}")
    elif a.cmd == "decide":
        d: dict = {"action": a.action, "note": a.note, "by": a.by}
        if a.output:
            d["output"] = a.output
        if a.args:
            d["args"] = json.loads(a.args)
        if a.plan_file:
            d["plan"] = json.load(open(a.plan_file, encoding="utf-8"))
        if not q.decide(a.review_id, d):
            raise SystemExit("could not record: review missing or already decided")
        print("decision recorded")
        if a.resume:
            runner.print_outcome(runner.resume(a.review_id, queue=q))
    elif a.cmd == "resume":
        runner.print_outcome(runner.resume(a.review_id, queue=q))


if __name__ == "__main__":
    main()