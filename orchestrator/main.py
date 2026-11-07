import argparse
import logging

from . import config
from .hitl import runner


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="?", help="Task for the supervisor")
    ap.add_argument("--user", default=config.DEFAULT_USER_ID, help="User id that owns the memories")
    ap.add_argument("--review", action="store_true",
                    help="Ask a human to approve the plan before any work starts")
    ap.add_argument("--approve-final", action="store_true",
                    help="Ask a human to approve the final deliverable before it is delivered")
    ap.add_argument("--diagram", action="store_true", help="Print the graph as Mermaid")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.diagram:
        from .graph import build_graph
        print(build_graph().get_graph().draw_mermaid())
        return
    if not args.task:
        ap.error("task is required")

    runner.print_outcome(runner.start(args.task, user_id=args.user, review=args.review,
                                      approve_final=args.approve_final))


if __name__ == "__main__":
    main()