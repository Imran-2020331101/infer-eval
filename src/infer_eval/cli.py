"""infer-eval CLI.

  infer-eval list
  infer-eval show <id>          print the prompt a model would see
  infer-eval reference <id>     print the expert reference solution
  infer-eval grade <id> FILE    grade a response file ("-" for stdin)
  infer-eval demo               reference vs. trap responses for every scenario
"""
from __future__ import annotations

import argparse
import sys

from .grader import grade, reference_response, trap_response
from .scenarios import SCENARIOS


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="infer-eval", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("show", "reference"):
        sub.add_parser(name).add_argument("id", choices=SCENARIOS)
    g = sub.add_parser("grade")
    g.add_argument("id", choices=SCENARIOS)
    g.add_argument("file")
    sub.add_parser("demo")
    args = p.parse_args(argv)

    if args.cmd == "list":
        for s in SCENARIOS.values():
            print(f"{s.id:22} L{s.difficulty}  {s.title}  [{', '.join(s.topics)}]")
    elif args.cmd == "show":
        print(SCENARIOS[args.id].prompt)
    elif args.cmd == "reference":
        print(reference_response(SCENARIOS[args.id]))
    elif args.cmd == "grade":
        text = sys.stdin.read() if args.file == "-" else open(args.file).read()
        print(grade(SCENARIOS[args.id], text).render())
    elif args.cmd == "demo":
        for s in SCENARIOS.values():
            print(f"=== {s.id}: reference ===")
            print(grade(s, reference_response(s)).render())
            wrong = trap_response(s)
            if wrong:
                print(f"--- {s.id}: plausible-but-wrong ---")
                print(grade(s, wrong).render())
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
