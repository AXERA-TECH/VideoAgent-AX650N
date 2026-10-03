"""CLI:  python -m VideoAgent status | index <video>... | ask "<question>" [--video NAME]... | rechapter | remove"""
import argparse
import json
import logging
import sys

from . import VideoAgent
from .media import fmt_time


def main():
    ap = argparse.ArgumentParser(prog="python -m VideoAgent")
    ap.add_argument("--working-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    p = sub.add_parser("index")
    p.add_argument("videos", nargs="+")
    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--video", action="append")
    p = sub.add_parser("rechapter", help="recompute chapter summaries of indexed videos")
    p.add_argument("names", nargs="*")
    p = sub.add_parser("remove")
    p.add_argument("name")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(levelname)s %(message)s")

    agent = VideoAgent(args.working_dir)
    if args.cmd == "status":
        print(json.dumps(agent.status(), ensure_ascii=False, indent=2))
    elif args.cmd == "index":
        agent.index(args.videos, progress=lambda f, m, data=None, stage="": print(f"[{f * 100:5.1f}%] {m}", flush=True))
    elif args.cmd == "rechapter":
        for name in args.names or agent.store.names():
            agent.indexer.rebuild_chapters(name, progress=lambda lv, m: print(m, flush=True))
    elif args.cmd == "remove":
        agent.store.remove_video(args.name)
    elif args.cmd == "ask":
        streamed = False
        for ev in agent.ask(args.question, args.video):
            if ev["type"] == "status":
                print(f"… {ev['text']}", file=sys.stderr, flush=True)
            elif ev["type"] == "delta":
                print(ev["text"], end="", flush=True)
                streamed = True
            elif ev["type"] == "error":
                print(f"\n错误: {ev['text']}", file=sys.stderr)
                sys.exit(1)
            elif ev["type"] == "done":
                print() if streamed else print(ev["answer"])
                print(f"\n[mode={ev['mode']}]", file=sys.stderr)
                for r in ev["refs"]:
                    print(f"  [{r['n']}] {r['video']} {fmt_time(r['start'])}-{fmt_time(r['end'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
