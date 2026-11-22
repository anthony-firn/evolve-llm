#!/usr/bin/env python3
"""Serve Evolve LLM on this computer, with a live trace of the assistant's work.

    python3 tools/serve_local.py [--dir dist] [--port 8080]

Open http://localhost:8080/play/evolve-llm/ in Chrome. Every chat run, model turn (with the model's
raw output and token counts), tool call and result, and approval is appended to
~/.cache/evolve-llm/traces/YYYY-MM-DD.jsonl, so it can be followed while it runs:

    python3 tools/trace_view.py --follow

Only this computer can connect (127.0.0.1). The page sends its trace only when it was served from
this computer by this server; the public site and plain static hosting never trace.
"""
import argparse
import datetime
import http.server
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
from browser_test import Handler  # noqa: E402  (the production headers and the /play/evolve-llm/ path)

TRACES = Path.home() / ".cache" / "evolve-llm" / "traces"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, default=ROOT / "dist", help="the built app (tools/build_web.py writes dist/)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--trace", type=Path, help="trace file (default: today's file in ~/.cache/evolve-llm/traces)")
    a = ap.parse_args()
    trace = a.trace or TRACES / f"{datetime.date.today()}.jsonl"
    trace.parent.mkdir(parents=True, exist_ok=True)
    Handler.trace_file = str(trace)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", a.port), lambda *x, **k: Handler(*x, directory=str(a.dir), **k))
    print(f"Evolve LLM: http://localhost:{a.port}/play/evolve-llm/\ntrace: {trace}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
