#!/usr/bin/env python3
"""Read the assistant's trace (tools/serve_local.py) as one line per event.

    python3 tools/trace_view.py [FILE] [--follow] [--full]

Default FILE: today's ~/.cache/evolve-llm/traces/YYYY-MM-DD.jsonl. --follow keeps reading as the
chat runs; --full prints whole model outputs and tool results instead of their first line.
"""
import argparse
import datetime
import json
import time
from pathlib import Path

TRACES = Path.home() / ".cache" / "evolve-llm" / "traces"


def short(text, full):
    text = str(text or "")
    if full:
        return "\n      " + text.replace("\n", "\n      ") if "\n" in text else text
    first = text.strip().splitlines()[0] if text.strip() else ""
    return first[:160] + (" …" if len(text.strip()) > len(first) or len(first) > 160 else "")


def line(e, full):
    t = e.get("time", "")[11:19]
    kind = e.get("type")
    if kind in ("run", "chat"):
        return f"\n{t} ▶ {e.get('model')}: {short(e.get('question'), True)}"
    if kind == "turn":
        return f"{t}   model {e.get('ms', 0) / 1000:.1f}s, {e.get('promptTokens', '?')} prompt + {e.get('tokens', '?')} tokens ({e.get('reason')}): {short(e.get('text'), full)}"
    if kind == "turn-error":
        return f"{t}   model stopped: {e.get('error')}"
    if kind == "tool":
        return f"{t}   → {e.get('name')} {e.get('args')}"
    if kind == "result":
        return f"{t}   {'✗' if e.get('error') else '←'} {short(e.get('output'), full)}"
    if kind == "retry":
        return f"{t}   (invalid tool call, asked again) {short(e.get('output'), full)}"
    if kind == "ask":
        return f"{t}   ? {e.get('question')} → {'allowed' if e.get('allowed') else 'declined'}"
    if kind in ("done", "reply"):
        extra = f", {e.get('calls')} steps" if kind == "done" else ""
        problem = f", stopped: {e.get('error')}" if e.get("error") else ""
        return f"{t} ■ {e.get('ms', 0) / 1000:.0f}s{extra}{problem}: {short(e.get('answer') or e.get('text'), full)}"
    return f"{t}   {json.dumps(e)[:200]}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", nargs="?", type=Path, default=TRACES / f"{datetime.date.today()}.jsonl")
    ap.add_argument("--follow", "-f", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    a.file.parent.mkdir(parents=True, exist_ok=True)
    a.file.touch()
    with a.file.open(encoding="utf-8") as f:
        while True:
            raw = f.readline()
            if raw:
                try:
                    print(line(json.loads(raw), a.full), flush=True)
                except ValueError:
                    pass
            elif a.follow:
                time.sleep(0.5)
            else:
                break


if __name__ == "__main__":
    main()
