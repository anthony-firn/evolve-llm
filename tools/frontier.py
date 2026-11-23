#!/usr/bin/env python3
"""Coding quality against time per answer on this engine: the useful frontier.

Quality comes from an independent evaluator (docs/frontier.json names it), so only speed is
measured here. Speed alone flatters models that write a lot (thinking ones above all), so the
chart's speed is time per answer: the output tokens the evaluator saw a model write, over its
tokens per second here, relative to Qwen3.5 4B without thinking on the GPU (1x).

Each model gets one row per way it can run and per thinking mode:
- on the GPU: measured with `evolve-llm bench`, or, for a catalog model not run, worked out from
  its file size the way the measured ones behave (tokens/s x GB is nearly constant);
- streamed from the drive or over the network: a model's per-token reads (`evolve-llm plan`)
  divided by the measured read rate, unless the run itself was measured.

A row is on the frontier when no row that runs from a local copy answers at least as fast and
scores at least as well (and better at one). Network rows are left out of it: the same model
from a local copy always beats them. web/model-manager.mjs applies the same rule per device to
choose the app's models.

    python3 tools/frontier.py                   # the table
    python3 tools/frontier.py --json rows.json  # {rows, meta} for a chart
"""
import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SETUPS = {"gpu": "On the GPU", "drive": "From the drive", "network": "Over the network"}
REFERENCE = ("Qwen3.5 4B", "no_thinking")


def rows_of(data):
    models = data["models"]
    by_name = {m["name"]: m for m in models}
    # GB read per token x tokens/s for models measured on the GPU this engine runs on.
    k = statistics.median(m["tok_s"] * m["file_gb"] for m in models
                          if m.get("tok_s") and m.get("setup", "gpu") == "gpu" and "device" not in m)
    rates = data["read_rates_gb_s"]
    rows = []
    for m in models:
        base = {"model": m["name"], "file_gb": m["file_gb"], "reads_gb": m.get("reads_gb"), "note": m.get("note", "")}
        if m.get("tok_s"):
            runs = [(m.get("setup", "gpu"), m["tok_s"], True, m.get("device", ""))]
        elif m.get("same_as"):
            runs = [("gpu", by_name[m["same_as"]]["tok_s"], False, "")]
            base["note"] = f"Same tensor shapes and quantization as {m['same_as']}, measured"
        elif m.get("reads_gb"):
            runs = [(s, rates[s] / m["reads_gb"], False, "") for s in ("drive", "network")]
        else:
            runs = [("gpu", k / m["file_gb"], False, "")]
            base["note"] = base["note"] or "Speed from file size, as the measured models behave"
        modes = list((m.get("coding") or {}).items()) or [(None, None)]
        for setup, tok_s, measured, device in runs:
            for mode, q in modes:
                rows.append({**base, "setup": SETUPS[setup], "measured": measured, "device": device,
                             "tok_s": round(tok_s, 6 if tok_s < 0.01 else 2), "speed": pace(tok_s), "how": "measured" if measured else "estimated",
                             "thinking": None if mode is None else mode == "thinking",
                             "coding_mode": None if mode is None else mode.replace("_", " "),
                             "coding": None if q is None else q["coding"], "output_mtok": None if q is None else q.get("output_mtok")})
    ref = next(r for r in rows if r["model"] == REFERENCE[0] and r["coding_mode"] == REFERENCE[1].replace("_", " ") and r["setup"] == SETUPS["gpu"])
    unit = ref["output_mtok"] / ref["tok_s"]
    for r in rows:
        r["time"] = round(r["output_mtok"] / r["tok_s"] / unit, 4) if r["output_mtok"] and r["coding"] is not None else None
        r["time_text"] = times(r["time"])
    local = [r for r in rows if r["time"] is not None and r["setup"] != SETUPS["network"]]
    for r in rows:
        r["frontier"] = r in local and not any(
            o is not r and o["time"] <= r["time"] and o["coding"] >= r["coding"]
            and (o["time"] < r["time"] or o["coding"] > r["coding"]) for o in local)
    return sorted(rows, key=lambda r: (r["time"] is None, r["time"] or 0, -r["tok_s"]))


def times(x):
    if x is None:
        return ""
    return f"{x:.2f}x" if x < 10 else f"{x:,.0f}x"


def pace(tok_s):
    if tok_s >= 1:
        return f"{tok_s:.1f} tokens/s"
    s = 1 / tok_s
    return f"{s:.1f} s per token" if s < 60 else f"{s / 60:.0f} min per token" if s < 3600 else f"{s / 3600:.1f} h per token"


def table(rows):
    out = ["| Model | Runs | Thinking | Speed | Time per answer | Coding index | Frontier |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        q = "no score" if r["coding"] is None else f"{r['coding']:.1f}"
        speed = r["speed"] + ("" if r["measured"] else ", estimated")
        think = "" if r["thinking"] is None else ("yes" if r["thinking"] else "no")
        out.append(f"| {r['model']} | {r['setup']}{' (' + r['device'] + ')' if r['device'] else ''} | {think} | {speed} | {r['time_text'] or '-'} | {q} | {'yes' if r['frontier'] else ''} |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "docs/frontier.json")
    ap.add_argument("--json", type=Path, help="write {rows, meta}: the rows, this script and its input")
    a = ap.parse_args()
    data = json.loads(a.data.read_text())
    rows = rows_of(data)
    print(table(rows))
    if a.json:
        meta = [{"label": "frontier.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
                {"label": "Input", "value": a.data.read_text(), "type": "code", "lang": "json"},
                {"label": "Quality", "value": f"{data['quality']['name']}, {data['quality']['source']}"}]
        a.json.write_text(json.dumps({"rows": rows, "meta": meta}, indent=1))


if __name__ == "__main__":
    main()
