#!/usr/bin/env python3
"""Speed against quality for one kind of work, from agent benchmark results: the useful frontier.

Reads the JSON files tests/agent_bench.py writes (--out), keeps one kind of task (coding by
default: docs-line, bugfix, new-file, merge, ci-fix, html-app), and for each model gives the share
of those tasks it passed and the median seconds a task took (passed or not). A model is on the
frontier when no other is both at least as fast and at least as good (and better at one).
Writes a page with the chart (log time axis) and the table, and prints the table.

    python3 tools/frontier.py results/*.json --kind coding --out frontier.html
"""
import argparse
import html
import json
import math
import re
import statistics
import sys
from pathlib import Path

KINDS = {
    "coding": {"docs-line", "bugfix", "new-file", "merge", "ci-fix", "html-app"},
    "research": {"research"}, "law": {"law"}, "browser": {"browser"}, "chat": {"question"}, "vision": {"look"},
}


def model_name(file):
    """Qwen3.5-4B-Q4_K_M.gguf -> Qwen3.5 4B; omnicoder-9b-q4_k_m.gguf -> omnicoder 9b."""
    stem = re.sub(r"\.gguf$", "", file)
    stem = re.sub(r"[-.](UD-)?(I?Q\d\w*|BF16|F16|F32|MXFP4\w*)$", "", stem, flags=re.I)
    return stem.replace("-", " ")


def summarize(paths, kind):
    tasks = KINDS[kind]
    runs = {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        for r in d["rows"]:
            if r["task"] in tasks:
                runs.setdefault(model_name(d["model"]), []).append((bool(r["ok"]), r["seconds"]))
    rows = []
    for name, rs in runs.items():
        ok = sum(o for o, _ in rs)
        rows.append({"model": name, "passed": ok, "tasks": len(rs), "quality": ok / len(rs), "quality_pct": round(100 * ok / len(rs)),
                     "seconds": statistics.median(s for _, s in rs)})
    for r in rows:
        r["frontier"] = not any(
            o is not r and o["seconds"] <= r["seconds"] and o["quality"] >= r["quality"]
            and (o["seconds"] < r["seconds"] or o["quality"] > r["quality"]) for o in rows)
    return sorted(rows, key=lambda r: r["seconds"])


def table(rows):
    out = ["| Model | Passed | Quality | Median s per task | Frontier |", "|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r['model']} | {r['passed']}/{r['tasks']} | {r['quality']:.0%} | {r['seconds']:.0f} | {'yes' if r['frontier'] else ''} |")
    return "\n".join(out)


def page(rows, kind, note):
    W, H, L, R, T, B = 760, 440, 64, 24, 28, 56
    lo = 10 ** math.floor(math.log10(min(r["seconds"] for r in rows)))
    hi = 10 ** math.ceil(math.log10(max(r["seconds"] for r in rows)))
    X = lambda s: L + (math.log10(s) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (W - L - R)
    Y = lambda q: T + (1 - q) * (H - T - B)
    grid = []
    for q in (0, 0.25, 0.5, 0.75, 1):
        grid.append(f'<line class="grid" x1="{L}" x2="{W - R}" y1="{Y(q):.1f}" y2="{Y(q):.1f}"/>'
                    f'<text class="tick" x="{L - 8}" y="{Y(q) + 4:.1f}" text-anchor="end">{q:.0%}</text>')
    t = lo
    while t <= hi:
        for m in (1, 2, 5):
            v = t * m
            if lo <= v <= hi:
                grid.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{T}" y2="{H - B}"/>'
                            f'<text class="tick" x="{X(v):.1f}" y="{H - B + 18}" text-anchor="middle">{v:g} s</text>')
        t *= 10
    front = sorted((r for r in rows if r["frontier"]), key=lambda r: r["seconds"])
    # The frontier as a staircase: the best quality reachable within each time.
    path = ""
    for i, r in enumerate(front):
        x, y = X(r["seconds"]), Y(r["quality"])
        path += f"M{x:.1f} {y:.1f}" if i == 0 else f" H{x:.1f} V{y:.1f}"
    dots, labels = [], []
    for r in rows:
        cls = "front" if r["frontier"] else "other"
        tip = f"{r['model']}: {r['passed']}/{r['tasks']} {kind} tasks passed, median {r['seconds']:.0f} s per task"
        dots.append(f'<g class="dot {cls}"><title>{html.escape(tip)}</title>'
                    f'<circle class="hit" cx="{X(r["seconds"]):.1f}" cy="{Y(r["quality"]):.1f}" r="14"/>'
                    f'<circle class="mark" cx="{X(r["seconds"]):.1f}" cy="{Y(r["quality"]):.1f}" r="5"/></g>')
        labels.append((X(r["seconds"]), Y(r["quality"]), r))
    # Labels beside each dot, nudged down when two would overlap.
    placed = []
    for x, y, r in sorted(labels, key=lambda v: (v[1], v[0])):
        ly = y - 10
        while any(abs(ly - py) < 14 and abs(x - px) < 140 for px, py in placed):
            ly += 14
        placed.append((x, ly))
        anchor = "end" if x > W - 200 else "start"
        dx = -10 if anchor == "end" else 10
        labels_svg = f'<text class="label" x="{x + dx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{html.escape(r["model"])}</text>'
        dots.append(labels_svg)
    rows_html = "".join(
        f"<tr><td>{html.escape(r['model'])}</td><td>{r['passed']}/{r['tasks']}</td><td>{r['quality']:.0%}</td>"
        f"<td>{r['seconds']:.0f}</td><td>{'yes' if r['frontier'] else ''}</td></tr>" for r in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evolve LLM: speed against quality ({kind})</title>
<style>
.viz-root {{ color-scheme: light; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e; --grid: #e6e5e1; --series-1: #2a78d6; --muted: #a3a29c; }}
@media (prefers-color-scheme: dark) {{ :root:where(:not([data-theme="light"])) .viz-root {{ color-scheme: dark; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --grid: #2e2e2c; --series-1: #3987e5; --muted: #77766f; }} }}
:root[data-theme="dark"] .viz-root {{ color-scheme: dark; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7; --grid: #2e2e2c; --series-1: #3987e5; --muted: #77766f; }}
body {{ margin: 0; font: 15px/1.45 system-ui, sans-serif; }}
.viz-root {{ background: var(--surface-1); color: var(--text-primary); min-height: 100vh; padding: 24px; box-sizing: border-box; }}
main {{ max-width: 800px; margin: 0 auto; }}
h1 {{ font-size: 20px; margin: 0 0 4px; }}
p {{ color: var(--text-secondary); margin: 0 0 16px; }}
svg {{ width: 100%; height: auto; display: block; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.tick, .axis {{ fill: var(--text-secondary); font-size: 12px; }}
.label {{ fill: var(--text-primary); font-size: 13px; }}
.frontier {{ fill: none; stroke: var(--series-1); stroke-width: 2; stroke-linejoin: round; }}
.dot .hit {{ fill: transparent; }}
.dot .mark {{ stroke: var(--surface-1); stroke-width: 2; }}
.front .mark {{ fill: var(--series-1); }}
.other .mark {{ fill: var(--muted); }}
.dot:hover .mark {{ r: 7; }}
.legend {{ display: flex; gap: 16px; color: var(--text-secondary); font-size: 13px; margin: 8px 0 0; }}
.key {{ display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; vertical-align: -1px; }}
table {{ border-collapse: collapse; margin-top: 20px; width: 100%; font-size: 14px; }}
th, td {{ text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--grid); }}
th {{ color: var(--text-secondary); font-weight: 600; }}
</style></head>
<body><div class="viz-root"><main>
<h1>Speed against quality: {kind}</h1>
<p>Each dot is a model on this device: the share of the agent benchmark's {kind} tasks it passed, against the median time a task took. The line joins the frontier, the models no other beats on both. {html.escape(note)}</p>
<svg viewBox="0 0 {W} {H}" role="img" aria-label="Scatter of {kind} quality against median seconds per task">
{''.join(grid)}
<path class="frontier" d="{path}"/>
{''.join(dots)}
<text class="axis" x="{(L + W - R) / 2}" y="{H - 14}" text-anchor="middle">Median seconds per task (log scale, faster to the left)</text>
<text class="axis" transform="translate(16 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle">Tasks passed</text>
</svg>
<div class="legend"><span><span class="key" style="background: var(--series-1)"></span>On the frontier</span><span><span class="key" style="background: var(--muted)"></span>Beaten on both by another model</span></div>
<table><thead><tr><th>Model</th><th>Passed</th><th>Quality</th><th>Median s per task</th><th>Frontier</th></tr></thead><tbody>{rows_html}</tbody></table>
</main></div></body></html>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", nargs="+", type=Path)
    ap.add_argument("--kind", default="coding", choices=sorted(KINDS))
    ap.add_argument("--out", type=Path)
    ap.add_argument("--json", type=Path, help="also write {rows, meta}: the rows, this script and its inputs")
    ap.add_argument("--note", default="", help="a sentence under the title (where the runs come from)")
    a = ap.parse_args()
    rows = summarize(a.results, a.kind)
    if not rows:
        sys.exit(f"no {a.kind} tasks in those results")
    print(table(rows))
    if a.out:
        a.out.write_text(page(rows, a.kind, a.note))
    if a.json:
        meta = [{"label": "frontier.py", "value": Path(__file__).read_text(), "type": "code", "lang": "python"},
                {"label": "Input", "value": ", ".join(x.name for x in a.results)},
                {"label": "Tasks", "value": ", ".join(sorted(KINDS[a.kind]))}]
        a.json.write_text(json.dumps({"rows": rows, "meta": meta}, indent=1))


if __name__ == "__main__":
    main()
