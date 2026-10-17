"""Print the per-run table of a recording as Markdown, or with --write replace the table between the markers in docs/results.md.

uv run python scripts/results_table.py [log] [--write]   (log defaults to demo/replay.jsonl)"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
START, END = "<!-- results-table -->", "<!-- /results-table -->"


def sci(x):
    return re.sub(r"e([+-])0*(\d)", r"e\1\2", f"{x:.1e}").replace("e+", "e")


def table(log):
    events = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    verdicts = {e["hypothesis"]: e for e in events if e["type"] == "verdict"}
    best = next(e for e in reversed(events) if e["type"] == "final_report")["report"]["best"]["run"]
    rows = [
        "| Run | Hypothesis | Branch | Config | Latency (ms) | vs parent | vs naive | Standard error | Adversarial error | Registers | Spills | Verdict |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for e in events:
        if e["type"] != "run_finished":
            continue
        r, m = e["run"], e["run"]["metrics"]
        p = r["params"]
        s = m["speedups"]
        v = verdicts[r["hypothesis"]]
        config = f"{p['block_m']}x{p['block_n']}x{p['block_k']}, {p['warps']} warps, {p['stages']} stages, {p['acc'].upper()} acc, {p['gelu']} GELU"
        parent = f"{s['parent']:.2f}x" if "parent" in s else f"{s['naive']:.2f}x (naive)"
        verdict = v["verdict"] + (" (provisional)" if v["provisional"] else "")
        run = f"**{r['id']}**" if r["id"] == best else r["id"]
        rows.append(
            f"| {run} | {r['hypothesis']} | {r['branch']} | {config} | {m['latency_ms']:.3f} | {parent} | {s['naive']:.2f}x "
            f"| {sci(m['max_error'])} | {sci(m['adv_error'])} | {m['n_regs']} | {m['n_spills']} | {verdict} |"
        )
    return "\n".join(rows)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--write"]
    md = table(Path(args[0]) if args else ROOT / "demo" / "replay.jsonl")
    if "--write" not in sys.argv:
        print(md)
        sys.exit()
    doc = ROOT / "docs" / "results.md"
    text = doc.read_text()
    if START not in text:
        sys.exit(f"{doc} has no {START} marker")
    doc.write_text(re.sub(f"{re.escape(START)}.*?{re.escape(END)}", lambda _: f"{START}\n{md}\n{END}", text, flags=re.DOTALL))
