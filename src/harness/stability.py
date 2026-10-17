"""uv run python -m harness.stability [N]: how stable are same-window speedups?

Runs the best FP32 tiles and the FP16 variant N times each, every run in its own process and timed interleaved with naive, eager
and the other config. Prints one JSON line per run and a summary: the spread within a window (from the per-round ratios) and the
spread across processes (coefficient of variation of each speedup). Evidence lands in runs/stability/.
"""

import json
import statistics
import sys

from harness import runner
from harness.kernel import NAIVE, Params

FP32 = Params(block_m=128, block_n=128, block_k=32, warps=4, stages=3)
FP16 = Params(block_m=128, block_n=128, block_k=32, warps=4, stages=3, acc="fp16")


def main(n=10):
    refs = {"naive": {"target": "triton", "params": NAIVE.dict()}, "eager": {"target": "eager"}}
    rows = {"fp32": [], "fp16": []}
    for i in range(n):
        for name, cand, parent in (("fp32", FP32, None), ("fp16", FP16, FP32)):
            spec = {"target": "triton", "params": cand.dict(), "refs": refs | ({"parent": {"target": "triton", "params": parent.dict()}} if parent else {})}
            m = runner.run(spec, run_id=f"stability/{name}-{i}")
            row = {"config": name, "i": i, "latency_ms": m.latency_ms, "speedups": m.speedups, "spreads": m.spreads, "error": m.error}
            rows[name].append(row)
            print(json.dumps(row), flush=True)
    summary = {}
    for name, rs in rows.items():
        for ref in rs[0]["speedups"]:
            xs = [r["speedups"][ref] for r in rs if not r["error"]]
            summary[f"{name} vs {ref}"] = {
                "median": round(statistics.median(xs), 3), "min": round(min(xs), 3), "max": round(max(xs), 3),
                "cv_across_runs": round(statistics.stdev(xs) / statistics.mean(xs), 4),
                "median_spread_within_run": round(statistics.median(r["spreads"][ref] for r in rs if not r["error"]), 4),
            }
    print(json.dumps({"summary": summary}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
