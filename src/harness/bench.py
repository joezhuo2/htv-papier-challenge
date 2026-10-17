"""uv run python -m harness.bench: prints one JSON line per run for eager, the naive kernel, and a tile/warps/stages sweep."""

import itertools
import json
from dataclasses import replace

from harness import runner
from harness.kernel import NAIVE, Params

TILES = [(32, 32, 32), (64, 64, 32), (64, 128, 32), (128, 64, 32), (128, 128, 32), (128, 128, 64), (128, 256, 64)]
WARPS_STAGES = [(4, 2), (4, 3), (8, 3), (8, 4)]


def triton(p):
    return runner.run({"target": "triton", "params": p.dict()})


def emit(label, m, eager=None, naive=None):
    row = {"label": label} | m.dict()
    if m.latency_ms:
        for name, ref in (("eager", eager), ("naive", naive)):
            if ref:
                row[f"speedup_vs_{name}"] = round(ref.latency_ms / m.latency_ms, 3)
    print(json.dumps(row), flush=True)
    return m


def main():
    eager = emit("eager", runner.run({"target": "eager"}))
    naive = emit("naive", triton(NAIVE), eager)

    sweep = []
    for (bm, bn, bk), (w, s) in itertools.product(TILES, WARPS_STAGES):
        p = Params(block_m=bm, block_n=bn, block_k=bk, warps=w, stages=s)
        sweep.append(emit("sweep", triton(p), eager, naive))

    ok = [m for m in sweep if m.correct_ok]
    if not ok:
        return
    best = emit("best", min(ok, key=lambda m: m.latency_ms), eager, naive)

    # Precision-sensitive variants of the best config, as early evidence for the failure path.
    base = Params(**best.params)
    for k, v in (("acc", "fp16"), ("gelu", "approx")):
        emit(f"variant_{k}", triton(replace(base, **{k: v})), eager, naive)


if __name__ == "__main__":
    main()
