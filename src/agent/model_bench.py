"""uv run python -m agent.model_bench: the Section 5 model benchmark. Same hypothesis prompt, N trials per model."""

import json
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import openai

from agent.llm import Client
from harness import runner
from harness.kernel import ACC, GELU, NAIVE, Params
from harness.metrics import Metrics

MODELS = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": 120,
    "openai/gpt-oss-20b": 120,
    "google/gemma-4-31b-it": 120,
}
TRIALS = 10
GIVE_UP = 3
BUDGET_S = 360
NAIVE_FILE = runner.RUNS / "model_bench_naive.json"

SYSTEM = f"""You are a GPU kernel research agent tuning a fused Triton kernel computing gelu(x @ w + b) in FP16 on an RTX 5060 laptop GPU (Blackwell, 8 GB). Shape M = N = K = 2048.
You may only edit these parameters:
- block_m, block_n, block_k: powers of two from 16 to 256
- warps: one of 1, 2, 4, 8, 16
- stages: 1 to 6
- acc: one of {list(ACC)} (accumulation dtype)
- gelu: one of {list(GELU)} (approx is the tanh formulation)
Shared memory is limited to about 100 KB per block.
Before any run, state a falsifiable hypothesis. Reply with only a JSON object, no prose, in this exact shape:
{{"claim": "<one sentence>", "explanation": "<two or three sentences on why you expect this>", "prediction": {{"min_speedup": <float>, "vs": "naive" or "parent_best", "max_error": <float>}}, "params": {{"block_m": int, "block_n": int, "block_k": int, "warps": int, "stages": int, "acc": str, "gelu": str}}}}"""

DIRECTIVE = "Speed lens: maximize throughput. The error bound max_error < 1e-3 on the standard batch must hold."


def user_prompt(params, last):
    return (f"Current parameters: {json.dumps(params)}\nDirective: {DIRECTIVE}\n"
            f"Last result: {json.dumps(last)}\nPropose the next hypothesis.")


def parse(text):
    """Extracts and validates a hypothesis. Returns (hypothesis, None) or (None, reason)."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no json object"
    try:
        h = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"bad json: {e}"
    p = h.get("prediction", {})
    if not isinstance(h.get("claim"), str) or not all(isinstance(p.get(k), (int, float)) for k in ("min_speedup", "max_error")) \
            or p.get("vs") not in ("naive", "parent_best"):
        return None, "missing or mistyped fields"
    try:
        Params(**h.get("params", {}))
    except (TypeError, ValueError) as e:
        return None, f"invalid params: {e}"
    return h, None


def trials(client, model, prompt):
    out, fails, t0 = [], 0, time.monotonic()
    for i in range(TRIALS):
        if time.monotonic() - t0 > BUDGET_S:
            print(f"  {model}: stopping after {i} trials, over the {BUDGET_S} s budget", file=sys.stderr, flush=True)
            break
        try:
            r = client.complete(prompt, model=model, tag=f"trial{i}", max_tokens=2048)
        except (openai.APITimeoutError, openai.APIConnectionError, openai.APIStatusError) as e:
            out.append({"error": f"{type(e).__name__}: {str(e)[:120]}"})
            fails += 1
            if fails >= GIVE_UP and not any("text" in o for o in out):
                break
            continue
        h, why = parse(r["text"])
        out.append(r | {"hypothesis": h, "invalid": why})
        print(f"  {model} trial {i}: {r['latency_s']}s{' (cached)' if r['cached'] else ''} {'ok' if h else why}", file=sys.stderr, flush=True)
    return out


def main():
    # The naive result is pinned to a file so the prompt, and with it the response cache, stays stable across reruns.
    if NAIVE_FILE.exists():
        naive = Metrics.from_dict(json.loads(NAIVE_FILE.read_text()))
    else:
        naive = runner.run({"target": "triton", "params": NAIVE.dict()})
        NAIVE_FILE.write_text(json.dumps(naive.dict()))
    last = naive.contract()
    prompt = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_prompt(NAIVE.dict(), last)}]

    with ThreadPoolExecutor(len(MODELS)) as pool:
        futures = {m: pool.submit(trials, Client(timeout=t), m, prompt) for m, t in MODELS.items()}
        results = {m: f.result() for m, f in futures.items()}

    # Run each distinct valid edit once through the real harness.
    checked = {}
    for rs in results.values():
        for r in rs:
            if r.get("hypothesis"):
                key = json.dumps(r["hypothesis"]["params"], sort_keys=True)
                if key not in checked:
                    checked[key] = runner.run({"target": "triton", "params": r["hypothesis"]["params"]})

    report = {}
    for model, rs in results.items():
        answered = [r for r in rs if "text" in r]
        live = [r["latency_s"] for r in answered if not r["cached"]] or [r["latency_s"] for r in answered]
        valid = [r for r in answered if r["hypothesis"]]
        runs = [checked[json.dumps(r["hypothesis"]["params"], sort_keys=True)] for r in valid]
        report[model] = {
            "attempts": len(rs),
            "errors": [r["error"] for r in rs if "error" in r],
            "median_latency_s": statistics.median(live) if live else None,
            "valid_json_rate": len(valid) / len(rs),
            "compile_rate": sum(m.compiled_ok for m in runs) / len(rs),
            "correct_rate": sum(m.correct_ok for m in runs) / len(rs),
            "max_latency_s": max(live) if live else None,
            "invalid_reasons": [r["invalid"] for r in answered if r["invalid"]],
            "edits": [{"params": r["hypothesis"]["params"], "latency_ms": m.latency_ms, "max_error": m.max_error,
                       "speedup_vs_naive": round(naive.latency_ms / m.latency_ms, 2) if m.latency_ms else None, "run_id": m.run_id}
                      for r, m in zip(valid, runs)],
            "examples": [{"claim": r["hypothesis"]["claim"], "explanation": r["hypothesis"].get("explanation"),
                          "prediction": r["hypothesis"]["prediction"]} for r in valid[:2]],
        }
    print(json.dumps({"naive": last, "models": report}, indent=2))


if __name__ == "__main__":
    main()
