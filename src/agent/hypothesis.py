"""Hypothesis prompts and parsing. The LLM may only choose values for the Params fields."""

import json
import re

from harness.kernel import ACC, GELU, Params

SYSTEM = f"""You are a GPU kernel research agent tuning a fused Triton kernel computing gelu(x @ w + b) in FP16 on an RTX 5060 laptop GPU (Blackwell, 8 GB). Shape M = N = K = 2048.
You may only edit these parameters:
- block_m, block_n, block_k: powers of two from 16 to 256
- warps: one of 1, 2, 4, 8, 16
- stages: 1 to 6
- acc: one of {list(ACC)} (accumulation dtype)
- gelu: one of {list(GELU)} (approx is the tanh formulation)
Shared memory is limited to about 100 KB per block.
Before any run, state a falsifiable hypothesis with thresholds. The harness, not you, decides whether it held.
Never propose a configuration listed as already tried.
min_speedup is the ratio start latency / new latency you expect. Against "parent_best" a realistic value is between 1.02 and 2.0. Against "naive" it is between 1.1 and 10.
Reply with only a JSON object, no prose, in this exact shape:
{{"claim": "<one sentence>", "explanation": "<two or three sentences on why you expect this>", "prediction": {{"min_speedup": <float>, "vs": "naive" or "parent_best", "max_error": <float>}}, "params": {{"block_m": int, "block_n": int, "block_k": int, "warps": int, "stages": int, "acc": str, "gelu": str}}}}"""


def r3(x):
    return float(f"{x:.3g}") if isinstance(x, float) else x


def brief(m):
    """Metrics rounded to 3 significant figures, so small run to run noise keeps prompts, and the LLM cache, stable."""
    return {k: r3(v) for k, v in m.items()}


def headroom_line(h):
    n, e = h["vs_naive"], h["vs_eager"]
    return (f"Measured headroom on this GPU: the best configuration from a tile, warps and stages sweep (FP32 accumulation, exact GELU) is "
            f"{n['median']:.2f}x faster than the naive start ({n['min']:.2f} to {n['max']:.2f} over {h['n']} runs) and {e['median']:.2f}x vs PyTorch eager "
            f"({e['min']:.2f} to {e['max']:.2f}). Keep predictions physically plausible against this.")


def prompt(branch, directives, start, tried, avoid, notes=(), bound=1e-3, headroom=None):
    lines = [
        f"Branch: {branch['id']} ({branch['label']}).",
        f"Error bound on this branch, set by its lens: max_error under {bound:.0e} on the standard batch. Use it as prediction.max_error.",
        f"Start from these parameters: {json.dumps(start['params'])}, measured {json.dumps(brief(start['metrics']))}.",
        "Directives: " + ("; ".join(f"{d['lens']} lens: {d['text']}" for d in directives) or "none yet, improve throughput while keeping max_error under 1e-3 on the standard batch."),
        "Already tried: " + (json.dumps([{"params": t["params"], "verdict": t["verdict"], **brief(t["metrics"])} for t in tried]) if tried else "nothing on this branch"),
    ]
    if headroom:
        lines.append(headroom_line(headroom))
    if avoid:
        lines.append("Ruled out by earlier post-mortems, do not repeat: " + "; ".join(avoid))
    lines += [f"Context note: {n}" for n in notes]
    lines.append('Propose the next hypothesis. Use "vs": "parent_best" unless this is the first change from the naive kernel.')
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(lines)}]


def parse(text):
    """Returns (hypothesis, None) or (None, reason). Extra keys in params, like free-form code, are refused."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None, "no json object"
    try:
        h = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"bad json: {e}"
    p = h.get("prediction")
    if not isinstance(h.get("claim"), str) or not isinstance(p, dict) \
            or not all(isinstance(p.get(k), (int, float)) for k in ("min_speedup", "max_error")) or p.get("vs") not in ("naive", "parent_best"):
        return None, "missing or mistyped fields"
    lo, hi = (1.1, 10.0) if p["vs"] == "naive" else (1.02, 2.0)
    if not lo <= p["min_speedup"] <= hi:
        return None, f"min_speedup {p['min_speedup']} vs {p['vs']} is outside the plausible range {lo} to {hi}"
    try:
        params = Params(**h.get("params", {}))
    except (TypeError, ValueError) as e:
        return None, f"invalid params: {e}"
    return {"claim": h["claim"], "explanation": h.get("explanation", ""), "prediction": {k: p[k] for k in ("min_speedup", "vs", "max_error")},
            "params": params}, None


BOUND_RE = re.compile(r"(\d+(?:\.\d+)?e-\d+|0\.0+\d+)", re.IGNORECASE)


SPEED_TOLERANCE = 2e-2
SPEED_CEILING = 5e-2
STRICT = 1e-3


def directive(lens, text):
    """The lens fixes the constraint type and its error bound, parsed in code. Speed tolerates more error on the standard batch,
    2e-2 by default and never above 5e-2. Precision demands 1e-3 on the standard and adversarial batches and can only tighten it."""
    found = BOUND_RE.search(text)
    value = float(found.group(1)) if found else None
    if lens == "speed":
        return {"constraint": "maximize_speed", "threshold": min(value, SPEED_CEILING) if value else SPEED_TOLERANCE}
    return {"constraint": "max_error", "threshold": min(value, STRICT) if value else STRICT}
