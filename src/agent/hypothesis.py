"""Hypothesis prompts and parsing. The LLM may only choose values for the Params fields."""

import json
import re

from harness.kernel import (
    ACC,
    ACC_LIMIT,
    GELU,
    SMEM_LIMIT,
    Params,
    acc_per_thread,
    fit_stages,
    smem_bytes,
)
from harness.metrics import MAX_ROUNDS, ROUNDS

SYSTEM = f"""You are a GPU kernel research agent tuning a fused Triton kernel computing gelu(x @ w + b) in FP16 on an RTX 5060 laptop GPU (Blackwell, 8 GB). Shape M = N = K = 2048.
You may only edit these parameters:
- block_m, block_n, block_k: powers of two from 16 to 256
- warps: one of 1, 2, 4, 8, 16
- stages: 1 to 6
- acc: one of {list(ACC)} (accumulation dtype)
- gelu: one of {list(GELU)} (approx is the tanh formulation)
Shared memory: a config needs (stages - 1) * (block_m * block_k + block_k * block_n) * 2 bytes and must stay at or under {SMEM_LIMIT} bytes.
For example 128x256x64 with 4 stages needs 147456 bytes and does not fit, 128x128x64 with 4 stages needs 98304 and fits.
Registers: each thread holds block_m * block_n / (warps * 32) accumulator elements, at most {ACC_LIMIT["fp32"]} with fp32 and {ACC_LIMIT["fp16"]} with fp16 accumulation, or the kernel spills to local memory and runs about 10x slower.
For example 128x256 with 4 warps puts 256 fp32 elements on each thread and is refused, 128x256 with 8 warps puts 128 and is allowed.
Before any run, state a falsifiable hypothesis with thresholds. The harness, not you, decides whether it held.
Never propose a configuration listed in the ledger, and never take a rejected configuration and change only a knob the ledger shows hurt.
If your change sets a knob to a value the ledger shows hurt (a run that changed only that knob and failed, broke the error bound or ran slower than its parent), the claim must cite that hypothesis id and say why this case differs.
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


def prompt(branch, directives, start, ledger, avoid, notes=(), bound=1e-3, headroom=None, refused=(), prior=""):
    """ledger: every run this session on any branch, as rows from Orchestrator.ledger(). refused: rejected single-knob changes.
    prior: the prior-verdicts section built by agent.fingerprint from the [RWR] notes earlier sessions wrote, empty when there are none."""
    lines = [
        f"Branch: {branch['id']} ({branch['label']}).",
        f"Error bound on this branch, set by its lens: max_error under {bound:.0e} on the standard batch. Use it as prediction.max_error.",
        f"Start from these parameters: {json.dumps(start['params'])}, measured {json.dumps(brief(start['metrics']))}.",
        "Directives: " + ("; ".join(f"{d['lens']} lens: {d['text']}" for d in directives) or "none yet, improve throughput while keeping max_error under 1e-3 on the standard batch."),
        "Ledger of every configuration run this session, on any branch: " + (json.dumps(ledger) if ledger else "nothing yet"),
    ]
    if headroom:
        lines.append(headroom_line(headroom))
    if avoid:
        lines.append("Ruled out by earlier post-mortems, do not repeat: " + "; ".join(avoid))
    if refused:
        lines.append("Changes that lost this session (slower than the parent, or over an error bound), on any branch. One that lost alone is "
                     "refused in any proposal; one that lost only together with other changes may be tested alone, never combined again: "
                     + "; ".join(refused))
    if prior:
        lines.append(prior)
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
    rounds = h.get("retest_rounds")
    if rounds is not None and (not isinstance(rounds, int) or isinstance(rounds, bool) or not ROUNDS < rounds <= MAX_ROUNDS):
        return None, f"retest_rounds must be an integer above {ROUNDS} and at most {MAX_ROUNDS}"
    try:
        params = Params(**h.get("params", {}))
    except (TypeError, ValueError) as e:
        return None, f"invalid params: {e}"
    # over the shared memory limit, the largest stage count that fits is tested and the caller rewrites the claim from the final params
    per, limit = acc_per_thread(params), ACC_LIMIT[params.acc]
    if per > limit:
        return None, (f"{params.block_m}x{params.block_n} tiles over {params.warps} warps put {per} {params.acc} accumulator elements on each thread, "
                      f"over the {limit} register limit; use more warps or a smaller output tile")
    fitted = fit_stages(params)
    adjusted = None if fitted == params else (f"Stages cut from {params.stages} to {fitted.stages}, the largest count that fits in shared memory "
                                              f"({smem_bytes(params)} bytes needed, {smem_bytes(fitted)} of {SMEM_LIMIT} after the cut)")
    return {"claim": re.sub(r"\bgelu\b", "GELU", h["claim"], flags=re.IGNORECASE), "explanation": h.get("explanation", ""),
            "prediction": {k: p[k] for k in ("min_speedup", "vs", "max_error")}, "params": fitted, "adjusted": adjusted, "rounds": rounds}, None


KNOB_RE = {
    "warps": re.compile(r"\b(\d+)[- ]warps?\b|\b(?:num_)?warps\s*(?:=|:|of|is)\s*(\d+)\b"),
    "stages": re.compile(r"\b(\d+)[- ](?:pipeline[- ])?stages?\b|\b(?:num_)?stages\s*(?:=|:|of|is)\s*(\d+)\b"),
    **{f"block_{k}": re.compile(rf"\bblock_{k}\s*(?:=|:|of|is)?\s*(\d+)\b") for k in "mnk"},
}


def named(claim, k, v):
    """Whether a claim names this parameter value."""
    t = claim.lower()
    if k == "acc":
        return v in t
    if k == "gelu":
        return bool(re.search(r"approx|tanh" if v == "approx" else r"exact|erf", t))
    if k.startswith("block_"):
        slot = "mnk".index(k[-1])
        tiles = [[int(g) for g in m.groups() if g] for m in re.finditer(r"\b(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?\b", t)]
        if any(len(d) > slot and d[slot] == v for d in tiles):
            return True
    return any(int(next(g for g in m.groups() if g)) == v for m in KNOB_RE[k].finditer(t))


def mismatches(claim, p):
    """Tile sizes, warps and stages a claim names that differ from the params that will run, as the quoted phrases."""
    t = claim.lower()
    bad = [m.group(0) for m in re.finditer(r"\b(\d+)\s*x\s*(\d+)(?:\s*x\s*(\d+))?\b", t)
           if [int(g) for g in m.groups() if g] != [p.block_m, p.block_n, p.block_k][:len([g for g in m.groups() if g])]]
    for k, rx in KNOB_RE.items():
        bad += [m.group(0) for m in rx.finditer(t) if int(next(g for g in m.groups() if g)) != getattr(p, k)]
    return bad


UP, DOWN = r"increas|rais|grow|doubl|bump|widen|deepen", r"decreas|reduc|lower|halv|shrink|cut|narrow"
CHANGE_RE = re.compile(rf"\b({UP}|{DOWN}|switch|chang|mov|set)\w*\s+(?:the\s+)?(?:num_)?(block_[mnk]|warps|stages)\b"
                       r"(?:\s+from\s+(\d+))?(?:\s+to\s+(\d+))?")
# an accumulation dtype or GELU form the claim says the config uses, not one it compares against
COMPARED = re.compile(r"(than|vs\.?|versus|against|instead of|from|over|unlike)\s+(?:the\s+)?(?:\S+\s+){0,3}$")
ACC_RE = re.compile(r"\b(fp16|fp32)(?=[\s-]+accum)|\baccum\w*\s+(?:in|to|at|of|=)\s+(fp16|fp32)\b|\bacc\s*=\s*(fp16|fp32)\b")
GELU_RE = re.compile(r"\b(approx\w*|tanh|exact|erf)(?=[\s-]+(?:\w+[\s-]+)?gelu)|\bgelu\s*(?:=|:)\s*(approx|exact)\b")
SWITCH_RE = re.compile(r"\b(switch|chang|mov|swap|replac|convert|flip)\w*\s+(?:\w+\s+){0,2}$")


def claim_errors(claim, start, p):
    """Where a claim disagrees with the change start -> p that will run: numbers the params do not have, a knob it says changes
    that does not (or moves the other way), and an accumulation dtype or GELU form other than the one in the params."""
    t = claim.lower()
    out = [f"it names {x}, the params do not" for x in mismatches(claim, p)]
    # only the accumulation dtype has a bit width, "64-bit K-dim" confuses a tile size with a precision
    out += [f"it names {n}-bit, which no parameter sets (block_k is a tile size in elements)" for n in re.findall(r"\b(\d+)[- ]bits?\b", t)
            if n not in ("16", "32")]
    for m in CHANGE_RE.finditer(t):
        verb, k, frm, to = m.groups()
        a, b = getattr(start, k), getattr(p, k)
        if a == b:
            out.append(f"\"{m.group(0)}\" names a change the params do not make, {k} stays {b}")
        elif frm and int(frm) != a:
            out.append(f"\"{m.group(0)}\" starts from {k} {frm}, the start config has {a}")
        elif to and int(to) != b:
            out.append(f"\"{m.group(0)}\" ends at {k} {to}, the params set {b}")
        elif re.match(UP, verb) and b < a or re.match(DOWN, verb) and b > a:
            out.append(f"\"{m.group(0)}\" goes the wrong way, {k} {a} to {b}")
    for k, rx, norm in (("acc", ACC_RE, str.upper), ("gelu", GELU_RE, lambda v: "approx" if v.startswith(("approx", "tanh")) else "exact")):
        what = "accumulation" if k == "acc" else "GELU"
        for m in rx.finditer(t):
            v, have = norm(next(g for g in m.groups() if g)), norm(getattr(p, k))
            before = t[:m.start()]
            if COMPARED.search(before):
                continue
            if v != have:
                out.append(f"it names {v} {what}, the params use {have} {what}")
            elif SWITCH_RE.search(before) and norm(getattr(start, k)) == v:
                out.append(f"it switches to {v} {what}, which the start config already uses")
    return list(dict.fromkeys(out))


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


VAGUE = re.compile(r"\b(bigger|larger|smaller|wider|narrower|more|fewer|double|halve|increase|decrease|reduce)\b[^.]{0,30}\b(tiles?|blocks?|block_[mnk]|warps|stages)\b"
                   r"|\bgelu\b|\b(fp16|bf16|fp32|tf32)\b|\bfus(e|ion)\b")


def hunch_change(text):
    """The parameter values a directive names, {} when it names a concrete change without a value (bigger tiles, a GELU change),
    None when it only sets a constraint."""
    t = text.lower()
    d = {}
    if re.search(r"\btanh\b|\bapprox", t):
        d["gelu"] = "approx"
    elif re.search(r"\berf\b|exact gelu", t):
        d["gelu"] = "exact"
    if re.search(r"\b(fp16|half)[- ]?(precision )?accum|accumulat\w* in (fp16|half)", t):
        d["acc"] = "fp16"
    elif re.search(r"\bfp32 accum|accumulat\w* in fp32", t):
        d["acc"] = "fp32"
    if m := re.search(r"\b(\d+)\s*x\s*(\d+)\s*x\s*(\d+)\b", t):
        d |= dict(zip(("block_m", "block_n", "block_k"), map(int, m.groups())))
    if m := re.search(r"\b(\d+)\s*warps?\b", t):
        d["warps"] = int(m.group(1))
    if m := re.search(r"\b(\d+)\s*(?:pipeline\s*)?stages?\b", t):
        d["stages"] = int(m.group(1))
    if d:
        return d
    return {} if VAGUE.search(t) else None
