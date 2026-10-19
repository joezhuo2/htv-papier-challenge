"""The fingerprint of a tested experiment: what changed, from which config, under which bounds. It is appended to the [RWR] note a human approves,
and read back by later sessions to refuse a repeat before it costs GPU time. Notes without it still parse and are context only."""

import re

from agent.verdict import sci
from harness.metrics import MAX_ROUNDS, ROUNDS

KERNEL = "matmul_bias_gelu"
CAP = 30
KEYS = ("params", "base", "kernel", "title", "bounds", "rounds")
REQUIRED = {"params", "base", "kernel", "bounds", "rounds"}


def delta(base, params):
    """The changed fields of params versus base, in field order."""
    return {k: v for k, v in params.items() if base.get(k) != v}


def bounds_key(bound):
    return {"std": sci(bound["standard"]), "adv": sci(bound["adversarial"]) if bound.get("adversarial") else "none"}


def pairs(d):
    return " ".join(f"{k}={v}" for k, v in d.items())


def unpairs(text):
    return {k: int(v) if v.isdigit() else v for k, v in (x.split("=", 1) for x in text.split())}


def tail(change, base, bound, rounds, title):
    """The note segments after the reason."""
    b = bounds_key(bound)
    return f" | params: {pairs(change)} | base: {pairs(base)} | kernel: {KERNEL} | title: {title} | bounds: std {b['std']} adv {b['adv']} | rounds: {rounds or ROUNDS}"


def parse_tail(segments):
    """The fingerprint from the note segments after the reason, None when one is missing or malformed."""
    got = dict(s.split(": ", 1) for s in segments if ": " in s)
    if not REQUIRED <= got.keys():
        return None
    try:
        m = re.fullmatch(r"std (\S+) adv (\S+)", got["bounds"])
        return {"params": unpairs(got["params"]), "base": unpairs(got["base"]), "kernel": got["kernel"], "title": got.get("title", ""),
                "bounds": {"std": m[1], "adv": m[2]}, "rounds": int(got["rounds"])}
    except (TypeError, ValueError):
        return None


def newest(priors):
    return sorted(priors, key=lambda p: p["run"], reverse=True)


def retestable(p, rounds):
    return p["verdict"] == "inconclusive" and bool(rounds) and rounds > p["fp"]["rounds"]


def matches(priors, base, params, bound):
    """Prior verdicts that tested this exact change from this exact base under these bounds, newest first."""
    change, b = delta(base, params), bounds_key(bound)
    return [p for p in newest(priors) if (f := p.get("fp")) and f["kernel"] == KERNEL and f["params"] == change and f["base"] == base and f["bounds"] == b]


def blocking(priors, base, params, bound, rounds=None):
    """(blocker, retested): the prior verdict that refuses the proposal, or None, and the inconclusive one a retest with more rounds allows."""
    hit = matches(priors, base, params, bound)
    blocker = next((p for p in hit if not retestable(p, rounds)), None)
    return blocker, next((p for p in hit if retestable(p, rounds)), None)


def compact(d):
    return "?" if not d else " ".join(f"{k.replace('block_', '')}={v}" for k, v in d.items())


def status(p, bound):
    f = p.get("fp")
    if not f:
        return "context, old note without a fingerprint"
    if f["bounds"] != bounds_key(bound):
        return f"context, bounds differ (now std {bounds_key(bound)['std']} adv {bounds_key(bound)['adv']})"
    return f"retest above {f['rounds']} rounds" if p["verdict"] == "inconclusive" else "blocks"


def row(p, bound):
    f = p.get("fp") or {}
    b = f.get("bounds")
    return (f"| {f.get('title') or p['reason'][:40]} | {compact(f.get('params'))} | {compact(f.get('base'))} | {p['verdict']} | {p['measured']} vs {p['bound']} "
            f"| {f'std {b['std']} adv {b['adv']}' if b else '?'} | {p['run']} | {status(p, bound)} |")


def section(priors, bound):
    """The prior-verdicts block of the hypothesis prompt, empty when there are none. Blocking rows come first, then at most CAP rows in all."""
    if not priors:
        return ""
    b = bounds_key(bound)
    order = {"blocks": 0, "retest": 1}
    rows = sorted(newest(priors), key=lambda p: order.get(status(p, bound).split()[0].rstrip(","), 2))[:CAP]
    return "\n".join([
        "Prior verdicts from earlier sessions, read from Papiers (past results, not evidence, never cite them as support for a claim).",
        (f"Bounds now: std {b['std']}, adv {b['adv']}. The harness refuses, before any run, a proposal that makes the same change from the same start config "
        f"as a row marked \"blocks\", so propose something different. A \"retest\" row (inconclusive) is allowed only with \"retest_rounds\" in your JSON, an integer "
        f"above the rounds shown and at most {MAX_ROUNDS}. Rows marked \"context\" never block."),
        "| title | change | from | verdict | numbers | bounds | session | status |",
        "|---|---|---|---|---|---|---|---|",
        *(row(p, bound) for p in rows),
    ])


def tested(p):
    f = p["fp"]
    return f"{pairs(f['params'])} from {pairs(f['base'])}, {p['verdict']} in {p['run']} ({p['measured']} vs {p['bound']}) under the same bounds"


def refusal(p):
    """Why a proposal is refused, in the words the model and the event feed see."""
    return (f"it repeats an experiment an earlier session already ran: {tested(p)}; "
            + (f"to retest it set retest_rounds above {p['fp']['rounds']} and at most {MAX_ROUNDS}" if p["verdict"] == "inconclusive" else "propose something different"))
