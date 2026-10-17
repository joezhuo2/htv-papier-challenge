"""Findings exported as Markdown files for manual Papiers import. Papiers is read-only, so nothing here claims to write to it."""

from agent.verdict import sci


def bound_text(b):
    if b["set_by"] == "default":
        return f"default {sci(b['standard'])}"
    adv = "" if b["adversarial"] is None else f", adversarial under {sci(b['adversarial'])}"
    return f"standard under {sci(b['standard'])}{adv} (set by {b['set_by'].capitalize()})"


def summary_md(s):
    moved = "\n".join(f"| {d['metric']} | {d['before']:.4g} | {d['after']:.4g} | {d['unit']} |" for d in s["moved"])
    return (f"# Branch summary: {s['branch']}\n\n**What I tried:** {s['tried']}\n\n"
            + (f"| metric | before | after | unit |\n|---|---|---|---|\n{moved}\n\n" if moved else "")
            + f"**Verdict:** {s['verdict']}\n\n**Confidence:** {s['confidence']}. {s['confidence_reason']}\n\n"
            + "**Uncertainties:**\n" + "".join(f"- {u}\n" for u in s["uncertainties"])
            + f"\n**What would change my mind:** {s['would_change_mind']}\n\n**Next step:** {s['next_step']}\n")


def negative_md(h, pm, bound):
    sweep = "\n".join(f"| {x['k']} | {sci(x['adv_error'])} |" for x in pm["sweep"])
    return (f"# Negative result: {h.id} rejected\n\n**Hypothesis:** {h.claim}\n\n**Parameters:** `{h.params.dict()}`\n\n"
            f"**Judged against:** {bound_text(bound)}\n\n**Expected:** {pm['expected']}\n\n**Observed:** {pm['observed']}\n\n"
            + (f"## Error versus K\n\n| K | adversarial error |\n|---|---|\n{sweep}\n\n" if sweep else "")
            + f"**Mechanism:** {pm['cause']}\n\n**Pivot:** {pm['pivot']}\n")


def ledger_md(hyps, best, naive_ms, eager_ms):
    rows = []
    for h in hyps:
        if not h.judgment:
            continue
        m = h.metrics
        result = f"{m.latency_ms:.3f} ms, std {sci(m.max_error)}, adv {sci(m.adv_error)}" if m and m.latency_ms else "failed"
        rows.append(f"| {h.id} | {h.branch} | {h.prediction['min_speedup']:.2f}x vs {h.prediction['vs']} | {result} | "
                    f"{' then '.join(h.history)} | {bound_text(h.bounds[-1]) if h.bounds else ''} | {h.note or ''} |")
    head = f"# Hypothesis ledger\n\nNaive {naive_ms:.3f} ms, PyTorch eager {eager_ms:.3f} ms."
    if best:
        head += f" Best confirmed run {best['run']}: {best['vs_naive']}x vs naive, {best['vs_eager']}x vs eager."
    return (head + "\n\n| id | branch | prediction | result | verdicts | final bound | note |\n|---|---|---|---|---|---|---|\n"
            + "\n".join(rows) + "\n")


def answer_md(req, cmd, plan):
    label = next(o["label"] for o in req["options"] if o["id"] == cmd["option"])
    return (f"# Human answer to {req['id']}\n\n**Asked the {req['target'].capitalize()} lens:** {req['question']}\n\n"
            + "**Evidence:**\n" + "".join(f"- {e}\n" for e in req["evidence"])
            + f"\n**Answer from the {cmd['by'].capitalize()} lens:** {label}" + (f", \"{cmd['note']}\"" if cmd.get("note") else "")
            + f"\n\n**Plan change:** {plan}\n")
