"""Regression tests from the 19:41 UTC recording (tests/fixtures/recording_1941.json, extracted from runs/sessions/20261017-153712/events.jsonl,
run on commit 4a6268c). Each test rebuilds the state the recording reached from its logged proposals and measured metrics."""

import json
import re
import threading
import time
from pathlib import Path

from test_agent import BAND, make, of, types
from test_recording import Script
from test_recording_1933 import wait

from agent import hypothesis as hyp
from agent import orchestrator
from agent.orchestrator import Branch, Hyp
from agent.verdict import judge
from harness.kernel import Params
from harness.metrics import Metrics

REC = json.loads((Path(__file__).parent / "fixtures" / "recording_1941.json").read_text())
PROPS = {x["id"]: x for x in REC["proposals"]}
RUNS = {r["hypothesis"]: r for r in REC["runs"]}
DIRS = {d["id"]: d for d in REC["directives"]}
BRANCHES = [("main", [], "both"), ("speed", ["main"], "speed"), ("speed2", ["speed"], "speed"), ("prec", ["main"], "precision")]


def state(tmp_path, ids, directives=("d1", "d2", "d3"), **kw):
    o = make(tmp_path, kw.pop("ask", None), max_hyps=8, **kw)
    o.naive, o.eager = (Metrics.from_dict(REC["baselines"][k]) for k in ("naive", "eager"))
    for d in directives:
        o.directives.append(DIRS[d] | hyp.directive(DIRS[d]["lens"], DIRS[d]["text"]) | {"at": -1000})
    for bid, parents, lens in BRANCHES:
        o.branches[bid] = Branch(bid, parents, bid, lens, "rejected" if bid == "speed" else "done")
    o.branches["speed2"].pivot = Params(**PROPS["H0"]["params"])
    for i in ids:
        x = PROPS[i]
        m = Metrics.from_dict(RUNS[i]["metrics"])
        h = Hyp(i, x["branch"], x["claim"], x["prediction"], Params(**x["params"]), run=f"r{int(i[1:]) + 1}", metrics=m)
        h.base = Params(**PROPS["H0"]["params"]) if i == "H1" else None
        h.bonus = i == "H4"  # the hunch from d3 ran in a reserved slot
        h.ref = "naive" if x["prediction"]["vs"] == "naive" else "parent"
        h.speedup, h.noise = m.speedups[h.ref], max(BAND, m.spreads.get(h.ref, 0.0))
        bound = o.bounds(o.branches[h.branch].lens)
        o.set_verdict(h, judge(x["prediction"] | {"max_error": bound["standard"]}, m, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
        o.hyps.append(h)
    return o


ALL = [f"H{i}" for i in range(9)]


# ---- 1. post-mortem causes follow the register and spill deltas ----

def test_a_spill_drop_that_did_not_pay_off_says_the_mechanism_worked_but_was_not_the_bottleneck(tmp_path):
    o = state(tmp_path, ALL)
    for i in ("H5", "H4", "H6", "H3"):
        o.post_mortem(o.hyps[int(i[1:])])
    h5, h4, h6, h3 = (o.pms[i]["cause"] for i in ("H5", "H4", "H6", "H3"))
    assert "registers fell from 255 to 156 and spills from 28 to 0 words per thread" in h5
    assert "so warps 4 to 8 did relieve register pressure, but that was not the bottleneck" in h5 and "barely moved" not in h5
    assert "spills from 28 to 0 words per thread" in h4 and "not the bottleneck" in h4 and "barely moved" not in h4
    assert "not the bottleneck" in h6 and "barely moved" not in h6
    assert "registers and spills barely moved (255 to 255 registers, 28 to 24 spilled words)" in h3  # both deltas small


# ---- 2. the best run's ratios are the same-window ones its run card shows ----

def test_the_best_run_reports_its_same_window_speedups(tmp_path):
    o = state(tmp_path, ALL)
    o.finish()
    best = of(o, "session_finished")[0]["best"]
    h4 = RUNS["H4"]["metrics"]["speedups"]
    assert best == {"run": "r5", "hypothesis": "H4", "verdict": "inconclusive", "vs_naive": round(h4["naive"], 2), "vs_eager": round(h4["eager"], 2)}
    report = of(o, "final_report")[0]["report"]["best"]
    assert (report["vs_naive"], report["vs_eager"]) == (best["vs_naive"], best["vs_eager"]) and "cross_window" not in report
    # the recording's own numbers for r1: 2.76x and 1.03x on its run card, but 2.81x and 0.93x in its report
    r1 = RUNS["H0"]["metrics"]
    assert (round(r1["speedups"]["naive"], 2), round(r1["speedups"]["eager"], 2)) == (2.76, 1.03)
    assert (REC["final_report"]["best"]["vs_naive"], REC["final_report"]["best"]["vs_eager"]) == (2.81, 0.93)


def test_a_best_run_without_same_window_ratios_is_labelled_cross_window(tmp_path):
    o = state(tmp_path, ALL)
    o.hyps[4].metrics.speedups = {}
    o.finish()
    best = of(o, "session_finished")[0]["best"]
    assert best["cross_window"] is True and best["vs_naive"] == round(o.naive.latency_ms / o.hyps[4].metrics.latency_ms, 2)


# ---- 3. next steps name a specific untried knob or say nothing ----

def test_next_steps_have_no_template_placeholder(tmp_path):
    o = state(tmp_path, ALL)
    for h in o.hyps:
        if h.judgment.verdict != "confirmed":
            o.post_mortem(h)
    o.finish()
    nxt = of(o, "final_report")[0]["report"]["next"]
    pivots = [pm["pivot"] for pm in o.pms.values()]
    assert not any("has not tried" in x for x in nxt + pivots)
    tried = {(k, getattr(o.baseline(h)[0], k), getattr(h.params, k)) for h in o.hyps for k in o.knobs(h)}
    named = [m.groups() for x in pivots for m in re.finditer(r"\btry (block_[mnk]|warps|stages) (\d+) to (\d+)", x)]
    assert named and all((k, int(a), int(b)) not in tried for k, a, b in named)


# ---- 4. recording mode ends after a short wait once the budget is spent ----

def test_a_recording_at_its_cap_waits_briefly_for_a_late_hunch_then_ends(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    table = {json.dumps(r["params"], sort_keys=True): r["metrics"] for r in REC["runs"]}

    def engine(spec, run_id):
        if spec["target"] == "eager":
            return Metrics.from_dict(REC["baselines"]["eager"])
        p = spec["params"]
        return Metrics.from_dict(REC["baselines"]["naive"] if p["block_m"] == 16 else table.get(json.dumps(p, sort_keys=True)) or RUNS["H8"]["metrics"])

    o = make(tmp_path, Script(json.dumps({"claim": PROPS["H0"]["claim"], "prediction": PROPS["H0"]["prediction"], "params": PROPS["H0"]["params"]})),
             engine=engine, directive_wait_s=30, end_wait_s=1.0)
    o.set_record(2)
    t = threading.Thread(target=o.run)
    start = time.time()
    t.start()
    try:
        assert wait(lambda: o.hyps and o.hyps[0].judgment)
        o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d1"]["text"]})
        assert wait(lambda: len(o.hyps) > 1 and o.hyps[1].judgment)  # H1, the seeded FP16 variant, spends the cap of 2
        time.sleep(0.3)
        o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d3"]["text"]})
        t.join(10)
        ended = not t.is_alive()
    finally:
        o.stop()
        t.join(5)
    assert ended and time.time() - start < 15
    assert o.hyps[2].note.startswith("Hunch from the Speed lens in directive d2") and o.hyps[2].bonus
    texts = [e["text"] for e in of(o, "agent_log")]
    assert not any(x.startswith("No lens directive in") for x in texts)
    assert "No directive in 1 s and the 2 hypothesis budget is spent, so the session ends." in texts
    assert types(o)[-2:] == ["final_report", "session_finished"]
