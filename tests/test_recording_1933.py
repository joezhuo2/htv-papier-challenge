"""Regression tests from the 19:33 UTC recording (tests/fixtures/recording_1933.json, extracted from runs/sessions/20261017-150715/events.jsonl).
The recording ran on a server started at 11:23, so it predates every fix since; these tests rebuild its state on the current code."""

import json
import threading
import time
from pathlib import Path

from test_agent import BAND, assert_contract, make, of, types
from test_recording import Script, assert_no_losing_repeat

from agent import hypothesis as hyp
from agent import orchestrator
from agent.orchestrator import Branch, Hyp
from agent.verdict import judge
from harness.kernel import Params
from harness.metrics import Metrics

REC = json.loads((Path(__file__).parent / "fixtures" / "recording_1933.json").read_text())
PROPS = {x["id"]: x for x in REC["proposals"]}
RUNS = {r["hypothesis"]: r for r in REC["runs"]}
DIRS = {d["id"]: d for d in REC["directives"]}
H0 = Params(**PROPS["H0"]["params"])
BRANCHES = [("main", [], "both"), ("speed", ["main"], "speed"), ("speed2", ["speed"], "speed"), ("prec", ["main"], "precision"),
            ("merge", ["prec", "speed2"], "both")]
BASES = {"H1": "H0", "H7": "H5"}  # the seeded FP16 variant and the hybrid were timed against these


def answer(i, **kw):
    x = PROPS[i] | kw
    return json.dumps({"claim": x["claim"], "prediction": x["prediction"], "params": x["params"]})


def state(tmp_path, ids, directives=("d1", "d2", "d3"), status=None, **kw):
    """The recording after the hypotheses in ids were judged. Every branch it opened is done, speed is rejected."""
    o = make(tmp_path, kw.pop("ask", None), max_hyps=8, **kw)
    o.naive, o.eager = (Metrics.from_dict(REC["baselines"][k]) for k in ("naive", "eager"))
    for d in directives:
        o.directives.append(DIRS[d] | hyp.directive(DIRS[d]["lens"], DIRS[d]["text"]) | {"at": -1000})
    status = {"speed": "rejected"} | (status or {})
    for bid, parents, lens in BRANCHES:
        if bid == "main" or any(PROPS[i]["branch"] == bid for i in ids):
            o.branches[bid] = Branch(bid, parents, bid, lens, status.get(bid, "done"), budget=1 if bid == "merge" else None)
    for i in ids:
        x = PROPS[i]
        m = Metrics.from_dict(RUNS[i]["metrics"])
        h = Hyp(i, x["branch"], x["claim"], x["prediction"], Params(**x["params"]), run=f"r{int(i[1:]) + 1}", metrics=m)
        h.base = Params(**PROPS[BASES[i]]["params"]) if i in BASES else None
        h.bonus = i == "H7"
        h.ref = "naive" if x["prediction"]["vs"] == "naive" else "parent"
        h.speedup, h.noise = m.speedups[h.ref], max(BAND, m.spreads.get(h.ref, 0.0))
        bound = o.bounds(o.branches[h.branch].lens)
        o.set_verdict(h, judge(x["prediction"] | {"max_error": bound["standard"]}, m, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
        o.hyps.append(h)
    return o


def engine(spec, run_id):
    """Measured metrics from the recording by params, with noiseless same-window ratios against the references."""
    table = {json.dumps(r["params"], sort_keys=True): r["metrics"] for r in REC["runs"]}
    one = lambda p: Metrics.from_dict(REC["baselines"]["naive"] if p["block_m"] == 16 else table.get(json.dumps(p, sort_keys=True)) or RUNS["H6"]["metrics"])
    if spec["target"] == "eager":
        return Metrics.from_dict(REC["baselines"]["eager"])
    if spec["target"] == "sweep":
        return Metrics(compiled_ok=True, errors={f"k{k}": 2e-3 * (k / 256) ** 0.5 for k in spec["ks"]})
    m = one(spec["params"])
    if spec.get("refs"):
        ref = lambda r: Metrics.from_dict(REC["baselines"]["eager"]) if r["target"] == "eager" else one(r["params"])
        m = Metrics.from_dict(m.dict() | {"speedups": {k: ref(r).latency_ms / m.latency_ms for k, r in spec["refs"].items()},
                                          "spreads": {k: 0.0 for k in spec["refs"]}})
    return m


def wait(pred, timeout=5):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        time.sleep(0.01)
    return pred()


# ---- 1. the idle tail ----

def test_a_recording_ends_once_every_branch_is_done_and_both_lenses_have_spoken(tmp_path):
    o = make(tmp_path, Script(answer("H0")), engine=engine, directive_wait_s=1.0, help_deadline_s=30)
    o.set_record(8)
    t = threading.Thread(target=o.run)
    t.start()
    try:
        assert wait(lambda: o.hyps and o.hyps[0].judgment)
        o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d1"]["text"]})
        assert wait(lambda: len(o.hyps) > 1 and o.hyps[1].judgment)
        o.command({"type": "add_directive", "lens": "precision", "text": DIRS["d2"]["text"]})
        t.join(6)
        ended = not t.is_alive()
    finally:
        o.stop()
        t.join(5)
    assert ended, "the session kept waiting after every branch was done and both lenses had sent a directive"
    texts = [e["text"] for e in of(o, "agent_log")]
    assert not any(x.startswith("No lens directive in") for x in texts)
    assert sum(not h.bonus for h in o.hyps) < 8  # budget remained, but no human was expected, so the session ended
    assert "Every branch is done and both lenses have sent a directive, so the session ends with 6 of its 8 hypotheses unused." in texts
    assert types(o)[-2:] == ["final_report", "session_finished"]


# ---- 2. help requests and budget ----

def test_no_help_request_when_the_budget_left_is_already_promised(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"], status={"main": "running"}, engine=engine)
    o.max_hyps = 10
    o.promises.append({"ref": "hq0", "branch": "speed2", "n": 2, "start": 2})  # two iterations promised on speed2, none ran
    o.ask = Script(answer("H8"))
    o.iterate(o.branches["main"])
    assert o.hyps[-1].id == "H8" and o.hyps[-1].judgment.verdict == "inconclusive"
    assert not of(o, "help_requested") and o.room() == (0, "the 10 hypothesis cap has 2 left and 2 of those are already promised")
    assert of(o, "summary")[-1]["summary"]["next_step"] == "Stop this branch: no hypothesis budget remains for more iterations."


def test_a_help_answer_says_how_many_hypotheses_remain(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"], status={"main": "running"})
    o.max_hyps = 12
    o.request_help(o.hyps[8], o.branches["main"], "speed")
    o.answer({"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})
    assert of(o, "plan_change")[-1]["text"] == ("Because the Speed lens wants more tile search, I am giving main two more iterations. "
                                                "The 12 hypothesis cap has 4 left.")


# ---- 3. the best run ----

def test_the_best_run_is_the_fastest_within_every_bound_whatever_its_verdict(tmp_path):
    o = state(tmp_path, [f"H{i}" for i in range(9)])
    assert o.hyps[8].judgment.verdict == "inconclusive" and o.hyps[8].metrics.latency_ms < o.hyps[0].metrics.latency_ms
    o.finish()
    best = of(o, "session_finished")[0]["best"]
    sp = o.hyps[8].metrics.speedups
    assert best == {"run": "r9", "hypothesis": "H8", "verdict": "inconclusive", "vs_naive": round(sp["naive"], 2), "vs_eager": round(sp["eager"], 2)}
    report = of(o, "final_report")[0]["report"]["best"]
    assert (report["run"], report["hypothesis"], report["verdict"], report["vs_naive"]) == ("r9", "H8", "inconclusive", best["vs_naive"])
    # H1 is faster still, but breaks the Precision bound
    assert o.hyps[1].metrics.latency_ms < o.hyps[8].metrics.latency_ms


def test_a_miss_within_every_bound_reads_as_a_smaller_gain_not_only_rejected(tmp_path):
    o = state(tmp_path, [f"H{i}" for i in range(9)])
    h5, h2 = o.hyps[5].judgment, o.hyps[2].judgment
    assert h5.verdict == "rejected" and h5.reason == ("Gain smaller than predicted: 1.02x vs best (noise +/- 0.02x) against >= 1.05x. "
                                                      "Within every error bound.")
    assert h2.verdict == "rejected" and h2.reason == "Slower than its parent: 0.96x vs best (noise +/- 0.05x) against >= 1.15x. Within every error bound."
    assert o.hyps[1].judgment.reason.startswith("standard max error 1.3e-2 missed")  # an error miss keeps its wording


# ---- 4. still pending from the previous prompt ----

def test_the_64_bit_k_dim_claim_is_sent_back():
    h6 = PROPS["H6"]
    assert "it names 64-bit, which no parameter sets (block_k is a tile size in elements)" in hyp.claim_errors(h6["claim"], H0, Params(**h6["params"]))
    assert hyp.claim_errors("Increasing block size to 128x128 with 64-bit K-dim improves throughput.", H0, Params(**h6["params"])) == [
        "it names 64-bit, which no parameter sets (block_k is a tile size in elements)"]
    assert hyp.claim_errors("16-bit accumulation is faster.", H0, Params(**H0.dict() | {"acc": "fp16"})) == []


def test_block_k_32_to_64_is_refused_after_losing_in_H2_and_H3(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5"])
    lost = o.losing()
    assert [x.id for x in lost[("block_k", 32, 64)]] == ["H2", "H3"]
    assert ("gelu", "exact", "approx") not in lost  # H5 gained 1.02x within every bound, a smaller gain is not a loss
    why = o.vet(o.start_point("prec")[0], Params(**PROPS["H6"]["params"]), PROPS["H6"]["claim"])
    assert why.startswith("it repeats changes that lost before without isolating them:") and "block_k 32 to 64 (lost in H2, H3)" in why
    assert o.vet(H0, Params(**H0.dict() | {"block_k": 64}), "Block_k 32 to 64 alone.") is None  # one isolation test is allowed
    # H8, the fastest run of the recording, changes only the output tile, which counts as one direction, so it is an isolation test too
    assert o.vet(H0, Params(**PROPS["H8"]["params"]), PROPS["H8"]["claim"]) is None
    prompt = hyp.prompt({"id": "prec", "label": "x"}, [], {"params": H0.dict(), "metrics": {}}, [], [], refused=o.refused_lines())
    assert "block_k 32 to 64 (lost in H2, H3)" in prompt[1]["content"]


def test_the_recorded_hybrid_of_H4_and_H5_is_not_proposed(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5"])
    assert o.merge_why_not(o.hyps[4]) == "0.96x vs its parent, under 1.00x"
    o.maybe_hybrid()
    assert not of(o, "hybrid_proposed")


def test_notes_from_this_recording_name_the_document_not_the_cnv_id(tmp_path):
    o = state(tmp_path, [], directives=(), status={"main": "running"}, engine=engine, context=lambda q: REC["papiers"])
    o.ask = Script(answer("H0"))
    o.iterate(o.branches["main"])
    hy = of(o, "hypothesis_proposed")[0]["hypothesis"]
    assert hy["note"] == "Context from the Papiers conversation \"Conversation overview\"" and hy["papiers"].startswith("cnv_")
    assert "cnv_" not in hy["claim"] + hy["note"] and "cnv_" not in o.ask.calls[0][2][1]["content"]


# ---- the whole recording, replayed ----

def test_the_recording_replayed_keeps_every_invariant(tmp_path, monkeypatch):
    """Recording mode, the recording's proposals in order, d1 after H0, d2 and d3 while runs are on the GPU (as they arrived), every help
    request answered with keep tuning and the hybrid approved when proposed."""
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    sent = []

    def gpu(spec, run_id):
        if spec.get("refs") and o.directives and len(sent) < 2:
            d = DIRS[["d2", "d3"][len(sent)]]
            sent.append(d["id"])
            o.command({"type": "add_directive", "lens": d["lens"], "text": d["text"]})
            wait(lambda: len(o.directives) == len(sent) + 1)
        return engine(spec, run_id)

    model = [answer(i) for i in ("H0", "H2", "H3", "H4", "H5", "H6", "H8")]
    o = make(tmp_path, Script(*model), engine=gpu, directive_wait_s=1.0, help_deadline_s=30)
    o.set_record(8)
    t = threading.Thread(target=o.run)
    t.start()
    done = set()
    end = time.time() + 25
    while t.is_alive() and time.time() < end:
        if o.hyps and o.hyps[0].judgment and not o.directives:
            o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d1"]["text"]})
            wait(lambda: o.directives)
        for e in of(o, "help_requested"):
            if e["help"]["id"] not in done:
                done.add(e["help"]["id"])
                o.command({"type": "answer_help", "help": e["help"]["id"], "option": "tile_search", "by": "speed"})
        for e in of(o, "hybrid_proposed"):
            if "hybrid" not in done:
                done.add("hybrid")
                o.command({"type": "approve_hybrid", "branches": e["branches"]})
        time.sleep(0.01)
    alive = t.is_alive()
    o.stop()
    t.join(5)
    ev = o.log.events
    assert not alive, "the session did not end by itself"
    assert_contract(ev)
    assert types(o)[-2:] == ["final_report", "session_finished"]
    assert not any(e["text"].startswith("No lens directive in") for e in of(o, "agent_log"))
    judged = {h.id: h.judgment.verdict for h in o.hyps if h.judgment}
    assert {i for i, v in judged.items() if v != "confirmed"} <= {e["post_mortem"]["hypothesis"] for e in of(o, "post_mortem")}
    # help requests only while budget remains
    for n, e in enumerate(ev):
        if e["type"] == "help_requested":
            used = sum(x["type"] == "hypothesis_proposed" and x["hypothesis"]["id"] not in {h.id for h in o.hyps if h.bonus} for x in ev[:n])
            assert used < 8, e
    # the best run is the fastest measured run within every active bound, with its verdict
    ok = [h for h in o.hyps if h.metrics and h.metrics.latency_ms and o.failed_bound(h) is None]
    best = min(ok, key=lambda h: h.metrics.latency_ms)
    fin = of(o, "session_finished")[0]["best"]
    assert (fin["run"], fin["verdict"]) == (best.run, best.judgment.verdict)
    # no proposal repeats a losing direction without isolating it, and no claim misdescribes its change
    for h in o.hyps:
        if (h.note or "").startswith(("Harness-seeded", "Hybrid")) or "Built in code" in (h.note or ""):
            continue
        assert_no_losing_repeat(o, h)
        assert hyp.claim_errors(h.claim.split(" Relative to the pivot config")[0], o.baseline(h)[0], h.params) == [], (h.id, h.claim)
    if o.hybrid:
        assert all(o.merge_why_not(h) is None for h in o.hyps if h.branch in o.hybrid["branches"] and h.params != o.hybrid["base"]
                   and all(getattr(h.params, k) == v for k, v in o.hybrid["params"].dict().items() if v != getattr(o.hybrid["base"], k)))
    assert not any("cnv_" in json.dumps(e) for e in ev if e["type"] != "papiers_context")
