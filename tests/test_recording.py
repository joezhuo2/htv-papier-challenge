"""Regression tests from the 14:38 recording (tests/fixtures/recording_1438.json, extracted from runs/sessions/20261017-143848/events.jsonl).
Each test rebuilds the session state the recording reached from its logged proposals and measured metrics, then checks the fix."""

import json
import threading
import time
from pathlib import Path

from test_agent import BAND, CLI, assert_contract, make, of, types

from agent import hypothesis as hyp
from agent import orchestrator
from agent.orchestrator import Branch, Hint, Hyp, direction
from agent.verdict import judge
from harness.kernel import Params
from harness.metrics import Metrics

REC = json.loads((Path(__file__).parent / "fixtures" / "recording_1438.json").read_text())
PROPS = {x["id"]: x for x in REC["proposals"]}
RUNS = {r["hypothesis"]: r for r in REC["runs"]}
DIRS = {d["id"]: d for d in REC["directives"]}
H0 = Params(**PROPS["H0"]["params"])
# (id, parents, lens) in the order the recording opened them
BRANCHES = [("main", [], "both"), ("speed", ["main"], "speed"), ("speed2", ["speed"], "speed"), ("prec", ["main"], "precision"),
            ("merge", ["prec", "speed"], "both")]


def answer(js):
    x = PROPS[js] if isinstance(js, str) else js
    return json.dumps({"claim": x["claim"], "prediction": x["prediction"], "params": x["params"]})


def state(tmp_path, ids, directives=("d1", "d2"), status=None, **kw):
    """The recording after the hypotheses in ids were judged, under the directives that had arrived. Statuses default to the recording's."""
    o = make(tmp_path, kw.pop("ask", None), max_hyps=8, **kw)
    o.naive, o.eager = (Metrics.from_dict(REC["baselines"][k]) for k in ("naive", "eager"))
    for d in directives:
        o.directives.append(DIRS[d] | hyp.directive(DIRS[d]["lens"], DIRS[d]["text"]) | {"at": -1000})
    status = {"main": "done", "speed": "rejected", "speed2": "running", "prec": "running", "merge": "running"} | (status or {})
    for bid, parents, lens in BRANCHES:
        if any(PROPS[i]["branch"] == bid for i in ids) or bid in ("speed2", "prec") and "d2" in directives:
            o.branches[bid] = Branch(bid, parents, bid, lens, status[bid], budget=1 if bid == "merge" else None)
    for i in ids:
        add(o, i)
    return o


def add(o, i, judged=True):
    x = PROPS[i]
    m = Metrics.from_dict(RUNS[i]["metrics"]) if judged else None
    h = Hyp(i, x["branch"], x["claim"], x["prediction"], Params(**x["params"]), run=f"r{int(i[1:]) + 1}", metrics=m)
    h.base = H0 if i == "H1" else None  # the seeded FP16 variant was timed against H0
    h.bonus = i == "H6"  # the approved hybrid ran outside the cap
    h.ref = "naive" if x["prediction"]["vs"] == "naive" else "parent"
    if judged:
        h.speedup, h.noise = m.speedups[h.ref], max(BAND, m.spreads.get(h.ref, 0.0))
        bound = o.bounds(o.branches[h.branch].lens)
        o.set_verdict(h, judge(x["prediction"] | {"max_error": bound["standard"]}, m, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
    o.hyps.append(h)
    return h


def assert_no_losing_repeat(o, h):
    """h makes no change that lost alone before it, and combines a change that lost with others only within one direction."""
    lost = o.losing(o.hyps[:o.hyps.index(h)])
    base = o.baseline(h)[0]
    changes = [(k, getattr(base, k), getattr(h.params, k)) for k in o.knobs(h)]
    assert not any(d in lost and o.isolated(d, lost[d]) for d in changes), (h.id, changes)
    assert len({direction(k) for k, _, _ in changes}) == 1 or not any(d in lost for d in changes), (h.id, changes)


class Script:
    """The model's answers in order, with hooks run while a given call is in flight (the state lock is released then)."""

    def __init__(self, *script, hooks=None):
        self.script, self.hooks, self.calls = list(script), dict(hooks or {}), []

    def __call__(self, messages, model, tag):
        self.calls.append((model, tag, messages))
        if model == "strong":
            raise TimeoutError("slow model")  # post-mortems use the factual template
        for prefix in [p for p in self.hooks if tag.startswith(p)]:
            self.hooks.pop(prefix)()
        return self.script.pop(0) if self.script else "no json"


# ---- 1. rejected single-knob changes and claims that misdescribe the change ----

def test_block_k_32_to_64_is_refused_on_every_branch_once_H5_rejected_it(tmp_path):
    # H3 changed block_k together with warps and stages, so it did not isolate the change and H5 may still test it alone
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4"])
    assert o.vet(H0, Params(**PROPS["H5"]["params"]), PROPS["H5"]["claim"]) is None
    o = state(tmp_path / "after", ["H0", "H1", "H2", "H3", "H4", "H5"])
    o.branches["merge"] = Branch("merge", ["prec", "speed"], "Hybrid", "both")
    lost = o.losing()
    assert o.isolated(("block_k", 32, 64), lost[("block_k", 32, 64)]) is o.hyps[5] and o.isolated(("acc", "fp32", "fp16"), lost[("acc", "fp32", "fp16")]) is o.hyps[1]
    refused = "it changes block_k 32 to 64, which H5 on speed2 already tested on its own and lost (rejected, 0.93x vs its parent)"
    for i in ("H7", "H8"):  # prec and merge both start from H0, so both repeat it together with other changes
        assert o.vet(o.start_point(PROPS[i]["branch"])[0], Params(**PROPS[i]["params"]), PROPS[i]["claim"]).startswith(refused)
    alt = PROPS["H7"] | {"claim": "Going to 8 warps on 128x128x32 tiles with 3 stages raises occupancy.", "params": H0.dict() | {"warps": 8}}
    o.ask = Script(answer("H7"), answer("H8"), answer(alt))
    h = o.propose(o.branches["prec"])
    assert h.params == Params(**alt["params"])
    assert [e["text"][:len("Refused a proposal on prec: " + refused)] for e in of(o, "agent_log")] == ["Refused a proposal on prec: " + refused] * 2
    prompt = o.ask.calls[0][2][1]["content"]
    assert "Changes that lost this session" in prompt and "block_k 32 to 64 (lost alone in H5, rejected, 0.93x vs its parent)" in prompt


def test_claims_that_name_a_change_the_params_do_not_make_are_sent_back(tmp_path):
    h2, h7 = PROPS["H2"], PROPS["H7"]
    assert hyp.claim_errors(h2["claim"], H0, Params(**h2["params"])) == ["it names FP16 accumulation, the params use FP32 accumulation"]
    assert hyp.claim_errors(h7["claim"], H0, Params(**h7["params"])) == ['"increasing block_n to 128" names a change the params do not make, block_n stays 128']
    assert hyp.claim_errors("Switching to approx GELU cuts the activation cost.", Params(**h2["params"]), Params(**h2["params"]) ) == [
        "it switches to approx GELU, which the start config already uses"]
    assert hyp.claim_errors("FP16 accumulation is faster than FP32 accumulation.", H0, Params(**H0.dict() | {"acc": "fp16"})) == []
    assert hyp.claim_errors("Decreasing warps from 8 to 4 frees registers.", Params(**H0.dict() | {"warps": 8}), H0) == []
    o = state(tmp_path, ["H0", "H1"], directives=("d1",), status={"speed": "running"})
    fixed = h2 | {"claim": "Switching to approx GELU with FP32 accumulation trims the activation cost."}
    o.ask = Script(answer("H2"), answer(fixed))
    h = o.propose(o.branches["speed"])
    assert h.claim == fixed["claim"]
    assert of(o, "agent_log")[-1]["text"] == ("Refused a proposal on speed: the claim does not match the change it makes (gelu exact to approx): "
                                              "it names FP16 accumulation, the params use FP32 accumulation.")
    assert o.ask.calls[1][2][-1]["content"].startswith("Invalid: the claim does not match the change it makes (gelu exact to approx)")


# ---- 2. post-mortems for every rejected or inconclusive run ----

def test_an_inconclusive_run_gets_a_code_generated_post_mortem(tmp_path):
    o = state(tmp_path, ["H0", "H1"], directives=("d1",), status={"speed": "running"},
              engine=lambda spec, run_id: Metrics.from_dict(RUNS["H2"]["metrics"]))
    o.ask = Script(answer(PROPS["H2"] | {"claim": "Switching to approx GELU trims the activation cost."}))
    o.iterate(o.branches["speed"])
    assert o.hyps[-1].judgment.verdict == "inconclusive"
    pm = of(o, "post_mortem")[-1]["post_mortem"]
    assert pm["hypothesis"] == "H2" and pm["cause"].startswith("It ran at 1.03x vs its parent, inside the +/- 9.0% noise band around the predicted 1.05x")
    assert pm["pivot"] == "Rerun gelu exact to approx with more timing rounds, or keep the parent and change a knob the ledger has not tried yet."
    assert not o.avoid  # an inconclusive run rules nothing out
    assert types(o).index("post_mortem") < types(o).index("help_requested")


# ---- 4. directive hunches ----

def test_a_hunch_for_a_config_being_measured_gets_one_plan_change_naming_that_run(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    o = state(tmp_path, ["H0", "H1"])
    add(o, "H2", judged=False)  # d3 arrived while H2, the same exact config, was being measured on speed
    o.add_directive("speed", DIRS["d3"]["text"])
    texts = [e["text"] for e in of(o, "plan_change") if e["help"] == "d3"]
    assert texts == [(f"Because the Speed lens sent \"{DIRS['d3']['text']}\", I am not running it again on speed2 because that exact config, 128x128x32 "
                      "tiles, 4 warps, 3 stages, FP32 accumulation, approx GELU, is being measured right now as H2 on speed.")]
    assert o.hunches["d3"] == "H2" and "speed2" not in o.hints and "d3" not in o.reserved


def test_hunch_duplicates_match_accumulation_and_gelu_and_queued_configs(tmp_path, monkeypatch):
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    o = state(tmp_path, ["H0", "H1"])
    o.add_directive("speed", "Try FP16 accumulation on the current tiles.")
    assert of(o, "plan_change")[-1]["text"].endswith("already ran as H1 on speed (rejected, 1.49x vs its parent).")
    queued = Params(**H0.dict() | {"gelu": "approx"})
    o.hints["prec"] = Hint("approx", "c", "hq9", {"gelu": "approx"}, queued)
    o.add_directive("speed", "Use the tanh GELU.")
    assert of(o, "plan_change")[-1]["text"].endswith("is already queued on prec for hq9.")
    assert [e["help"] for e in of(o, "plan_change")] == ["d3", "d4"]


def test_a_bound_directive_at_the_cap_still_gets_its_one_plan_change(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8"], directives=("d1", "d2", "d3"))
    o.add_directive("precision", "Keep max error under 5e-4.")
    assert [e["text"] for e in of(o, "plan_change") if e["help"] == "d4"] == [
        ("d4 sets a bound, not a specific change. No new hypotheses run on prec because the session reached its 8 hypothesis cap, "
        "so the bound applies to the logged runs only.")]


# ---- 5. a rejected branch stays rejected ----

def test_a_branch_rejected_while_its_proposal_is_in_flight_gets_no_hypothesis(tmp_path):
    o = state(tmp_path, ["H0", "H1"], directives=("d1",), status={"speed": "running"})
    o.hints["speed"] = Hint(DIRS["d1"]["text"], "Tested under the Speed lens bound set by d1", "d1", None)
    o.ask = Script(answer("H2"), hooks={"speed-": lambda: o.add_directive("precision", DIRS["d2"]["text"])})
    o.iterate(o.branches["speed"])
    assert o.branches["speed"].status == "rejected" and not of(o, "hypothesis_proposed") and len(o.hyps) == 2
    assert "speed was rejected while its next hypothesis was being proposed, so nothing more runs on it." in [e["text"] for e in of(o, "agent_log")]
    assert [e["status"] for e in of(o, "branch_status") if e["branch"] == "speed"] == ["rejected"]


def test_a_help_answer_never_reopens_a_rejected_branch(tmp_path):
    # H2 was being measured on speed when d2 rejected it, then came back inconclusive and asked for help
    o = state(tmp_path, ["H0", "H1", "H2"])
    o.request_help(o.hyps[2], o.branches["speed"], "speed")
    req = of(o, "help_requested")[-1]["help"]
    assert req["blocks"] == "speed2" and "speed was rejected while H2 ran, so an answer applies to speed2" in req["question"]
    o.answer({"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})
    assert o.branches["speed"].status == "rejected" and not [e for e in of(o, "branch_status") if e["branch"] == "speed"]
    assert of(o, "plan_change")[-1]["text"] == "Because the Speed lens wants more tile search, I am giving speed2 two more iterations. The 8 hypothesis cap has 5 left."
    o.reopen(o.branches["speed"])
    o.close(o.branches["speed"], "done")
    assert o.branches["speed"].status == "rejected"


# ---- 6. help answers promise only what the cap allows ----

def test_keep_tuning_at_hq2_says_no_iterations_fit_the_cap(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4", "H5", "H6", "H7"], directives=("d1", "d2", "d3"),
              status={"speed": "running", "speed2": "done", "merge": "waiting_on_human"})
    o.promises.append({"ref": "hq1", "branch": "speed", "n": 2, "start": 2})  # hq1 promised speed two iterations, none ran
    o.request_help(o.hyps[6], o.branches["merge"], "speed")
    o.answer({"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})
    assert of(o, "plan_change")[-1]["text"] == ("The Speed lens wants more tile search on merge, but the 8 hypothesis cap has 1 left and 2 of those are already "
                                                "promised, so no iterations fit and I am closing merge with its best configuration.")
    assert o.branches["merge"].status == "done" and "merge" not in o.extra and len(o.promises) == 1


def test_keep_tuning_at_hq1_promises_two_iterations_that_fit(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3"])
    assert o.room() == (2, "the 8 hypothesis cap has 4 left")
    o.max_hyps = 5
    assert o.room() == (1, "the 5 hypothesis cap has 1 left")


# ---- 7. hybrid ----

def test_the_recorded_hybrid_of_H2_and_H4_is_not_proposed(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H2", "H3", "H4"], directives=("d1", "d2", "d3"))
    assert o.merge_why_not(o.hyps[4]) == "0.99x vs its parent, under 1.00x"
    assert o.merge_why_not(o.hyps[1]) == "standard error 1.3e-2 is over the Precision lens bound 1.0e-3"
    assert o.merge_why_not(o.hyps[2]) is None  # 1.03x and under every bound, inconclusive or not
    o.maybe_hybrid()
    assert not of(o, "hybrid_proposed") and o.hybrid is None
    assert of(o, "agent_log")[-1]["text"].startswith("No hybrid of H0 on main (128x128x32 tiles, 4 warps, 3 stages, FP32 accumulation, exact GELU) "
                                                     "with approx GELU from H2 on speed: ") and "already ran as H2 on speed" in of(o, "agent_log")[-1]["text"]


def test_no_qualifying_parent_run_skips_the_hybrid_and_logs_why_at_the_end(tmp_path):
    o = state(tmp_path, ["H0", "H1", "H3", "H4"], directives=("d1", "d2"))
    o.maybe_hybrid()
    o.maybe_hybrid()
    assert not of(o, "agent_log")  # a later run may still qualify, so the reason waits for the end of the session
    o.finish()
    logs = [e["text"] for e in of(o, "agent_log")]
    assert logs == [("No hybrid: no run on another branch measured at least 1.00x over its parent while passing every active bound ("
                    "H1 on speed, acc fp32 to fp16: standard error 1.3e-2 is over the Precision lens bound 1.0e-3; "
                    "H3 on speed2, block_k 32 to 64, warps 4 to 8, stages 3 to 4: 0.97x vs its parent, under 1.00x; "
                    "H4 on prec, block_n 128 to 256, warps 4 to 8, stages 3 to 4: 0.99x vs its parent, under 1.00x).")]
    assert not of(o, "hybrid_proposed")


# ---- 8. Papiers ----

CONVERSATIONS = {"results": [
    {"kind": "conversation", "id": "cnv_6630997cd59e09a4331ad143fa6088c4", "title": "Conversation overview",
     "snippet": "Summarize this article ## Summary The article is a Triton tutorial on implementing a high-performance FP16 matrix-multiplication kernel."},
    {"kind": "conversation", "id": "cnv_fb1bac55818cfbd648a74abec8d21b7e", "title": "Conversation messages", "snippet": "Agent: something unrelated."}]}
DOCUMENTS = {"items": [{"id": "doc_1", "kind": "document", "title": "Matrix Multiplication \u2014 Triton  documentation"},
                       {"id": "doc_2", "kind": "document", "title": "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations"},
                       {"id": "doc_3", "kind": "document", "title": "MIXED PRECISION TRAINING"}]}


def test_conversations_are_searched_only_when_nothing_else_matched_and_named_by_their_document():
    from agent.papiers import Library
    cli = CLI({("search", "conversation", None): CONVERSATIONS, ("list", "document", None): DOCUMENTS})
    got = Library(cli).retrieve("tile size shared memory")
    searched = [a[a.index("--kind") + 1] for a in cli.calls if a[0] == "search"]
    assert searched[0] == "highlight" and searched[-1] == "conversation" and searched.count("conversation") == 1
    assert got[0]["document_title"] == "Matrix Multiplication \u2014 Triton  documentation"
    assert got[1]["document_title"] == "Conversation messages"  # no document shares two words with it
    hl = {"results": [{"kind": "document", "id": "doc_3", "title": "MIXED PRECISION TRAINING", "evidence": [
        {"kind": "highlight", "id": "hl_da32a6e4a587243f484621f05f9f81bb", "snippet": "accumulates the partial products into an FP32 value"}]}]}
    read = {"items": [{"id": "hl_da32a6e4a587243f484621f05f9f81bb", "ok": True, "content": "accumulates into FP32",
                       "metadata": {"annotationText": "FP16 accumulation breaks past $K = 1024$, costs $$"}}]}
    cli = CLI({("search", "highlight", None): hl, ("search", "conversation", None): CONVERSATIONS, ("read", None, None): read})
    got = Library(cli).retrieve("accumulation precision FP16")
    assert [a for a in cli.calls if "conversation" in a] == []
    assert got[0]["kind"] == "highlight" and got[0]["annotation"] == "FP16 accumulation breaks past K = 1024, costs"


def test_raw_papiers_ids_in_model_prose_become_the_document_title(tmp_path):
    item = {"id": "cnv_6630997cd59e09a4331ad143fa6088c4", "kind": "conversation", "document_title": "Matrix Multiplication \u2014 Triton  documentation",
            "title": "Matrix Multiplication \u2014 Triton  documentation", "excerpt": "A Triton tutorial on tiled matmul."}
    o = state(tmp_path, [], directives=(), context=lambda q: [item])
    o.branches["main"] = Branch("main", [], "Baseline", "both")
    o.ask = Script(answer(PROPS["H0"] | {"claim": "Per cnv_6630997cd59e09a4331ad143fa6088c4, 128x128x32 tiles with 4 warps and 3 stages are faster."}))
    o.engine = lambda spec, run_id: Metrics.from_dict(RUNS["H0"]["metrics"])
    o.iterate(o.branches["main"])
    hy = of(o, "hypothesis_proposed")[0]["hypothesis"]
    assert hy["claim"].startswith("Per \"Matrix Multiplication \u2014 Triton  documentation\", 128x128x32") and "cnv_" not in hy["note"]
    assert hy["note"] == "Context from the Papiers conversation \"Matrix Multiplication \u2014 Triton  documentation\"" and hy["papiers"] == item["id"]
    assert "cnv_" not in o.ask.calls[0][2][1]["content"]


# ---- the whole recording, replayed ----

def test_the_recording_replayed_keeps_every_invariant(tmp_path, monkeypatch):
    """The recording's proposals in order, the precision directive arriving while the model answers for speed (as d2 did), the tanh hunch
    arriving while a run is measured (as d3 did) and every help request answered with keep tuning."""
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    table = {json.dumps(r["params"], sort_keys=True): Metrics.from_dict(r["metrics"]) for r in REC["runs"]}
    table[json.dumps(Params().dict(), sort_keys=True)] = Metrics.from_dict(REC["baselines"]["naive"])
    sent = threading.Event()

    def engine(spec, run_id):
        if spec["target"] == "eager":
            return Metrics.from_dict(REC["baselines"]["eager"])
        if spec["target"] == "sweep":
            return Metrics(compiled_ok=True, errors={f"k{k}": 2e-3 * (k / 256) ** 0.5 for k in spec["ks"]})
        one = lambda p: table.get(json.dumps(p, sort_keys=True)) or Metrics.from_dict(RUNS["H5"]["metrics"])
        m = one(spec["params"])
        if spec.get("refs"):
            if len(o.directives) == 2 and not sent.is_set():
                sent.set()
                o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d3"]["text"]})
                wait(lambda: len(o.directives) == 3)
            ref = lambda r: Metrics.from_dict(REC["baselines"]["eager"]) if r["target"] == "eager" else one(r["params"])
            m = Metrics.from_dict(m.dict() | {"speedups": {k: ref(r).latency_ms / m.latency_ms for k, r in spec["refs"].items()},
                                              "spreads": {k: 0.0 for k in spec["refs"]}})
        return m

    model = [answer(i) for i in ("H0", "H2", "H3", "H4", "H5", "H7", "H8")]
    ask = Script(*model, hooks={"speed-": lambda: (o.command({"type": "add_directive", "lens": "precision", "text": DIRS["d2"]["text"]}),
                                                   wait(lambda: len(o.directives) == 2))})
    o = make(tmp_path, ask, engine=engine, main_iters=1, branch_iters=3, max_hyps=8, idle_s=0.5, help_deadline_s=30)
    t = threading.Thread(target=o.run)
    t.start()
    answered = set()
    end = time.time() + 20
    while t.is_alive() and time.time() < end:
        if o.hyps and o.hyps[0].judgment and not o.directives:
            o.command({"type": "add_directive", "lens": "speed", "text": DIRS["d1"]["text"]})
            wait(lambda: o.directives)
        for e in of(o, "help_requested"):
            if e["help"]["id"] not in answered:
                answered.add(e["help"]["id"])
                o.command({"type": "answer_help", "help": e["help"]["id"], "option": "tile_search", "by": "speed"})
        time.sleep(0.01)
    t.join(5)
    ev = o.log.events
    assert_contract(ev)
    # 3. the final report comes right before session_finished
    assert types(o)[-2:] == ["final_report", "session_finished"]
    # 2. every rejected or inconclusive hypothesis has a post-mortem
    judged = {h.id: h.judgment.verdict for h in o.hyps if h.judgment}
    assert {i for i, v in judged.items() if v != "confirmed"} <= {e["post_mortem"]["hypothesis"] for e in of(o, "post_mortem")}
    # 5. nothing is proposed on, or reopens, a branch after it was rejected
    rejected_at = {}
    for n, e in enumerate(ev):
        if e["type"] == "branch_status" and e["branch"] in rejected_at:
            raise AssertionError(f"{e['branch']} changed status after it was rejected: {e}")
        if e["type"] == "branch_status" and e["status"] == "rejected":
            rejected_at[e["branch"]] = n
        if e["type"] == "hypothesis_proposed":
            assert e["hypothesis"]["branch"] not in rejected_at
    assert "speed" in rejected_at
    # 1. no proposal repeats a rejected single-knob change, and every model claim matches its change
    for h in o.hyps:
        base = o.baseline(h)[0]
        if "Built in code" not in (h.note or ""):
            assert_no_losing_repeat(o, h)
        if not (h.note or "").startswith(("Harness-seeded", "Hybrid")):
            assert hyp.claim_errors(h.claim.split(" Relative to the pivot config")[0], base, h.params) == [], (h.id, h.claim)
    # 4. exactly one plan_change per directive
    for d in o.directives:
        assert [e["help"] for e in of(o, "plan_change")].count(d["id"]) == 1, d["id"]
    # 6. nothing promised in a help answer was cut by the cap
    assert not any("cap ended the session before" in e["text"] for e in of(o, "plan_change"))
    # 7. a proposed hybrid only merges a change that measured at least 1.00x and passes every active bound
    if o.hybrid:
        assert all(o.merge_why_not(h) is None for h in o.hyps if h.params != o.hybrid["base"] and h.branch in o.hybrid["branches"]
                   and all(getattr(h.params, k) == v for k, v in o.hybrid["params"].dict().items() if v != getattr(o.hybrid["base"], k)))
    # 8. no raw Papiers id leaks into prose
    assert not any("cnv_" in json.dumps(e) for e in ev if e["type"] != "papiers_context")


def wait(pred, timeout=5):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        time.sleep(0.01)
    assert pred()
