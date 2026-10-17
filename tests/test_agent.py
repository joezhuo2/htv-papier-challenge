import json
import threading
import time

import pytest

from agent import hypothesis as hyp
from agent.events import Log
from agent.orchestrator import QUERIES, Orchestrator
from agent.verdict import judge
from harness.metrics import Metrics

BAND = 0.0165
P = {"block_m": 128, "block_n": 128, "block_k": 32, "warps": 4, "stages": 3, "acc": "fp32", "gelu": "exact"}


def met(latency, err=4.9e-4, adv=5.2e-4, ok=True):
    return Metrics(compiled_ok=ok, correct_ok=err < 1e-3, latency_ms=latency, tflops=17.18 / latency, max_error=err, adv_error=adv, vram_mb=96.0)


TABLE = {
    "naive": met(1.891),
    "eager": met(0.669, err=1.5e-3, adv=2.5e-3),
    json.dumps(P, sort_keys=True): met(0.642),
    json.dumps(P | {"stages": 4}, sort_keys=True): met(0.640),
    json.dumps(P | {"acc": "fp16"}, sort_keys=True): met(0.450, err=1.3e-2, adv=7.1e-2),
    json.dumps(P | {"gelu": "approx"}, sort_keys=True): met(0.642, err=6.9e-4, adv=7.2e-4),
    json.dumps(P | {"block_n": 64}, sort_keys=True): met(0.500, err=5e-4, adv=3e-3),
    json.dumps(P | {"warps": 8}, sort_keys=True): met(0.628),
}


class Engine:
    def __init__(self, table=TABLE):
        self.table, self.calls = dict(table), []

    def __call__(self, spec, run_id):
        self.calls.append((spec, run_id))
        m = self.one(spec)
        if spec.get("refs") and m.latency_ms:
            # same-window speedups, noiseless in the fake
            m = Metrics.from_dict(m.dict() | {"speedups": {k: self.one(r).latency_ms / m.latency_ms for k, r in spec["refs"].items()},
                                              "spreads": {k: 0.0 for k in spec["refs"]}})
        return m

    def one(self, spec):
        if spec["target"] == "eager":
            return self.table["eager"]
        if spec["target"] == "sweep":
            return Metrics(compiled_ok=True, errors={f"k{k}": 2e-3 * (k / 256) ** 0.5 for k in spec["ks"]})
        key = json.dumps(spec["params"], sort_keys=True)
        if spec["params"]["block_m"] == 16:
            return self.table["naive"]
        if spec["params"]["acc"] == "fp16" and key not in self.table:
            return met(0.46, err=1.3e-2, adv=7e-2)
        return self.table.get(key) or Metrics(error="exit 1: out of resources")


def h(params, min_speedup=1.3, vs="parent_best", claim="A claim."):
    return json.dumps({"claim": claim, "prediction": {"min_speedup": min_speedup, "vs": vs, "max_error": 1e-3}, "params": params})


class Asker:
    """Fast model answers come from a script in order. The strong model fails unless strong_ok, which exercises the factual fallback."""

    def __init__(self, *script, strong_ok=False, default="no json"):
        self.script, self.strong_ok, self.default, self.calls = list(script), strong_ok, default, []

    def __call__(self, messages, model, tag):
        self.calls.append((model, tag, messages))
        if model == "strong":
            if not self.strong_ok:
                raise TimeoutError("slow model")
            return json.dumps({"cause": "Mechanism sentence one. Sentence two.", "pivot": "Go back to fp32.",
                               "uncertainties": ["one shape only"], "would_change_mind": "A second adversarial set over the bound."})
        if not self.script:
            if self.default:
                return self.default
            raise AssertionError("Asker ran out of scripted responses")
        return self.script.pop(0)


def make(tmp_path, asker, engine=None, **kw):
    log = Log(tmp_path / "events.jsonl")
    return Orchestrator(engine or Engine(), asker, log, "sessions/test", "fast", "strong", BAND, **({"retry_s": 0.05, "idle_s": 0.3} | kw))


def types(o):
    return [e["type"] for e in o.log.events]


def of(o, t):
    return [e for e in o.log.events if e["type"] == t]


REQUIRED = {
    "session_started": {"baselines", "branch"}, "directive_added": {"directive"}, "conflict_detected": {"directives", "reason"},
    "branch_created": {"branch"}, "branch_status": {"branch", "status"}, "hypothesis_proposed": {"hypothesis"},
    "run_started": {"run", "branch", "hypothesis"}, "run_finished": {"run"},
    "verdict": {"hypothesis", "verdict", "provisional", "reason", "checks", "bound"}, "post_mortem": {"post_mortem"}, "summary": {"summary"},
    "agent_log": {"text"}, "session_finished": {"ledger", "best"}, "papiers_context": set(), "findings_exported": {"items"},
    "help_requested": {"help"}, "help_answered": {"help", "option", "by"}, "help_cancelled": {"help", "reason"}, "plan_change": {"help", "text"}, "agent_error": {"kind", "text"},
    "hybrid_proposed": {"branches", "text"}, "final_report": {"report"},
}
HELP_KEYS = {"id", "target", "question", "evidence", "options", "default", "blocks", "deadline_s", "what_changes"}
SUMMARY_KEYS = {"branch", "hypothesis", "tried", "moved", "verdict", "confidence", "confidence_reason", "uncertainties", "would_change_mind", "next_step"}


def assert_contract(events):
    ts = [e["t"] for e in events]
    assert ts == sorted(ts) and all(isinstance(t, int) for t in ts)
    for e in events:
        assert e["type"] in REQUIRED, e["type"]
        assert REQUIRED[e["type"]] <= e.keys(), e
        assert "mock" not in e
        if e["type"] == "summary":
            assert set(e["summary"]) == SUMMARY_KEYS
        if e["type"] == "run_finished":
            assert set(e["run"]["metrics"]) - {"speedups", "spreads", "n_regs", "n_spills"} == {"latency_ms", "tflops", "max_error", "adv_error", "vram_mb", "compiled_ok", "correct_ok"}
        if e["type"] in ("verdict", "hypothesis_proposed"):
            b = e["bound"] if e["type"] == "verdict" else e["hypothesis"]["prediction"]["bound"]
            assert set(b) == {"standard", "adversarial", "set_by"} and b["set_by"] in ("speed", "precision", "default")
        if e["type"] == "help_requested":
            assert set(e["help"]) == HELP_KEYS
        if e["type"] == "findings_exported":
            assert all(set(i) == {"kind", "title", "file"} and i["file"].startswith("exports/") for i in e["items"])
        if e["type"] == "final_report":
            assert set(e["report"]) == {"best", "ruled_out", "next", "human", "hunches"}
            assert all(set(x) == {"directive", "lens", "text", "outcome"} for x in e["report"]["hunches"])
        if e["type"] == "post_mortem":
            assert set(e["post_mortem"]) - {"note"} == {"hypothesis", "expected", "observed", "cause", "pivot", "sweep"}
        if e["type"] == "papiers_context" and "note" in e:
            assert {"id", "title", "excerpt"} <= set(e["note"]) <= {"id", "title", "excerpt", "kind", "document_title", "highlight_text", "annotation"}


# ---- verdicts ----

def test_verdict_confirmed_rejected_inconclusive():
    pred = {"min_speedup": 2.0, "vs": "naive", "max_error": 1e-3}
    assert judge(pred, met(0.642), 1.891 / 0.642, BAND, 1e-3, None).verdict == "confirmed"
    assert judge(pred, met(0.95), 1.891 / 0.95, BAND, 1e-3, None).verdict == "inconclusive"
    j = judge(pred, met(1.2), 1.891 / 1.2, BAND, 1e-3, None)
    assert j.verdict == "rejected" and j.checks[0].passed is False


def test_verdict_rejects_on_error_and_failed_runs_regardless_of_speed():
    pred = {"min_speedup": 1.3, "vs": "parent_best", "max_error": 1e-3}
    j = judge(pred, met(0.45, err=1.3e-2), 0.642 / 0.45, BAND, 1e-3, None)
    assert j.verdict == "rejected" and [c.passed for c in j.checks] == [True, False]
    assert judge(pred, Metrics(error="timeout"), 0.642 / 1, BAND, 1e-3, None).verdict == "rejected"


def test_verdict_adversarial_is_provisional_until_a_precision_bound():
    pred = {"min_speedup": 1.2, "vs": "parent_best", "max_error": 1e-3}
    m = met(0.5, err=5e-4, adv=3e-3)
    j = judge(pred, m, 0.642 / 0.5, BAND, 1e-3, None)
    assert (j.verdict, j.provisional) == ("confirmed", True)
    j = judge(pred, m, 0.642 / 0.5, BAND, 1e-3, 1e-3)
    assert j.verdict == "rejected" and j.checks[-1].name == "adversarial max error"


def test_speed_tolerance_makes_fp16_provisional_and_precision_rejects_it():
    pred = {"min_speedup": 1.3, "vs": "parent_best", "max_error": 2e-2}
    fp16 = met(0.45, err=1.3e-2, adv=7.1e-2)
    j = judge(pred, fp16, 0.642 / 0.45, BAND, 2e-2, None)
    assert (j.verdict, j.provisional) == ("confirmed", True) and j.checks[1].predicted == "< 2.0e-2"
    j = judge(pred, fp16, 0.642 / 0.45, BAND, 1e-3, 1e-3)
    assert j.verdict == "rejected" and [c.passed for c in j.checks] == [True, False, False]
    assert (judge(pred, met(0.6), 0.642 / 0.6, BAND, 2e-2, None).provisional) is False


# ---- parsing ----

def test_parse_accepts_fenced_and_refuses_free_form_code(fake_llm):
    r = fake_llm.responses
    assert hyp.parse(r["valid_hypothesis"])[0]["params"].block_m == 128
    assert hyp.parse(r["fenced_hypothesis"])[0]["params"].gelu == "approx"
    assert hyp.parse(r["malformed_json"])[0] is None
    assert hyp.parse(r["free_form_code"])[1].startswith("invalid params")


def test_directive_parsing():
    assert hyp.directive("speed", "Push throughput harder") == {"constraint": "maximize_speed", "threshold": 2e-2}
    assert hyp.directive("speed", "Push throughput, tolerate 5e-3") == {"constraint": "maximize_speed", "threshold": 5e-3}
    assert hyp.directive("speed", "anything goes, error up to 1.0e-0")["threshold"] == 5e-2
    assert hyp.directive("speed", "be loose")["threshold"] == 2e-2
    assert hyp.directive("speed", "tolerate 9e-1")["threshold"] == 5e-2
    assert hyp.directive("precision", "relax to 1e-2")["threshold"] == 1e-3
    assert hyp.directive("precision", "Enforce max error under 1e-3 on both batches")["threshold"] == 1e-3
    assert hyp.directive("precision", "keep it under 0.0005")["threshold"] == 5e-4
    assert hyp.directive("precision", "be strict")["threshold"] == 1e-3


# ---- the loop ----

def test_one_autonomous_iteration(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive", "Bigger tiles help.")), main_iters=1)
    o.run()
    assert [t for t in types(o) if t != "agent_log"] == ["session_started", "hypothesis_proposed", "run_started", "run_finished",
                                                          "verdict", "summary", "branch_status", "findings_exported", "final_report",
                                                          "session_finished"]
    v = of(o, "verdict")[0]
    assert v["verdict"] == "confirmed" and v["checks"][0]["observed"] == "2.95x (noise +/- 0.05x)"
    assert of(o, "session_finished")[0]["best"] == {"run": "r1", "vs_naive": 2.95, "vs_eager": 1.04}
    assert_contract(o.log.events)
    lines = (tmp_path / "events.jsonl").read_text().splitlines()
    assert [json.loads(x) for x in lines] == o.log.events


def test_llm_cannot_choose_the_verdict(tmp_path, fake_llm):
    # The response claims success in prose, but the measured standard error decides.
    o = make(tmp_path, Asker(h(P | {"acc": "fp16"}, 1.3, "naive", "This will be a resounding success.")), main_iters=1)
    o.run()
    assert of(o, "verdict")[0]["verdict"] == "rejected"


def test_invalid_responses_are_retried_then_the_branch_closes(tmp_path, fake_llm):
    r = fake_llm.responses
    o = make(tmp_path, Asker(r["malformed_json"], r["free_form_code"], "no json here"), main_iters=1)
    o.run()
    assert "hypothesis_proposed" not in types(o)
    assert (of(o, "branch_status")[-1]["branch"], of(o, "branch_status")[-1]["status"]) == ("main", "done")


def run_with(o, script):
    """Runs the session in a thread, sending each directive once n hypotheses have verdicts."""
    t = threading.Thread(target=o.run)
    t.start()
    while t.is_alive():
        judged = sum(x.judgment is not None for x in o.hyps)
        while script and judged >= script[0][0]:
            _, lens, text = script.pop(0)
            o.command({"type": "add_directive", "lens": lens, "text": text})
        time.sleep(0.01)


def test_failure_path_fp16_provisional_under_speed_then_rejected_by_precision(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"acc": "fp16"}), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05), default="no json"),
             main_iters=1, branch_iters=2)
    run_with(o, [(1, "speed", "Push throughput harder than tile tuning alone."), (2, "precision", "Max error under 1e-3 on both batches.")])
    assert of(o, "hypothesis_proposed")[1]["hypothesis"]["prediction"]["bound"] == {"standard": 2e-2, "adversarial": None, "set_by": "speed"}
    assert [e["bound"]["set_by"] for e in of(o, "verdict") if e["hypothesis"] == "H1"] == ["speed", "precision"]
    h1 = [e for e in of(o, "verdict") if e["hypothesis"] == "H1"]
    assert [(e["verdict"], e["provisional"]) for e in h1] == [("confirmed", True), ("rejected", False)]
    assert h1[0]["checks"][1] == {"name": "standard max error", "predicted": "< 2.0e-2", "observed": "1.3e-2", "pass": True}
    assert [c["pass"] for c in h1[1]["checks"]] == [True, False, False]
    assert of(o, "conflict_detected")[0]["directives"] == ["d1", "d2"]
    assert "on the standard and adversarial batches" in of(o, "conflict_detected")[0]["reason"]
    assert [e["bound"]["set_by"] for e in of(o, "verdict") if e["hypothesis"] == "H0"] == ["default", "precision"]  # every run is re-judged
    assert types(o).index("branch_created", types(o).index("directive_added", types(o).index("directive_added") + 1)) < types(o).index("conflict_detected")
    pm = next(e["post_mortem"] for e in of(o, "post_mortem") if e["post_mortem"]["hypothesis"] == "H1")
    assert [s["k"] for s in pm["sweep"]] == [256, 512, 1024, 2048, 4096]
    assert "Drop FP16 accumulation" in pm["pivot"]
    assert ("speed", "rejected") in [(e["branch"], e["status"]) for e in of(o, "branch_status")]
    pivot_prompt = next(m for model, tag, m in o.ask.calls if tag.startswith("speed2-"))
    assert "FP16 accumulation (H1 rejected" in pivot_prompt[1]["content"] and "under 1e-03" in pivot_prompt[1]["content"]
    assert of(o, "session_finished")[0]["ledger"][1]["history"] == ["confirmed", "rejected"]
    assert_contract(o.log.events)


def test_a_converged_session_proposes_a_hybrid_at_the_cap_and_runs_it_outside_the_cap(tmp_path):
    w8, approx = P | {"warps": 8}, P | {"gelu": "approx"}
    merged = w8 | {"gelu": "approx"}
    engine = Engine(TABLE | {json.dumps(w8, sort_keys=True): met(0.600), json.dumps(approx, sort_keys=True): met(0.610, err=6.9e-4, adv=7.2e-4),
                             json.dumps(merged, sort_keys=True): met(0.550, err=6.9e-4, adv=7.2e-4)})
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(approx, 1.02), h(w8, 1.02), default="no json"), engine=engine, main_iters=1, branch_iters=3,
             help_deadline_s=5, max_hyps=4)
    proposed = lambda: bool(of(o, "hybrid_proposed"))
    drive(o, [(judged(o, 1), {"type": "add_directive", "lens": "precision", "text": "Max error under 1e-3 on both batches."}),
              (judged(o, 1), SPEED), (proposed, {"type": "approve_hybrid", "branches": ["speed2", "prec"]})])
    assert not of(o, "conflict_detected")  # the lenses never disagreed
    hy = of(o, "hybrid_proposed")[0]
    assert hy["branches"] == ["speed2", "prec"] and "best confirmed config, H3 on speed2" in hy["text"] and "approx GELU from H1 on prec" in hy["text"]
    run = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "merge")
    assert run["params"] == merged and run["id"] == "H4" and o.hyps[-1].bonus and sum(not x.bonus for x in o.hyps) == 4
    assert next(e for e in of(o, "verdict") if e["hypothesis"] == "H4")["verdict"] == "confirmed"
    assert {("speed2", "merged"), ("prec", "merged"), ("merge", "done")} <= {(e["branch"], e["status"]) for e in of(o, "branch_status")}
    assert_contract(o.log.events)


def hybrid_state(tmp_path, donor):
    """H0 on main, the best confirmed H1 on prec (256x128x64, 8 warps, 3 stages), and a confirmed single-knob donor H2 on speed2."""
    from agent.orchestrator import Branch, Hyp
    from agent.verdict import Judgment
    from harness.kernel import Params
    o = make(tmp_path, Asker())
    o.naive = met(2.0)
    for b in (Branch("main", [], "Baseline", "both"), Branch("prec", ["main"], "Precision lens", "precision"), Branch("speed2", ["main"], "Speed lens", "speed")):
        o.branches[b.id] = b
    base = Params(128, 128, 64, 4, 4)
    for i, (bid, p, lat) in enumerate([("main", base, 0.78), ("prec", Params(256, 128, 64, 8, 3), 0.60), ("speed2", donor, 0.74)]):
        x = Hyp(f"H{i}", bid, "c", {"min_speedup": 1.02, "vs": "parent_best", "max_error": 1e-3}, p, metrics=met(lat),
                speedup=0.78 / lat if i else 2.5, ref="parent" if i else "naive", base=base if i else None)
        x.judgment = Judgment("confirmed", False, "Prediction holds.")
        o.hyps.append(x)
    return o


def test_the_hybrid_is_the_best_config_plus_the_best_single_knob_change_from_another_branch(tmp_path):
    from harness.kernel import Params
    o = hybrid_state(tmp_path, Params(128, 128, 64, 4, 4, gelu="approx"))
    o.maybe_hybrid()
    assert o.hybrid["params"] == Params(256, 128, 64, 8, 3, gelu="approx") and o.hybrid["branches"] == ["prec", "speed2"]
    assert of(o, "hybrid_proposed")[0]["text"].startswith("Combine the best confirmed config, H1 on prec (256x128x64 tiles, 8 warps, 3 stages, "
                                                          "FP32 accumulation, exact GELU) with approx GELU from H2 on speed2")


def test_a_hybrid_over_the_accumulator_guard_is_skipped_and_explained_once(tmp_path):
    from harness.kernel import Params
    o = hybrid_state(tmp_path, Params(128, 256, 64, 4, 4))
    o.maybe_hybrid()
    o.maybe_hybrid()
    assert o.hybrid is None and not of(o, "hybrid_proposed")
    logs = [e["text"] for e in of(o, "agent_log") if e["text"].startswith("No hybrid")]
    assert logs == [("No hybrid of H1 on prec (256x128x64 tiles, 8 warps, 3 stages, FP32 accumulation, exact GELU) with block_n 256 from H2 on speed2: "
                    "256x256x64 tiles, 8 warps, 2 stages, FP32 accumulation, exact GELU puts 256 FP32 accumulator elements on each of 256 threads, over 128.")]


def test_a_hybrid_that_already_ran_is_not_proposed(tmp_path):
    from harness.kernel import Params
    o = hybrid_state(tmp_path, Params(128, 128, 64, 4, 4, gelu="approx"))
    from agent.orchestrator import Hyp
    o.hyps.append(Hyp("H3", "main", "c", {}, Params(256, 128, 64, 8, 3, gelu="approx"), metrics=met(0.61)))
    o.maybe_hybrid()
    assert o.hybrid is None and "already ran as H3 on main" in of(o, "agent_log")[-1]["text"]


def test_harness_variant_tests_fp16_when_the_agent_avoids_it(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05), default="no json"),
             main_iters=1, branch_iters=3)
    run_with(o, [(1, "speed", "Go faster.")])
    on_speed = [e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "speed"]
    variant = on_speed[0]
    assert variant["params"] == P | {"acc": "fp16"} and variant["note"].startswith("Harness-seeded variant, not proposed by the model")
    assert "Section" not in variant["note"]
    assert variant["claim"] == ("FP16 accumulation on 128x128x32 tiles, 4 warps, 3 stages with exact GELU cuts latency by at least 1.20x "
                                "against the same config with FP32 accumulation, with standard max error under 2.0e-2, the Speed lens bound.")
    v = next(e for e in of(o, "verdict") if e["hypothesis"] == variant["id"])
    assert (v["verdict"], v["provisional"]) == ("confirmed", True)


def test_precision_directive_rejects_a_provisional_result_without_a_new_run(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"block_n": 64}, 1.2), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05)),
             main_iters=1, branch_iters=1, help_deadline_s=0.3)
    run_with(o, [(1, "speed", "Push throughput harder."), (2, "precision", "Max error under 1e-3 on the adversarial batch too.")])
    h1 = [e for e in of(o, "verdict") if e["hypothesis"] == "H1"]
    assert [(e["verdict"], e["provisional"]) for e in h1] == [("confirmed", True), ("rejected", False)]
    assert of(o, "conflict_detected")[0]["directives"] == ["d1", "d2"]
    runs_before = sum(1 for e in o.log.events[: o.log.events.index(h1[1])] if e["type"] == "run_started")
    assert runs_before == 2
    assert {"speed2", "prec"} <= {e["branch"]["id"] for e in of(o, "branch_created")}
    assert of(o, "session_finished")[0]["ledger"][1]["history"] == ["confirmed", "rejected"]
    assert_contract(o.log.events)


def test_explanation_uses_the_strong_model_when_it_answers(tmp_path):
    o = make(tmp_path, Asker(h(P | {"acc": "fp16"}, 1.3, "naive"), strong_ok=True), main_iters=1)
    o.run()
    assert of(o, "post_mortem")[0]["post_mortem"]["cause"] == "Mechanism sentence one. Sentence two."
    assert not any(tag.startswith("sum-") for _, tag, _ in o.ask.calls)  # summaries are computed in code


def test_hybrid_approval_without_a_proposal_is_ignored(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1)
    o.command({"type": "approve_hybrid", "branches": ["a", "b"]})
    o.run()
    assert "merge" not in o.branches and "approve_hybrid" in (tmp_path / "debug.log").read_text()


# ---- server ----

def test_websocket_sends_history_then_streams_and_accepts_commands(tmp_path):
    from fastapi.testclient import TestClient

    from server.app import create

    gate = threading.Event()

    class Gated(Orchestrator):
        def iterate(self, b):
            gate.wait(5)
            super().iterate(b)

    def factory():
        return Gated(Engine(), Asker(h(P, 2.0, "naive"), default="no json"), Log(tmp_path / "events.jsonl"), "sessions/test", "fast", "strong", BAND,
                     main_iters=1, idle_s=1.0)

    with TestClient(create(factory, autostart=True)) as client:
        deadline = time.time() + 5
        while client.get("/health").json()["events"] < 3 and time.time() < deadline:
            time.sleep(0.02)
        with client.websocket_connect("/ws") as ws:
            history = json.loads(ws.receive_text())
            assert [e["type"] for e in history] == ["agent_log", "session_started", "agent_log"]
            ws.send_text("not json")
            ws.send_text(json.dumps({"type": "add_directive", "lens": "speed", "text": "Go faster."}))
            gate.set()
            seen = []
            # the directive can apply while H0's model call runs, before or after hypothesis_proposed
            while not {"directive_added", "hypothesis_proposed"} <= set(seen):
                seen.append(json.loads(ws.receive_text())["type"])


def test_pivots_stop_at_three_branches_per_lens(tmp_path):
    fp16 = {"acc": "fp16"}
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), *[h(P | fp16 | {"warps": w}, claim=f"Unlike H1, H2 and H3, {w} warps.") for w in (1, 2, 8, 16)],
                             default="no json"), main_iters=1, branch_iters=1)
    run_with(o, [(1, "precision", "Strict."), (1, "speed", "Faster.")])
    assert [b for b in o.branches if b.startswith("speed")] == ["speed", "speed2", "speed3"]


def test_server_command_validation():
    from server.app import valid
    assert valid({"type": "add_directive", "lens": "speed", "text": "Go."})
    assert not valid({"type": "add_directive", "lens": "both", "text": "Go."})
    assert not valid({"type": "add_directive", "lens": "speed", "text": "  "})
    assert not valid({"type": "rm_rf"})
    assert not valid(["add_directive"])


def test_speedup_miss_is_not_a_branch_failure(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.5), h(P | {"warps": 8}, 1.3), default="no json"), main_iters=1, branch_iters=3)
    run_with(o, [(1, "speed", "Faster.")])
    ev = o.log.events
    v1 = next(i for i, e in enumerate(ev) if e["type"] == "verdict" and e["hypothesis"] == "H2")
    assert ev[v1]["verdict"] == "rejected" and ev[v1]["checks"][0]["pass"] is False
    nxt = next(i for i, e in enumerate(ev) if i > v1 and e["type"] == "hypothesis_proposed")
    between = [e["type"] for e in ev[v1:nxt]]
    assert "branch_created" not in between and ("speed", "rejected") not in [(e["branch"], e["status"]) for e in of(o, "branch_status")]
    assert ev[nxt]["hypothesis"]["branch"] == "speed"
    pm = next(e["post_mortem"] for e in ev[v1:nxt] if e["type"] == "post_mortem")
    assert pm["hypothesis"] == "H2" and pm["sweep"] == [] and pm["observed"].startswith("1.00x vs the parent")
    assert pm["cause"].startswith("It ran at 1.00x vs its parent, faster but short of the predicted 1.50x") and "Changed stages 3 to 4" in pm["cause"]
    assert "a deeper pipeline holds more shared memory per block" in pm["cause"] and pm["pivot"].startswith("Revert stages 3 to 4")
    assert any(i["file"] == "exports/H2-negative-result.md" for e in of(o, "findings_exported") for i in e["items"])


# ---- Papiers ----

def test_papiers_results_parsing():
    from agent.papiers import results
    out = json.dumps({"results": [{"kind": "conversation", "id": "cnv_1", "title": "Conversation overview", "snippet": " Triton tutorial "},
                                  {"kind": "note", "id": "nte_2", "title": None, "snippet": None}]})
    assert results(out) == [{"id": "cnv_1", "kind": "conversation", "title": "Conversation overview", "excerpt": "Triton tutorial"},
                            {"id": "nte_2", "kind": "note", "title": "nte_2", "excerpt": ""}]


def test_papiers_context_is_read_once_cited_and_prompted(tmp_path):
    queries = []

    def context(q):
        queries.append(q)
        return [{"id": "cnv_1", "title": "Triton tutorial", "excerpt": "Accumulate in FP32 for accuracy. [[src:P15:31]]"}]

    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), default="no json"), main_iters=2, context=context)
    o.run()
    notes = of(o, "papiers_context")
    assert notes == [{**notes[0], "query": QUERIES["both"], "note": {"id": "cnv_1", "title": "Triton tutorial", "excerpt": "Accumulate in FP32 for accuracy."}}]
    assert len(queries) == 1
    assert of(o, "hypothesis_proposed")[0]["hypothesis"]["note"] == "Context from Papiers note cnv_1"  # old-shape items have no kind
    assert "Papiers cnv_1 (Triton tutorial): Accumulate in FP32 for accuracy." in o.ask.calls[0][2][1]["content"]
    assert_contract_papiers(o.log.events)


def assert_contract_papiers(events):
    assert_contract(events)


def test_papiers_failure_does_not_stop_the_loop(tmp_path):
    def context(q):
        raise subprocess_error()

    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, context=context)
    o.run()
    assert any("Papiers search unavailable" in e["text"] for e in of(o, "agent_log"))
    assert not of(o, "papiers_context")
    assert of(o, "verdict")[0]["verdict"] == "confirmed"


def subprocess_error():
    import subprocess
    return subprocess.TimeoutExpired("papiers", 30)


def test_empty_papiers_search_is_logged_without_a_note(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, context=lambda q: [])
    o.run()
    assert [(e.get("query"), "note" in e) for e in of(o, "papiers_context")] == [(QUERIES["both"], False)]
    assert of(o, "hypothesis_proposed")[0]["hypothesis"].get("note") is None


@pytest.mark.papiers
def test_real_papiers_search_returns_items():
    from agent import papiers
    assert isinstance(papiers.search("Triton matrix multiplication"), list)


def test_implausible_predictions_are_refused():
    assert hyp.parse(h(P, 15.0))[1].startswith("min_speedup 15.0")
    assert hyp.parse(h(P, 0.15))[1].startswith("min_speedup 0.15")
    assert hyp.parse(h(P, 3.0, "naive"))[0]


def test_compile_failure_is_a_miss_with_a_templated_post_mortem(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"block_k": 64}, 1.1), strong_ok=True, default="no json"), main_iters=1, branch_iters=3)
    run_with(o, [(1, "speed", "Faster.")])
    v = next(e for e in of(o, "verdict") if e["hypothesis"] == "H2")
    assert v["verdict"] == "rejected" and v["checks"][0]["name"] == "compiles and runs"
    pm = next(e["post_mortem"] for e in of(o, "post_mortem") if e["post_mortem"]["hypothesis"] == "H2")
    assert pm["sweep"] == [] and pm["cause"].startswith("The run failed before measuring")
    assert not any(tag == "pm-H2" for _, tag, _ in o.ask.calls)
    assert "speed2" not in o.branches


# ---- help requests ----

INCONCLUSIVE = P | {"warps": 8}  # 0.628 ms vs 0.642 ms: 1.022x, inside the noise band around a 1.02x prediction


def drive(o, steps, timeout=10):
    """Runs the session, sending each (condition, command) once its condition holds, in order."""
    t = threading.Thread(target=o.run)
    t.start()
    end = time.time() + timeout
    while t.is_alive() and time.time() < end:
        if steps and steps[0][0]():
            o.command(steps.pop(0)[1])
        time.sleep(0.01)
    t.join(timeout)
    assert not steps, "a step never became ready"


judged = lambda o, n: lambda: sum(x.judgment is not None for x in o.hyps) >= n
asked = lambda o: lambda: bool(of(o, "help_requested"))
SPEED = {"type": "add_directive", "lens": "speed", "text": "Faster."}


def help_session(tmp_path, *script, deadline=30.0, **kw):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(INCONCLUSIVE, 1.02), *script, default="no json"), main_iters=1, branch_iters=2,
             help_deadline_s=deadline, **kw)
    return o


def wait_for(o, pred, timeout=5):
    end = time.time() + timeout
    while not pred() and time.time() < end:
        time.sleep(0.01)
    assert pred()


def test_inconclusive_verdict_asks_the_branch_lens_and_blocks_only_that_branch(tmp_path):
    o = help_session(tmp_path, deadline=0.5)
    run_with(o, [(1, "speed", "Faster.")])
    req = of(o, "help_requested")[0]["help"]
    assert req["target"] == "speed" and req["blocks"] == "speed" and req["default"] == "tile_search"
    assert "inside the noise band" in req["question"] and req["evidence"][0].startswith("r3:")
    ev = o.log.events
    i = ev.index(of(o, "help_requested")[0])
    assert ev[i + 1] == {**ev[i + 1], "type": "branch_status", "branch": "speed", "status": "waiting_on_human"}
    assert of(o, "verdict")[2]["verdict"] == "inconclusive"
    # no answer: the deadline passes and the branch proceeds on the default
    pc = next(e for e in of(o, "plan_change") if e["help"] == "hq1")
    assert pc["help"] == "hq1" and "default assumption" in pc["text"]
    assert ("speed", "running") in [(e["branch"], e["status"]) for e in ev[i:] if e["type"] == "branch_status"]
    assert_contract(ev)


def test_suggestion_answer_becomes_the_next_hypothesis_and_is_exported(tmp_path):
    o = help_session(tmp_path, h(P | {"gelu": "approx"}, 1.05))
    drive(o, [(judged(o, 1), SPEED), (asked(o), {"type": "answer_help", "help": "hq1", "option": "other", "note": "Try the tanh approximate GELU.", "by": "speed"})])
    ans = of(o, "help_answered")[0]
    assert ans == {**ans, "help": "hq1", "option": "other", "note": "Try the tanh approximate GELU.", "by": "speed"}
    assert "suggests" in next(e for e in of(o, "plan_change") if e["help"] == "hq1")["text"]
    nxt = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["t"] >= ans["t"])
    assert nxt["note"] == "Suggested by the Speed lens in answer to hq1" and nxt["params"]["gelu"] == "approx"
    prompt = next(m for model, tag, m in o.ask.calls if tag.startswith("speed-3"))
    assert "Try the tanh approximate GELU." in prompt[1]["content"]  # a first test goes to the model
    files = [i["file"] for e in of(o, "findings_exported") for i in e["items"]]
    assert "exports/hq1-answer.md" in files
    assert "Try the tanh approximate GELU." in (tmp_path / "exports" / "hq1-answer.md").read_text()
    assert_contract(o.log.events)


def test_accept_closes_the_branch_and_unknown_answers_are_ignored(tmp_path):
    o = help_session(tmp_path)
    a = lambda hq, opt: {"type": "answer_help", "help": hq, "option": opt, "by": "speed"}
    drive(o, [(judged(o, 1), SPEED), (asked(o), a("hq9", "accept")), (lambda: True, a("hq1", "accept")), (lambda: True, a("hq1", "tile_search"))])
    assert len(of(o, "help_answered")) == 1
    assert sum("Ignored an answer" in e["text"] for e in of(o, "agent_log")) == 2
    assert ("speed", "done") in [(e["branch"], e["status"]) for e in of(o, "branch_status")]
    assert "accepts" in next(e for e in of(o, "plan_change") if e["help"] == "hq1")["text"]


# ---- findings export ----

def test_failure_and_session_end_export_markdown_files(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"acc": "fp16"}), default="no json"), main_iters=1, branch_iters=1)
    run_with(o, [(1, "speed", "Faster."), (2, "precision", "Strict, 1e-3 on both batches.")])
    items = [i for e in of(o, "findings_exported") for i in e["items"]]
    files = [i["file"] for i in items]
    assert "exports/H1-negative-result.md" in files and "exports/summary-speed.md" in files and "exports/ledger.md" in files
    assert [i["kind"] for i in items if i["file"] == "exports/H1-negative-result.md"] == ["negative_result"]
    neg = (tmp_path / "exports" / "H1-negative-result.md").read_text()
    assert "Error versus K" in neg and "set by Precision" in neg
    ledger = (tmp_path / "exports" / "ledger.md").read_text()
    assert "| H1 | speed |" in ledger and "confirmed then rejected" in ledger
    assert all("written" not in i["title"].lower() for i in items)
    assert of(o, "findings_exported")[-1]["t"] <= of(o, "session_finished")[0]["t"]


def test_seeded_variant_is_the_first_speed_hypothesis_even_when_fp16_was_tried_on_main(tmp_path):
    fp16_main = P | {"acc": "fp16", "warps": 2}
    o = make(tmp_path, Asker(h(fp16_main, 2.0, "naive"), h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), default="no json"), main_iters=2, branch_iters=3)
    run_with(o, [(2, "speed", "Faster.")])
    first = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "speed")
    assert first["params"] == P | {"acc": "fp16"} and first["prediction"]["min_speedup"] == 1.2


def test_seeded_variant_uses_the_fastest_correct_run_when_nothing_is_confirmed(tmp_path):
    # H0 misses an over-optimistic prediction, so nothing is confirmed, and the variant goes on H0's correct tiles, not the naive ones
    o = make(tmp_path, Asker(h(P, 5.0, "naive"), h(P | {"stages": 4}, 1.05), default="no json"), main_iters=1, branch_iters=2)
    run_with(o, [(1, "speed", "Faster.")])
    on_speed = [e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "speed"]
    assert on_speed[0]["params"] == P | {"acc": "fp16"}
    parent = next(spec for spec, _ in o.engine.calls if spec.get("params") == P | {"acc": "fp16"})["refs"]["parent"]
    assert parent["params"] == P


def test_runs_request_same_window_references_including_the_parent(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05)), main_iters=2)
    o.run()
    specs = [spec for spec, _ in o.engine.calls if spec.get("refs")]
    assert set(specs[0]["refs"]) == {"naive", "eager"}
    assert specs[1]["refs"]["parent"]["params"] == P
    rf = of(o, "run_finished")[0]["run"]["metrics"]
    assert set(rf["speedups"]) == {"naive", "eager"} and rf["spreads"]["naive"] == 0.0



# ---- live controls ----

class Flaky(Asker):
    """Times out on the first fast-model call, then answers from the script."""

    def __init__(self, *script):
        super().__init__(*script)
        self.failed = False

    def __call__(self, messages, model, tag):
        if model == "fast" and not self.failed:
            self.failed = True
            raise TimeoutError("read timed out")
        return super().__call__(messages, model, tag)


def test_llm_timeout_is_visible_and_retried(tmp_path):
    o = make(tmp_path, Flaky(h(P, 2.0, "naive")), main_iters=1, retry_s=0.05)
    o.run()
    err = of(o, "agent_error")[0]
    assert err["kind"] == "llm_timeout" and err["branch"] == "main" and err["retry_in_s"] == 0
    assert of(o, "verdict")[0]["verdict"] == "confirmed"
    assert_contract(o.log.events)


def test_retry_command_wakes_a_waiting_session(tmp_path):
    o = make(tmp_path, Flaky(h(P, 2.0, "naive")), main_iters=1, retry_s=30)
    drive(o, [(lambda: bool(of(o, "agent_error")), {"type": "retry"})], timeout=5)
    assert of(o, "verdict")


def test_harness_timeout_is_a_visible_failed_run(tmp_path):
    eng = Engine()
    real = eng.one
    eng.one = lambda spec: Metrics(error="timeout", timed_out=True) if spec.get("params", {}).get("warps") == 16 else real(spec)
    o = make(tmp_path, Asker(h(P | {"warps": 16}, 2.0, "naive")), engine=eng, main_iters=1)
    o.run()
    assert of(o, "agent_error")[0]["kind"] == "harness_timeout"
    assert of(o, "run_finished")[0]["error"] == "timed out"
    assert of(o, "verdict")[0]["verdict"] == "rejected"


def test_stop_finishes_the_session(tmp_path):
    o = make(tmp_path, Flaky(h(P, 2.0, "naive")), main_iters=1, retry_s=30)
    drive(o, [(lambda: bool(of(o, "agent_error")), {"type": "stop"})], timeout=5)
    assert types(o)[-1] == "session_finished"
    assert any("Stopped by the presenter" in e["text"] for e in of(o, "agent_log"))


def test_iteration_cap_and_directive_rate_limit(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05)), main_iters=3, max_hyps=1, idle_s=30)
    o.command(SPEED)
    o.command(SPEED | {"text": "Faster again."})
    o.run()
    assert len(of(o, "hypothesis_proposed")) == 1
    assert len(of(o, "directive_added")) == 1
    assert any("Ignored a speed directive" in e["text"] for e in of(o, "agent_log"))
    assert types(o)[-1] == "session_finished"


# ---- providers ----

def test_gemini_first_then_nim_fallback_is_logged(tmp_path, monkeypatch):
    from agent import session

    class FakeClient:
        def __init__(self, provider, **kw):
            self.provider = provider

        def complete(self, messages, model, **kw):
            if self.provider == "gemini":
                raise TimeoutError("gemini slow")
            return {"text": f"from {model}"}

    calls = []
    monkeypatch.setattr(session, "Client", lambda timeout, max_retries, provider: calls.append(provider) or FakeClient(provider))
    debug = tmp_path / "debug.log"
    ask = session.asker(debug, {"fast": [("gemini", "g-fast"), ("nim", "n-fast")]})
    assert ask([], "fast", "t") == "from n-fast"
    assert ask([], "fast", "t") == "from n-fast"
    text = debug.read_text()
    assert "g-fast on gemini failed after 0.0 s (TimeoutError: gemini slow), skipped for 300 s" in text
    assert "n-fast on nim answered in 0.0 s after TimeoutError on an earlier route" in text
    assert text.count("g-fast on gemini failed") == 1  # the second call skips the failed route


def test_a_failed_route_does_not_take_down_other_models_on_the_same_provider(tmp_path, monkeypatch):
    from agent import session

    class FakeClient:
        def complete(self, messages, model, **kw):
            if model == "g-strong":
                raise TimeoutError("quota")
            return {"text": f"from {model}"}

    monkeypatch.setattr(session, "Client", lambda timeout, max_retries, provider: FakeClient())
    ask = session.asker(tmp_path / "debug.log", {"fast": [("gemini", "g-fast"), ("nim", "n-fast")], "strong": [("gemini", "g-strong")]})
    with pytest.raises(TimeoutError):
        ask([], "strong", "t")
    assert ask([], "fast", "t") == "from g-fast"


def test_provider_chain_tries_omniroute_then_gemini_when_its_key_is_set_then_nim(monkeypatch):
    from agent import session
    monkeypatch.delenv("OMNIROUTE", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert session.chains()["fast"][0] == ("omniroute", "auto/coding")
    assert [p for p, _ in session.chains()["fast"]] == ["omniroute", "gemini", "nim"]
    monkeypatch.setenv("OMNIROUTE", "0")
    monkeypatch.delenv("GEMINI_API_KEY")
    assert [p for p, _ in session.chains()["strong"]] == ["nim"]


def test_preflight_reports_each_check_and_start_command_starts_the_session(tmp_path):
    from fastapi.testclient import TestClient

    from server.app import create

    def boom():
        raise RuntimeError("no GPU here")

    checks = {"gpu": boom, "nim": lambda: (True, "200")}
    factory = lambda: make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, idle_s=5)
    with TestClient(create(factory, checks=checks, autostart=False)) as client:
        pf = client.get("/preflight").json()
        assert pf["started"] is False
        assert pf["checks"]["gpu"] == {"ok": False, "detail": "RuntimeError: no GPU here"}
        assert pf["checks"]["nim"] == {"ok": True, "detail": "200"}
        assert client.get("/health").json()["events"] == 0
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "start"}))
            assert json.loads(ws.receive_text())["type"] == "agent_log"
        assert client.get("/health").json()["started"] is True


def test_record_start_lifts_the_live_cap_and_saves_the_log_as_a_replay(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    from server.app import create

    monkeypatch.setenv("RECORD_HYPS", "3")
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), h(P | {"warps": 8}, 1.3), h(P | {"gelu": "approx"}, 1.3)),
             main_iters=4, max_hyps=1, idle_s=5, directive_wait_s=0.2)
    with TestClient(create(lambda: o, checks={}, autostart=False)) as client:
        assert client.get("/preflight").json()["cap"] == 1
        with client.websocket_connect("/ws") as ws:
            ws.send_text(json.dumps({"type": "start", "record": True}))
        deadline = time.time() + 10
        while "session_finished" not in types(o) and time.time() < deadline:
            time.sleep(0.02)
    assert len(of(o, "hypothesis_proposed")) == 3
    assert of(o, "agent_log")[2]["text"].startswith("Recording mode: the full loop for up to 3 hypotheses")
    assert types(o)[-1] == "session_finished"
    replay = [json.loads(x) for x in (tmp_path / "replay.jsonl").read_text().splitlines()]
    assert replay == o.log.events and replay[-1]["type"] == "session_finished"
    assert "/runs/sessions/test/replay.jsonl" in (tmp_path / "debug.log").read_text()
    assert not any("replay" in e["text"] for e in of(o, "agent_log"))


def test_prompt_quotes_the_measured_headroom():
    from harness import noise

    hr = noise.headroom()
    text = hyp.prompt({"id": "main", "label": "Baseline"}, [], {"params": P, "metrics": {}}, [], [], headroom=hr)[1]["content"]
    assert f"{hr['vs_naive']['median']:.2f}x faster than the naive start" in text and f"{hr['vs_eager']['median']:.2f}x vs PyTorch eager" in text


def test_stop_acts_during_a_slow_model_call(tmp_path):
    gate = threading.Event()

    def slow(messages, model, tag):
        gate.wait(10)
        return h(P, 2.0, "naive")

    o = make(tmp_path, slow, main_iters=3)
    t = threading.Thread(target=o.run)
    t.start()
    deadline = time.time() + 5
    while "session_started" not in types(o) and time.time() < deadline:
        time.sleep(0.01)
    time.sleep(0.2)
    o.command({"type": "stop"})
    t.join(2)
    gate.set()
    assert not t.is_alive()
    assert types(o)[-1] == "session_finished" and "hypothesis_proposed" not in types(o)
    assert [e["text"] for e in of(o, "agent_log")].count("Stopped by the presenter. Finishing the session with what is logged so far.") == 1


def test_restart_replaces_a_stopped_session_with_one_waiting_for_start(tmp_path):
    from fastapi.testclient import TestClient

    from server.app import create

    made = []

    def factory():
        made.append(make(tmp_path / str(len(made)), Asker(h(P, 2.0, "naive"), default="no json"), main_iters=1, idle_s=5))
        return made[-1]

    with TestClient(create(factory, checks={}, autostart=True)) as client:
        client.post("/restart")
        assert len(made) == 2 and made[0].stopped and types(made[0])[-1] == "session_finished"
        assert client.get("/health").json() | {} and client.get("/preflight").json()["started"] is False
        assert made[1].log.events == []


def test_a_baseline_that_cannot_run_is_reported_instead_of_crashing(tmp_path):
    broken = Metrics(error="exit 1: Traceback ...\nRuntimeError: Found no NVIDIA driver on your system.")
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), engine=Engine(TABLE | {"eager": broken}))
    o.run()
    assert "session_started" not in types(o)
    assert of(o, "agent_log")[-1]["text"] == ("Cannot start: the eager baseline did not run (RuntimeError: Found no NVIDIA driver on your system.). "
                                              "Check the GPU, then press New session.")
    assert of(o, "agent_log")[0]["t"] < 1000


def test_recording_waits_after_h0_then_continues_on_main_as_a_flagged_default(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.3), h(P | {"warps": 8}, 1.3), default="no json"),
             main_iters=2, directive_wait_s=0.3, context=lambda q: [{"id": "cnv_1", "title": "t", "excerpt": "e"}])
    o.set_record(3)
    o.run()
    logs = [e["text"] for e in of(o, "agent_log")]
    hyps = [e["hypothesis"] for e in of(o, "hypothesis_proposed")]
    assert len(hyps) == 3 and types(o)[-1] == "session_finished"
    waited = next(i for i, e in enumerate(o.log.events) if e["type"] == "agent_log" and e["text"].startswith("Waiting for a lens directive"))
    assert o.log.events.index(next(e for e in o.log.events if e["type"] == "verdict")) < waited
    assert any("proceeding on the default assumption" in t for t in logs)
    assert "flagged: default continuation" not in hyps[0].get("note", "")
    assert all(x["note"].endswith("(flagged: default continuation, no lens directive arrived)") and "Papiers note cnv_1 " in x["note"] for x in hyps[1:])


def test_recording_runs_the_speed_branch_after_a_directive_and_never_finishes_on_idle(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.3), default="no json"), directive_wait_s=30, idle_s=0.1)
    o.set_record(3)
    run_with(o, [(1, "speed", "Push throughput harder.")])
    hyps = [e["hypothesis"] for e in of(o, "hypothesis_proposed")]
    assert [x["branch"] for x in hyps] == ["main", "speed", "speed"]
    assert hyps[1]["params"]["acc"] == "fp16"
    assert not any("proceeding on the default assumption" in e["text"] for e in of(o, "agent_log"))


def test_shared_memory_budget_is_checked_before_any_run():
    from harness.kernel import SMEM_LIMIT, Params, smem_bytes
    assert smem_bytes(Params(128, 256, 64, 8, 4)) == 147456 and smem_bytes(Params(256, 256, 64, 8, 4)) == 196608  # the two logged failures
    assert smem_bytes(Params(128, 128, 64, 4, 4)) == 98304 <= SMEM_LIMIT
    h_, _ = hyp.parse(h(P | {"block_n": 256, "block_k": 64, "warps": 8, "stages": 4}))
    assert h_["params"].stages == 3 and h_["adjusted"].startswith("Stages cut from 4 to 3") and "147456 bytes needed, 98304 of 101376" in h_["adjusted"]
    assert hyp.parse(h(P | {"block_m": 256, "block_n": 256, "block_k": 64, "warps": 16, "stages": 5}))[0]["params"].stages == 2  # largest that fits, not just one less
    assert hyp.parse(h(P))[0]["adjusted"] is None
    assert "101376 bytes" in hyp.SYSTEM


def test_a_stage_cut_rewrites_the_claim_from_the_final_params(tmp_path):
    big = P | {"block_m": 256, "block_k": 64, "warps": 8, "stages": 4}
    fit = big | {"stages": 3}
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(big, 1.03, claim="Changing to 256x128x64 with 4 stages and 8 warps helps."), default="no json"),
             engine=Engine(TABLE | {json.dumps(fit, sort_keys=True): met(0.55)}), main_iters=2)
    o.run()
    h1 = of(o, "hypothesis_proposed")[1]["hypothesis"]
    assert h1["params"] == fit and "3 stages" in h1["claim"] and "4 stages" not in h1["claim"]
    assert h1["claim"].startswith("Block_m 128 to 256, block_k 32 to 64, warps 4 to 8, giving 256x128x64 tiles, 8 warps, 3 stages")
    assert "Stages cut from 4 to 3" in h1["note"]


def test_model_claims_spell_gelu_in_capitals():
    assert hyp.parse(h(P, claim="Switching to approximate GELu helps."))[0]["claim"] == "Switching to approximate GELU helps."


def test_out_of_resources_post_mortem_is_templated_with_the_arithmetic():
    from agent.orchestrator import failure_cause
    from harness.kernel import Params
    m = Metrics(error="OutOfResources: out of resource: shared memory, Required: 147456, Hardware limit: 101376.")
    pm = failure_cause(Params(128, 256, 64, 8, 4), m)
    assert pm["cause"].startswith("Shared memory: (stages 4 - 1) x (128x64 + 64x256) x 2 bytes = 147456 bytes, over the 101376 byte limit")
    assert pm["pivot"] == "Keep the tiles with 3 stages (98304 bytes)."


def test_summary_fields_follow_the_verdict_and_the_measured_spread(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"block_k": 64}, 1.1), default="no json"), main_iters=2)
    o.run()
    ok, failed = [e["summary"] for e in of(o, "summary")]
    assert ok["confidence"] == "high" and "clears the 2.00x threshold" in ok["confidence_reason"] and "1.6%" not in ok["confidence_reason"]
    assert failed["confidence"] == "low" and "no measurement" in failed["confidence_reason"] and "cleared" not in failed["confidence_reason"]
    assert "post-mortem ruled out" not in failed["next_step"] or "H1 post-mortem" in failed["next_step"]
    assert "Papiers note" not in ok["tried"]


def test_directive_applies_while_a_model_call_is_in_flight(tmp_path):
    gate = threading.Event()
    script = [h(P, 2.0, "naive")]

    def slow(messages, model, tag):
        if tag == "main-2-0":
            gate.wait(10)
        return script.pop(0) if script else "no json"

    o = make(tmp_path, slow, main_iters=2)
    t = threading.Thread(target=o.run)
    t.start()
    wait_for(o, lambda: len(of(o, "summary")) == 1)
    time.sleep(0.2)
    o.command({"type": "add_directive", "lens": "speed", "text": "Faster."})
    wait_for(o, lambda: bool(of(o, "directive_added")), timeout=2)  # applied while main-2-0 is still waiting on the model
    gate.set()
    t.join(5)


def test_a_duplicate_answer_is_ignored_silently_and_the_cap_reports_cut_promises(tmp_path):
    o = help_session(tmp_path, h(P | {"stages": 4}, 1.02), max_hyps=4)
    ans = {"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"}
    drive(o, [(judged(o, 1), SPEED), (asked(o), ans), (lambda: bool(of(o, "help_answered")), ans)])
    assert len(of(o, "help_answered")) == 1 and not any("Ignored an answer" in e["text"] for e in of(o, "agent_log"))
    assert "duplicate answer to hq1" in (tmp_path / "debug.log").read_text()
    cut = [e for e in of(o, "plan_change") if "cap ended the session" in e["text"]]
    assert cut and cut[0]["help"] == "hq1"
    ev = o.log.events
    last_status = {e["branch"]: e["status"] for e in ev if e["type"] == "branch_status"}
    assert all(s not in ("running", "waiting_on_human") for s in last_status.values()) and set(last_status) == set(o.branches)
    assert types(o)[-1] == "session_finished"


# ---- hybrid guard and register post-mortem ----

def lens_state(tmp_path):
    """The 11:00 recording: H0 on main, H4 on prec (256x128) and H5 on speed2 (128x256, approx GELU), after a conflict."""
    from agent.orchestrator import Branch, Hyp
    from agent.verdict import Judgment
    from harness.kernel import Params
    o = make(tmp_path, Asker())
    o.naive = met(2.0)
    for b in (Branch("main", [], "Baseline", "both"), Branch("prec", ["main"], "Precision lens", "precision"),
              Branch("speed2", ["main"], "Speed lens", "speed")):
        o.branches[b.id] = b
    base = Params(128, 128, 64, 4, 4)
    for i, (bid, p, lat) in enumerate([("main", base, 0.78), ("prec", Params(256, 128, 64, 8, 3), 0.70),
                                        ("speed2", Params(128, 256, 64, 8, 3, gelu="approx"), 0.697)]):
        x = Hyp(f"H{i}", bid, "c", {"min_speedup": 1.02, "vs": "parent_best", "max_error": 1e-3}, p, metrics=met(lat))
        x.judgment = Judgment("confirmed", False, "Prediction holds.")
        o.hyps.append(x)
    o.directives = [{"id": "d1", "lens": "precision", "threshold": 1e-3}]
    o.conflicts = 1
    return o


def test_a_run_far_slower_than_its_parent_gets_a_register_post_mortem(tmp_path):
    big = P | {"block_n": 256, "warps": 8}
    slow = Metrics.from_dict(met(6.4, err=6.9e-4, adv=7.2e-4).dict() | {"n_regs": 40, "n_spills": 1492})
    key = json.dumps(P, sort_keys=True)
    table = TABLE | {json.dumps(big, sort_keys=True): slow, key: Metrics.from_dict(TABLE[key].dict() | {"n_regs": 255, "n_spills": 24})}

    class Window(Engine):
        def __call__(self, spec, run_id):
            m = super().__call__(spec, run_id)
            if spec.get("refs") and m.latency_ms:
                m.ref_ms = {k: self.one(r).latency_ms for k, r in spec["refs"].items()}
            return m

    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(big, 1.05), default="no json"), engine=Window(table), main_iters=2)
    o.run()
    pm = next(e["post_mortem"] for e in of(o, "post_mortem") if e["post_mortem"]["hypothesis"] == "H1")
    assert "40 registers per thread" in pm["cause"] and "1492 32-bit words" in pm["cause"] and "255 registers and 24 spilled words for the parent" in pm["cause"]
    assert "128 elements per thread, within the 128 limit" in pm["cause"] and "6.400 ms" in pm["observed"] and pm["sweep"] == []
    assert any(i["kind"] == "negative_result" and i["file"] == "exports/H1-negative-result.md" for e in of(o, "findings_exported") for i in e["items"])
    assert_contract(o.log.events)


# ---- hunch attribution ----

def test_hunch_change_tells_constraints_from_concrete_changes():
    assert hyp.hunch_change("Push throughput harder than tile tuning alone. I can accept max error up to 2e-2.") is None
    assert hyp.hunch_change("Accuracy matters more than speed here. Keep max error under 1e-3 on every batch.") is None
    assert hyp.hunch_change("Try the tanh approximate GELU.") == {"gelu": "approx"}
    assert hyp.hunch_change("Use FP16 accumulation") == {"acc": "fp16"}
    assert hyp.hunch_change("try 128x256x64 with 8 warps") == {"block_m": 128, "block_n": 256, "block_k": 64, "warps": 8}
    assert hyp.hunch_change("bigger tiles please") == {}


def test_constraint_directives_are_bounds_and_a_repeated_hunch_is_retested_on_the_best_tiles(tmp_path, monkeypatch):
    from agent import orchestrator
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    w8 = P | {"warps": 8}
    engine = Engine(TABLE | {json.dumps(w8, sort_keys=True): met(0.55), json.dumps(w8 | {"gelu": "approx"}, sort_keys=True): met(0.54, err=6.9e-4, adv=7.2e-4)})
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"gelu": "approx"}, 1.02), h(w8, 1.05), default="no json"),
             engine=engine, main_iters=1, branch_iters=3, help_deadline_s=0.3)
    run_with(o, [(1, "speed", "Push throughput harder than tile tuning alone."), (4, "speed", "Try the tanh approximate GELU.")])
    hyps = {e["hypothesis"]["id"]: e["hypothesis"] for e in of(o, "hypothesis_proposed")}
    assert hyps["H2"]["note"] == "Tested under the Speed lens bound set by d1"
    assert not any("Hunch" in (x.get("note") or "") for x in hyps.values() if x["id"] != "H4")
    texts = [e["text"] for e in of(o, "plan_change")]
    assert any(t.startswith("d1 sets a bound, not a specific change") for t in texts)
    assert any("already tested in H2 (rejected, 1.00x vs its parent) on 128x128x32 tiles" in t for t in texts)
    d2 = [e["text"] for e in of(o, "plan_change") if e["help"] == "d2"]
    assert len(d2) == 1 and "This is a retest" in d2[0] and d2[0].endswith("so the run changes only gelu exact to approx on the current best 128x128x32 tiles, 8 warps, 3 stages.")
    assert o.hunches["d2"] == "H4"
    h4 = hyps["H4"]
    assert h4["params"] == w8 | {"gelu": "approx"} and h4["note"] == "Hunch from the Speed lens in directive d2, a retest of H2 on the current best tiles"
    assert h4["claim"].startswith("Retest of H2: gelu exact to approx on the current best 128x128x32 tiles, 8 warps, 3 stages is at least 1.02x")
    assert not any(tag.startswith(f"{h4['branch']}-1") for _, tag, _ in o.ask.calls)  # built in code, no model call
    assert_contract(o.log.events)


# ---- help requests after a verdict flip ----

def test_a_flip_moves_a_pending_help_request_to_the_successor_and_keep_tuning_never_revives_the_rejected_branch(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 1.02), default="no json"), main_iters=1, branch_iters=2, help_deadline_s=30)
    asked = lambda: bool(of(o, "help_requested"))
    reissued = lambda: len(of(o, "help_requested")) > 1
    drive(o, [(judged(o, 1), SPEED), (asked, {"type": "add_directive", "lens": "precision", "text": "Max error under 1e-3 on both batches."}),
              (reissued, {"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})])
    first, again = of(o, "help_requested")[:2]
    assert first["help"]["blocks"] == "speed" and any(e.startswith("r2:") and "provisionally confirmed" in e for e in first["help"]["evidence"])
    assert again["help"]["id"] == "hq1" and again["help"]["blocks"] == "speed2" and "applies to speed2" in again["help"]["question"]
    assert any(e.startswith("r2:") and "rejected" in e for e in again["help"]["evidence"])
    assert not any("confirmed" in e for e in again["help"]["evidence"] if e.startswith("r2:"))
    statuses = [e["status"] for e in of(o, "branch_status") if e["branch"] == "speed"]
    assert statuses[statuses.index("rejected"):].count("running") == 0
    plan = next(e["text"] for e in of(o, "plan_change") if e["help"] == "hq1")
    assert "speed2" in plan and o.extra.get("speed2") == 2 and "speed" not in o.extra
    assert_contract(o.log.events)


def test_a_flip_withdraws_a_help_request_when_nothing_continues_the_branch(tmp_path):
    from agent import orchestrator
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 1.02), default="no json"), main_iters=1, branch_iters=2, help_deadline_s=30)
    o.pivot = lambda h: None  # the per-lens branch cap leaves no successor
    asked = lambda: bool(of(o, "help_requested"))
    cancelled = lambda: bool(of(o, "help_cancelled"))
    drive(o, [(judged(o, 1), SPEED), (asked, {"type": "add_directive", "lens": "precision", "text": "Max error under 1e-3 on both batches."}),
              (cancelled, {"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})])
    assert of(o, "help_cancelled")[0]["help"] == "hq1" and "no branch continues it" in of(o, "help_cancelled")[0]["reason"]
    assert not of(o, "help_answered") and any("withdrawn" in e["text"] for e in of(o, "agent_log"))
    assert orchestrator.open_help(o.helps["hq1"]) is False
    assert_contract(o.log.events)


def test_the_waiting_line_is_logged_once_and_explanation_fallbacks_stay_out_of_the_replay(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 1.02), default="no json"), main_iters=1, branch_iters=2,
             help_deadline_s=0.3, directive_wait_s=30)
    o.set_record(4)
    o.main_iters = 1
    drive(o, [(judged(o, 1), SPEED), (judged(o, 3), {"type": "stop"})])
    logs = [e["text"] for e in of(o, "agent_log")]
    assert sum(t.startswith("Waiting for a lens directive") for t in logs) == 1
    assert not any(t.startswith("Explanation model unavailable") for t in logs)



# ---- Papiers retrieval through the CLI, mocked ----

HL_SEARCH = {"results": [{"kind": "document", "id": "doc_mp", "title": "MIXED PRECISION TRAINING", "evidence": [
    {"kind": "highlight", "id": "hl_mp1", "snippet": "FP16 vector dot-product accumulates the partial products into an FP32 value"}]}]}
HL_READ = {"items": [{"id": "hl_mp1", "kind": "highlight", "ok": True, "content": "Some networks require that FP16 vector dot-product accumulates into FP32.",
                      "metadata": {"documentTitle": "MIXED PRECISION TRAINING", "annotationText": "This is why our FP16 accumulator fails at large K."}}]}
NOTE_SEARCH = {"results": [{"kind": "note", "id": "nte_1", "title": "Tiling plan", "snippet": "Keep the accumulator under 128 per thread."}]}
EMPTY = {"results": [], "quality": {"signal": "empty"}}


class CLI:
    """Answers papiers argument lists from a table keyed on (command, kind), records every call."""

    def __init__(self, table, fail=None):
        self.table, self.fail, self.calls = table, fail, []

    def __call__(self, args):
        self.calls.append(args)
        if self.fail:
            raise self.fail
        kind = args[args.index("--kind") + 1] if "--kind" in args else None
        key = (args[0], kind, "annotated" if any("annotated" in a for a in args) else None)
        return self.table.get(key) or self.table.get((args[0], kind, None)) or self.table.get((args[0], None, None)) or {"results": [], "items": []}


def test_retrieval_puts_annotated_highlights_first_with_the_document_title_and_the_users_words():
    from agent.papiers import Library
    cli = CLI({("search", "highlight", None): HL_SEARCH, ("search", "note", None): NOTE_SEARCH, ("read", None, None): HL_READ})
    got = Library(cli).retrieve("accumulation precision FP16")
    assert got[0] == {"id": "hl_mp1", "kind": "highlight", "document_title": "MIXED PRECISION TRAINING", "title": "MIXED PRECISION TRAINING",
                      "excerpt": "Some networks require that FP16 vector dot-product accumulates into FP32.",
                      "highlight_text": "Some networks require that FP16 vector dot-product accumulates into FP32.",
                      "annotation": "This is why our FP16 accumulator fails at large K."}
    assert got[1]["kind"] == "note" and got[1]["id"] == "nte_1" and "annotation" not in got[1]
    assert not any(c[0] == "search" and "passage" in c for c in cli.calls)  # judgement found, no fallback


def test_a_note_outranks_an_unannotated_highlight():
    from agent.papiers import Library
    cli = CLI({("search", "highlight", None): HL_SEARCH, ("search", "note", None): NOTE_SEARCH,
               ("read", None, None): {"items": [HL_READ["items"][0] | {"metadata": {"annotationText": None}}]}})
    assert [x["id"] for x in Library(cli).retrieve("accumulation")] == ["nte_1", "hl_mp1"]


def test_retrieval_falls_back_to_listed_marks_then_annotated_passages_then_passages_then_conversations():
    from agent.papiers import Library
    listed = {"items": [{"id": "hl_x", "kind": "highlight", "preview": "accumulation length breakdown region", "metadata": {"documentTitle": "Bit-width"}}]}
    cli = CLI({("list", "highlight", None): listed, ("read", None, None): {"items": []}})
    assert [x["id"] for x in Library(cli).retrieve("accumulation error grows with length")] == ["hl_x"]
    passage = {"results": [{"kind": "document", "id": "doc_t", "title": "Triton", "evidence": [{"kind": "passage", "id": "pas_1", "snippet": "tiles"}]}]}
    cli = CLI({("search", "passage", None): passage})
    got = Library(cli).retrieve("tile size shared memory")
    assert got == [{"id": "pas_1", "kind": "passage", "document_title": "Triton", "title": "Triton", "excerpt": "tiles"}]
    assert ["search", "tile size shared memory", "--kind", "passage", "--filter", "annotated.eq=true", "--limit", "2"] in cli.calls
    convo = {"results": [{"kind": "conversation", "id": "cnv_9", "title": "Chat", "snippet": "about tiles"}]}
    assert Library(CLI({("search", "conversation", None): convo})).retrieve("tiles")[0]["kind"] == "conversation"


def test_an_empty_library_returns_nothing_and_the_hypothesis_cites_nothing(tmp_path):
    from agent.papiers import Library
    assert Library(CLI({})).retrieve("tile size shared memory") == []
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, context=Library(CLI({})).retrieve)
    o.run()
    assert [(e["query"], "note" in e) for e in of(o, "papiers_context")] == [("tile size shared memory", False)]
    assert of(o, "hypothesis_proposed")[0]["hypothesis"].get("note") is None


def test_a_cli_error_is_logged_and_the_loop_continues(tmp_path):
    import subprocess

    from agent.papiers import Library
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, context=Library(CLI({}, fail=subprocess.CalledProcessError(1, "papiers"))).retrieve)
    o.run()
    assert any("Papiers search unavailable (CalledProcessError)" in e["text"] for e in of(o, "agent_log"))
    assert of(o, "verdict")[0]["verdict"] == "confirmed"


def test_the_fp16_hypothesis_and_its_post_mortem_cite_what_was_retrieved(tmp_path):
    from agent.papiers import Library
    cli = CLI({("search", "highlight", None): HL_SEARCH, ("read", None, None): HL_READ})
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), default="no json"), main_iters=1, branch_iters=1, context=Library(cli).retrieve)
    run_with(o, [(1, "speed", "Push throughput harder."), (2, "precision", "Max error under 1e-3 on both batches.")])
    seed = of(o, "hypothesis_proposed")[1]["hypothesis"]
    assert seed["params"]["acc"] == "fp16" and seed["note"].endswith("Context from Papiers highlight hl_mp1")
    reads = [e for e in of(o, "papiers_context") if e.get("note")]
    assert {e["query"] for e in reads} >= {"accumulation precision FP16", "accumulation error grows with length"}
    assert reads[0]["note"]["annotation"] == "This is why our FP16 accumulator fails at large K."
    pm = of(o, "post_mortem")[0]["post_mortem"]
    assert pm["hypothesis"] == seed["id"] and pm["note"] == "hl_mp1"
    queries = {c[1] for c in cli.calls if c[0] == "search"}
    assert "Push throughput harder." not in queries  # queries name the change or the cause, not the directive text
    assert_contract(o.log.events)


def test_counts_for_the_preflight():
    from agent.papiers import Library
    cli = CLI({("list", "highlight", None): {"items": [{"id": "hl_1"}] * 50, "page": {"nextCursor": "c"}}, ("list", "document", None): {"items": [{"id": "doc_1"}] * 6}})
    assert Library(cli).counts() == {"highlight": "50+", "note": 0, "document": 6}


def test_preflight_reports_papiers_counts_and_warns_without_highlights(monkeypatch):
    from agent import papiers
    from server import app
    monkeypatch.setattr(papiers, "binary", lambda: "/bin/papiers")
    monkeypatch.setattr(papiers.Library, "counts", lambda self: {"highlight": 0, "note": 2, "document": 6})
    r = app.run_check(app.check_papiers)
    assert r == {"ok": True, "warn": True, "detail": "the CLI sees 0 highlights, 2 notes and 6 documents. No highlights, so hypotheses fall back to passages and conversations"}
    monkeypatch.setattr(papiers.Library, "counts", lambda self: {"highlight": 5, "note": 0, "document": 6})
    assert app.run_check(app.check_papiers) == {"ok": True, "detail": "the CLI sees 5 highlights, 0 notes and 6 documents"}



def test_a_tiling_hypothesis_never_cites_a_note_retrieved_for_another_change(tmp_path):
    def context(q):
        return [{"id": "hl_acc1", "kind": "highlight", "title": "Mixed precision", "excerpt": "FP32 accumulation."}] if "accumulation" in q else []
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.02), default="no json"), main_iters=1, branch_iters=2, context=context, help_deadline_s=0.3)
    run_with(o, [(1, "precision", "Max error under 1e-3 on both batches.")])
    tiling = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["params"]["stages"] == 4)
    assert "hl_acc1" not in (tiling.get("note") or "")
    assert any(e.get("note", {}).get("id") == "hl_acc1" for e in of(o, "papiers_context"))  # read for the prompt, not cited


# ---- fixes from the 16:00 recording ----

def test_a_config_over_the_register_limit_is_refused_before_any_run():
    h_, why = hyp.parse(h(P | {"block_n": 256, "block_k": 64, "warps": 4}))
    assert h_ is None and "256 fp32 accumulator elements" in why and "over the 128 register limit" in why
    assert hyp.parse(h(P | {"block_n": 256, "block_k": 64, "warps": 8}))[0]
    assert "Registers:" in hyp.SYSTEM


def recorded_state(tmp_path, **kw):
    """The 16:00 recording after the conflict: H0 on main, H1 on prec (8 warps) and H2 on speed2 (128x256, approx GELU)."""
    from harness.kernel import Params
    o = lens_state(tmp_path)
    o.hyps[1].params, o.hyps[1].metrics = Params(128, 128, 64, 8, 4), met(0.728)
    o.hyps[2].metrics = met(0.698, err=6.9e-4, adv=7.2e-4)
    for k, v in kw.items():
        setattr(o, k, v)
    return o


def test_a_summary_is_measured_against_the_base_the_run_was_timed_against(tmp_path):
    from agent.orchestrator import Hyp
    o = recorded_state(tmp_path)
    o.hyps += [Hyp("H3", "speed2", "c", {}, o.hyps[0].params, base=o.hyps[1].params), Hyp("H4", "speed2", "c", {}, o.hyps[0].params)]
    assert o.baseline(o.hyps[3])[1] is o.hyps[1].metrics
    assert o.baseline(o.hyps[4])[1] is o.hyps[2].metrics  # the branch best, without a base


def test_a_speed_directive_after_a_precision_one_says_the_precision_bound_still_applies(tmp_path):
    o = recorded_state(tmp_path)
    o.add_directive("speed", "Faster, I accept 2e-2.")
    assert any("Precision bound of 1.0e-3 on the standard and adversarial batches still applies" in e["text"] for e in of(o, "agent_log"))
    assert not any(e["text"].startswith("Speed lens bound assigned") for e in of(o, "agent_log"))


def test_at_the_cap_a_bound_opens_no_branch_and_a_hunch_runs_in_a_reserved_slot(tmp_path, monkeypatch):
    from agent import orchestrator
    from harness.kernel import Params
    monkeypatch.setattr(orchestrator, "DIRECTIVE_GAP_S", 0)
    engine = Engine(TABLE | {json.dumps(Params(128, 256, 64, 8, 3, gelu="approx").dict(), sort_keys=True): met(0.698, err=6.9e-4, adv=7.2e-4)})
    o = recorded_state(tmp_path, max_hyps=3, engine=engine, ask=Asker(h(P, 1.05)))
    o.branches["speed2"].status = "done"
    o.add_directive("speed", "Faster.")
    assert not of(o, "branch_created") and "because the session reached its 3 hypothesis cap" in of(o, "plan_change")[-1]["text"]
    o.add_directive("speed", "Try 128x128x32 tiles with 3 stages.")
    b = o.next_branch()
    assert b.id == of(o, "branch_created")[0]["branch"]["id"] and not o.capped()
    o.iterate(b)
    assert o.hyps[-1].params.dict() == P and o.hyps[-1].bonus and o.hyps[-1].note.startswith("Hunch from the Speed lens in directive d")
    assert o.capped() and o.next_branch() is None


def test_keep_tuning_guarantees_its_iterations_past_two_misses(tmp_path):
    from harness.kernel import Params
    slow = P | {"stages": 2}
    engine = Engine(TABLE | {json.dumps(Params(128, 256, 64, 8, 3, gelu="approx").dict(), sort_keys=True): met(0.698, err=6.9e-4, adv=7.2e-4),
                             json.dumps(slow, sort_keys=True): met(0.9)})
    o = recorded_state(tmp_path, engine=engine, ask=Asker(h(slow, 1.05)))
    b = o.branches["speed2"]
    b.misses = 1
    o.promise("hq1", b, 2)
    o.iterate(b)
    assert o.hyps[-1].judgment.verdict == "rejected" and b.misses == 2
    assert b.status == "running" and o.promised("speed2") == 1
    assert of(o, "summary")[-1]["summary"]["next_step"] == f"Try a different change on this branch, avoiding what the {o.hyps[-1].id} post-mortem ruled out."


# ---- fixes from the 17:07 recording ----

def converged_state(tmp_path):
    """The 17:07 recording: H0 64x64x32 on main, approximate GELU as the best correct run of each lens (H1 on prec, H2 on speed2)."""
    from agent.verdict import Judgment
    from harness.kernel import Params
    o = lens_state(tmp_path)
    base = Params(64, 64, 32, 4, 3)
    for x, p, lat in zip(o.hyps, (base, replace_gelu(base), Params(128, 128, 32, 8, 4, gelu="approx")), (0.720, 0.706, 0.732)):
        x.params, x.metrics = p, met(lat, err=6.9e-4 if p.gelu == "approx" else 4.9e-4, adv=7.2e-4)
    o.hyps[1].judgment = Judgment("inconclusive", False, "noise")
    return o


def replace_gelu(p):
    from dataclasses import replace
    return replace(p, gelu="approx")


def test_a_retest_refusal_says_which_run_already_had_the_config(tmp_path):
    o = converged_state(tmp_path)
    o.hunch(o.branches["prec"], "Try the tanh approximate GELU.", "Hunch from the Speed lens in directive d3", "d3", "speed", {"gelu": "approx"})
    assert [e["text"] for e in of(o, "plan_change")] == [
        ("Because the Speed lens sent \"Try the tanh approximate GELU.\", I am not testing it because that exact config, 64x64x32 tiles, 4 warps, 3 stages, "
        "FP32 accumulation, approx GELU, already ran as H1 on prec (inconclusive). This is a retest: it was already tested in H1 (inconclusive) on 64x64x32 tiles.")]
    assert "prec" not in o.hints and o.hunches["d3"].startswith("not run: that exact config")


# ---- fixes from the latest full recording ----

def test_a_rejudge_during_a_run_keeps_the_branch_rejected_and_its_help_applies_to_the_successor(tmp_path):
    prec = {"type": "add_directive", "lens": "precision", "text": "Max error under 1e-3 on both batches."}

    class Racing(Engine):
        """The Precision directive lands while the inconclusive H2 is on the GPU, as the listener thread would apply it."""
        def __call__(self, spec, run_id):
            if spec.get("params") == INCONCLUSIVE and "/H2-" in run_id:
                def listener():
                    with o.lock:
                        o.handle(prec)
                threading.Thread(target=listener, daemon=True).start()
                wait_for(o, lambda: "speed2" in o.branches)
            return super().__call__(spec, run_id)

    o = help_session(tmp_path, engine=Racing())
    drive(o, [(judged(o, 1), SPEED), (asked(o), {"type": "answer_help", "help": "hq1", "option": "tile_search", "by": "speed"})])
    req = of(o, "help_requested")
    assert len(req) == 1 and req[0]["help"]["blocks"] == "speed2" and "applies to speed2" in req[0]["help"]["question"]
    statuses = [e["status"] for e in of(o, "branch_status") if e["branch"] == "speed"]
    assert statuses.count("rejected") == 1 and statuses[statuses.index("rejected"):] == ["rejected"]
    assert o.branches["speed"].status == "rejected" and o.extra.get("speed2") == 2 and "speed" not in o.extra
    assert "speed2" in next(e["text"] for e in of(o, "plan_change") if e["help"] == "hq1")
    assert_contract(o.log.events)


def test_a_rejudge_that_flips_several_runs_reissues_each_help_request_once(tmp_path):
    from agent.orchestrator import Branch, Hyp
    from harness.kernel import Params
    o = make(tmp_path, Asker())
    o.naive = met(2.0)
    for b in (Branch("main", [], "Baseline", "both"), Branch("speed", ["main"], "Speed lens", "speed", status="waiting_on_human")):
        o.branches[b.id] = b
    for i, p in enumerate((Params(128, 128, 32, 4, 3, acc="fp16"), Params(128, 128, 32, 8, 3, acc="fp16"))):
        x = Hyp(f"H{i + 1}", "speed", "c", {"min_speedup": 1.02, "vs": "parent_best", "max_error": 2e-2}, p, run=f"r{i + 1}",
                metrics=met(0.45, err=1.3e-2, adv=7.1e-2), speedup=1.4, bounds=[{"standard": 2e-2, "adversarial": None, "set_by": "speed"}])
        x.judgment = judge(x.prediction, x.metrics, 1.4, BAND, 2e-2, None)
        o.hyps.append(x)
    o.directives = [{"id": "d1", "lens": "speed", "threshold": 2e-2}]
    o.request_help(o.hyps[1], o.branches["speed"], "speed")
    o.add_directive("precision", "Max error under 1e-3 on both batches.")
    assert [e["help"]["id"] for e in of(o, "help_requested")] == ["hq1", "hq1"]
    assert of(o, "help_requested")[1]["help"]["blocks"] == "speed2"


def ledger_state(tmp_path, **kw):
    """H0 on main, H1 (stages 4) slower than H0 on speed, H2 (warps 8 with stages 4) rejected on speed."""
    from agent.orchestrator import Branch, Hyp
    from harness.kernel import Params
    o = make(tmp_path, Asker(), **kw)
    o.naive = met(1.891)
    for b in (Branch("main", [], "Baseline", "both"), Branch("speed", ["main"], "Speed lens", "speed")):
        o.branches[b.id] = b
    base = Params(**P)
    rows = [("main", base, None, 0.642, 2.95, "naive", 200, 0), ("speed", Params(**P | {"stages": 4}), base, 0.70, 0.92, "parent", 210, 0),
            ("speed", Params(**P | {"stages": 4, "warps": 8}), base, 0.66, 0.97, "parent", 180, 12)]
    for i, (bid, p, parent, lat, sp, ref, regs, spills) in enumerate(rows):
        m = Metrics.from_dict(met(lat).dict() | {"n_regs": regs, "n_spills": spills})
        x = Hyp(f"H{i}", bid, "c", {"min_speedup": 1.02 if parent else 2.0, "vs": "parent_best" if parent else "naive", "max_error": 1e-3},
                p, run=f"r{i + 1}", metrics=m, speedup=sp, noise=BAND, base=parent, ref=ref)
        x.judgment = judge(x.prediction, m, sp, BAND, 1e-3, None)
        o.hyps.append(x)
    return o


def test_the_proposer_sees_the_whole_ledger_with_registers_and_spills(tmp_path):
    o = ledger_state(tmp_path)
    rows = o.ledger()
    assert [r["id"] for r in rows] == ["H0", "H1", "H2"] and rows[1]["change"] == "stages 3 to 4"
    assert rows[2] | {"params": None} == {"id": "H2", "branch": "speed", "params": None, "change": "warps 4 to 8, stages 3 to 4",
                                          "outcome": "rejected, 0.97x vs its parent", "n_regs": 180, "n_spills": 12}
    o.branches["speed2"] = __import__("agent.orchestrator", fromlist=["Branch"]).Branch("speed2", ["main"], "Speed lens", "speed")
    o.ask = Asker(h(P | {"block_k": 64}, 1.05))
    o.propose(o.branches["speed2"])
    content = o.ask.calls[0][2][1]["content"]
    assert '"id": "H2", "branch": "speed"' in content and '"n_spills": 12' in content and "on any branch" in content


def test_duplicates_and_known_harmful_knobs_are_refused_with_a_logged_reason(tmp_path):
    from harness.kernel import Params
    o = ledger_state(tmp_path)
    start = o.hyps[0].params
    assert o.vet(start, Params(**P | {"stages": 4}), "x").startswith("128x128x32 tiles, 4 warps, 4 stages, FP32 accumulation, exact GELU already ran as H1")
    assert o.vet(start, Params(**P | {"block_k": 64}), "x") is None  # block_k 64 was never shown to hurt
    assert o.vet(Params(**P | {"warps": 8}), Params(**P | {"warps": 8, "stages": 4, "gelu": "approx"}), "H1 aside") is None
    why = o.vet(start, Params(**P | {"stages": 4, "block_k": 64}), "Deeper pipelining hides latency.")
    assert why == "the claim sets stages 4, which H1 showed hurts (rejected, 0.92x vs its parent), without citing H1; cite it and say why this case differs"
    assert o.vet(start, Params(**P | {"stages": 4, "block_k": 64}), "Unlike H1, block_k 64 halves the loop count, so 4 stages pay off.") is None
    o.ask = Asker(h(P | {"stages": 4, "block_k": 64}, 1.05), h(P | {"block_k": 64}, 1.05))
    o.propose(o.branches["speed"])
    assert any(e["text"].startswith("Refused a proposal on speed: the claim sets stages 4, which H1") for e in of(o, "agent_log"))
    assert "Invalid: the claim sets stages 4" in o.ask.calls[1][2][-1]["content"]


def test_a_rejected_config_with_one_more_harmful_knob_is_refused(tmp_path):
    from agent.verdict import judge as j
    from harness.kernel import Params
    o = ledger_state(tmp_path)
    x = o.hyps[2]
    x.params = Params(**P | {"warps": 8})
    x.judgment = j(x.prediction, x.metrics, 0.97, BAND, 1e-3, None)
    why = o.vet(Params(**P | {"block_k": 64}), Params(**P | {"warps": 8, "stages": 4}), "Cites H1 and H2.")
    assert why == "it is the rejected H1 with only warps 4 to 8, a change H2 already showed hurts (rejected, 0.97x vs its parent)"


def test_every_rejection_gets_a_post_mortem_with_registers_spills_and_the_param_diff(tmp_path):
    from agent.orchestrator import miss_cause
    from harness.kernel import Params
    base, parent = Params(**P), Metrics.from_dict(met(0.642).dict() | {"n_regs": 168, "n_spills": 0})
    spill = miss_cause(Params(**P | {"block_n": 256, "warps": 8}), base, Metrics.from_dict(met(0.9).dict() | {"n_regs": 255, "n_spills": 40}),
                       parent, 0.71, {"min_speedup": 1.1}, "parent")
    assert spill["cause"] == ("It ran at 0.71x vs its parent, slower than what it was timed against. Changed block_n 128 to 256, warps 4 to 8 "
                              "(168 to 255 registers and 0 to 40 spilled words per thread). Likely cause: it spills 40 words per thread to local memory, "
                              "so the inner loop waits on memory.")
    regs = miss_cause(Params(**P | {"block_k": 64}), base, Metrics.from_dict(met(0.7).dict() | {"n_regs": 232, "n_spills": 0}), parent, 0.92,
                      {"min_speedup": 1.1}, "parent")
    assert "registers per thread rose from 168 to 232, so fewer warps fit on each SM" in regs["cause"] and regs["pivot"].startswith("Revert block_k 32 to 64")
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 1.02), default="no json"), main_iters=1, branch_iters=2, help_deadline_s=0.5)
    drive(o, [(judged(o, 1), SPEED), (asked(o), {"type": "add_directive", "lens": "precision", "text": "Max error under 1e-3 on both batches."})])
    rejected = {e["hypothesis"] for e in of(o, "verdict") if e["verdict"] == "rejected"}
    assert rejected and rejected <= {e["post_mortem"]["hypothesis"] for e in of(o, "post_mortem")}


def test_claim_numbers_must_match_the_final_params():
    from harness.kernel import Params
    p = Params(128, 128, 32, 4, 3)
    assert hyp.mismatches("Using 128x128x32 tiles with 4 warps and 3 stages, block_k = 32, is faster.", p) == []
    assert hyp.mismatches("A 64x64 tile with 8 warps and num_stages=4 and BLOCK_M=64 helps.", p) == ["64x64", "8 warps", "num_stages=4", "block_m=64"]
    assert hyp.mismatches("At least 1.3x faster than 2 runs ago.", p) == []


def test_a_claim_that_names_other_numbers_is_rebuilt_from_the_params(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 1.05, claim="Going to 8 warps with 4 stages beats H0."), default="no json"), main_iters=2)
    o.run()
    h1 = of(o, "hypothesis_proposed")[1]["hypothesis"]
    assert h1["claim"].startswith("Warps 4 to 8, giving 128x128x32 tiles, 8 warps, 3 stages") and h1["claim"].endswith(" Revisits H0.")
    assert "4 stages" not in h1["claim"] and "the model's text named 4 stages" in h1["note"]
    assert hyp.mismatches(h1["claim"].split(", giving ")[1], o.hyps[1].params) == []


def test_a_hunch_gets_one_plan_change_at_directive_time_even_when_the_seed_runs_first(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"gelu": "approx"}, 1.02), default="no json"), main_iters=1, branch_iters=2, help_deadline_s=0.3)
    run_with(o, [(1, "speed", "Try the tanh approximate GELU.")])
    texts = [e["text"] for e in of(o, "plan_change") if e["help"] == "d1"]
    assert texts == [("Because the Speed lens sent \"Try the tanh approximate GELU.\", I am testing it after the scheduled FP16 variant on speed, judged against "
                     "max error under 2.0e-2, the Speed lens bound. This is not a retest, nothing in the session has tested it yet, so the agent turns it "
                     "into the next hypothesis on that branch.")]
    hyps = [e["hypothesis"] for e in of(o, "hypothesis_proposed")]
    assert hyps[1]["params"]["acc"] == "fp16" and hyps[2]["note"].startswith("Hunch from the Speed lens in directive d1") and o.hunches["d1"] == "H2"


def test_the_cap_ends_the_session_at_once_with_pending_help_reported_unanswered(tmp_path):
    # main keeps running past the help request on speed (H3) and reaches the cap with H4
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.02), h(INCONCLUSIVE, 1.02), h(P | {"gelu": "approx"}, 1.02), default="no json"),
             main_iters=3, branch_iters=2, help_deadline_s=30, max_hyps=5)
    start = time.time()
    drive(o, [(judged(o, 1), SPEED)])
    assert time.time() - start < 5 and of(o, "help_requested")
    texts = [e["text"] for e in of(o, "plan_change")]
    assert any(t.startswith("Session ended with hq1 unanswered: the 5 hypothesis cap was reached, so ") for t in texts)
    assert not any("proceeding on the default" in t or "cap ended the session" in t for t in texts)
    assert types(o)[-1] == "session_finished"


def test_no_help_is_requested_once_the_cap_is_reached(tmp_path):
    o = help_session(tmp_path, max_hyps=3)
    drive(o, [(judged(o, 1), SPEED)])
    assert o.hyps[-1].judgment.verdict == "inconclusive" and not of(o, "help_requested")


def test_the_final_report_comes_before_session_finished_with_one_line_per_hunch(tmp_path):
    from agent import orchestrator
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"gelu": "approx"}, 1.02), h(P | {"stages": 4}, 1.02), default="no json"),
             main_iters=1, branch_iters=2, help_deadline_s=0.3)
    orchestrator.DIRECTIVE_GAP_S, gap = 0, orchestrator.DIRECTIVE_GAP_S
    try:
        run_with(o, [(1, "speed", "Try the tanh approximate GELU."), (3, "precision", "Max error under 1e-3 on both batches.")])
    finally:
        orchestrator.DIRECTIVE_GAP_S = gap
    assert types(o)[-2:] == ["final_report", "session_finished"]
    r = of(o, "final_report")[0]["report"]
    assert r["best"]["run"] == of(o, "session_finished")[0]["best"]["run"] and r["best"]["hypothesis"] == "H0"
    assert any(x.startswith("H1 on speed, acc fp32 to fp16: rejected") for x in r["ruled_out"])
    assert any("Decide whether that error is acceptable" in x and x.startswith("H1 ran 1.43x faster") for x in r["human"])
    assert r["hunches"] == [{"directive": "d1", "lens": "speed", "text": "Try the tanh approximate GELU.",
                             "outcome": "H2 on speed ran 128x128x32 tiles, 4 warps, 3 stages, FP32 accumulation, approx GELU: " + o.outcome(o.hyps[2])}]
    assert r["next"]
    assert_contract(o.log.events)


def test_annotations_lose_latex_dollars_and_an_annotated_highlight_wins_wherever_it_ranks():
    from agent.papiers import Library, untex
    assert untex("Error grows with $K$, roughly $$\\sqrt{K}$$ here.") == "Error grows with K, roughly \\sqrt{K} here."
    search = {"results": [{"kind": "document", "id": "doc", "title": "Paper", "evidence": [
        {"kind": "highlight", "id": f"hl_{i}", "snippet": f"accumulation precision {i}"} for i in range(5)]}]}
    read = {"items": [{"id": f"hl_{i}", "ok": True, "content": f"text {i}", "metadata": {"annotationText": "Breaks at $K = 4096$." if i == 4 else None}}
                      for i in range(5)]}
    got = Library(CLI({("search", "highlight", None): search, ("read", None, None): read})).retrieve("accumulation precision FP16")
    assert got[0]["id"] == "hl_4" and got[0]["annotation"] == "Breaks at K = 4096."
    listed = {"items": [{"id": "hl_l", "kind": "highlight", "preview": "accumulation precision in FP16 breaks", "metadata": {"documentTitle": "Paper"}}]}
    read2 = {"items": [{"id": "hl_l", "ok": True, "content": "listed", "metadata": {"annotationText": "mine"}}]}
    one = {"results": [{"kind": "document", "id": "doc", "title": "Paper", "evidence": [{"kind": "highlight", "id": "hl_0", "snippet": "plain"}]}]}
    got = Library(CLI({("search", "highlight", None): one, ("list", "highlight", None): listed, ("read", None, None): read2})).retrieve("accumulation precision FP16")
    assert [x["id"] for x in got] == ["hl_l", "hl_0"]


def test_a_post_mortem_cites_the_retrieved_item_that_best_matches_its_cause():
    from agent.orchestrator import Orchestrator
    items = [{"id": "cnv_1", "kind": "conversation", "excerpt": "Triton tiles and shared memory."},
             {"id": "hl_a", "kind": "highlight", "excerpt": "x", "highlight_text": "Register pressure makes the kernel spill to local memory.", "annotation": "spills"},
             {"id": "hl_b", "kind": "highlight", "excerpt": "y", "highlight_text": "Accumulation error grows with the length K."}]
    assert Orchestrator.best_note(items, "It spills 40 words per thread to local memory.")["id"] == "hl_a"
    assert Orchestrator.best_note(items, "Adversarial error grows with K: accumulation length.")["id"] == "hl_b"
    assert Orchestrator.best_note([], "anything") is None


def test_the_best_run_is_ranked_by_latency_and_reported_with_the_same_metric(tmp_path):
    class Windows(Engine):
        """r2 is faster in absolute latency, but its window measured naive slower, so its same-window ratio is the higher one for r3."""
        def __call__(self, spec, run_id):
            m = super().__call__(spec, run_id)
            if spec.get("refs") and m.latency_ms and spec["params"] != P:
                m.speedups["naive"] = {0.628: 2.66, 0.640: 2.69}[m.latency_ms]
            return m

    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"warps": 8}, 2.0, "naive"), h(P | {"stages": 4}, 2.0, "naive"), default="no json"),
             engine=Windows(), main_iters=3)
    o.run()
    best = of(o, "session_finished")[0]["best"]
    assert best == {"run": "r2", "vs_naive": round(1.891 / 0.628, 2), "vs_eager": round(0.669 / 0.628, 2)}
    assert of(o, "final_report")[0]["report"]["best"]["run"] == "r2" and of(o, "final_report")[0]["report"]["best"]["vs_naive"] == best["vs_naive"]


def test_a_hunch_is_a_retest_only_if_the_earlier_run_changed_no_other_knob(tmp_path):
    from harness.kernel import Params
    o = hybrid_state(tmp_path, Params(128, 128, 64, 8, 4, acc="fp16", gelu="approx"))
    o.hunch(o.branches["prec"], "Try the tanh approximate GELU.", "Hunch from the Speed lens in directive d3", "d3", "speed", {"gelu": "approx"})
    text = of(o, "plan_change")[-1]["text"]
    assert "This is not a retest: it has not been tested in isolation, H2 changed it together with warps, acc" in text and "H2 (" not in text
    assert o.tested({"gelu": "approx"}) is None and o.hints["prec"].params is None  # a first test goes to the model
    o.hyps[2].params = Params(128, 128, 64, 4, 4, gelu="approx")
    assert o.tested({"gelu": "approx"}) is o.hyps[2]


def test_claim_naming_helper():
    assert hyp.named("Use 256x128x64 tiles", "block_m", 256) and hyp.named("Use 256x128x64 tiles", "block_k", 64)
    assert not hyp.named("Use 256x128 tiles", "block_k", 64) and hyp.named("go to 8 warps", "warps", 8) and hyp.named("the tanh GELU", "gelu", "approx")
    assert not hyp.named("FP32 accumulation", "acc", "fp16")


def test_a_pivot_claim_states_every_change_from_the_pivot_config(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.02), h(P | {"warps": 8, "stages": 4}, 1.02, claim="Going to 8 warps should help."),
                             default="no json"), main_iters=1, branch_iters=1)
    run_with(o, [(1, "speed", "Faster."), (2, "precision", "Max error under 1e-3 on both batches.")])
    assert o.branches["speed2"].pivot.dict() == P
    h2 = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "speed2")
    assert h2["claim"] == ("Going to 8 warps should help. Relative to the pivot config (128x128x32 tiles, 4 warps, 3 stages, FP32 accumulation, exact GELU) "
                           "it changes warps 4 to 8, stages 3 to 4.")
    prompt = next(m for _, tag, m in o.ask.calls if tag.startswith("speed2-"))[1]["content"]
    assert "This branch pivots to 128x128x32 tiles, 4 warps, 3 stages, FP32 accumulation, exact GELU." in prompt
