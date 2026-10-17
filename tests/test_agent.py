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
P = dict(block_m=128, block_n=128, block_k=32, warps=4, stages=3, acc="fp32", gelu="exact")


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

    def __init__(self, *script, strong_ok=False, default=None):
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
    return Orchestrator(engine or Engine(), asker, log, "sessions/test", "fast", "strong", BAND, idle_s=0.3, **kw)


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
    "help_requested": {"help"}, "help_answered": {"help", "option", "by"}, "plan_change": {"help", "text"},
}
HELP_KEYS = {"id", "target", "question", "evidence", "options", "default", "blocks", "deadline_s", "what_changes"}
SUMMARY_KEYS = {"branch", "tried", "moved", "verdict", "confidence", "confidence_reason", "uncertainties", "would_change_mind", "next_step"}


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
            assert set(e["run"]["metrics"]) == {"latency_ms", "tflops", "max_error", "adv_error", "vram_mb", "compiled_ok", "correct_ok"}
        if e["type"] in ("verdict", "hypothesis_proposed"):
            b = e["bound"] if e["type"] == "verdict" else e["hypothesis"]["prediction"]["bound"]
            assert set(b) == {"standard", "adversarial", "set_by"} and b["set_by"] in ("speed", "precision", "default")
        if e["type"] == "help_requested":
            assert set(e["help"]) == HELP_KEYS
        if e["type"] == "findings_exported":
            assert all(set(i) == {"kind", "title", "file"} and i["file"].startswith("exports/") for i in e["items"])
        if e["type"] == "post_mortem":
            assert set(e["post_mortem"]) == {"hypothesis", "expected", "observed", "cause", "pivot", "sweep"}


# ---- verdicts ----

def test_verdict_confirmed_rejected_inconclusive():
    pred = {"min_speedup": 2.0, "vs": "naive", "max_error": 1e-3}
    assert judge(pred, met(0.642), 1.891, BAND, 1e-3, None).verdict == "confirmed"
    assert judge(pred, met(0.95), 1.891, BAND, 1e-3, None).verdict == "inconclusive"
    j = judge(pred, met(1.2), 1.891, BAND, 1e-3, None)
    assert j.verdict == "rejected" and j.checks[0].passed is False


def test_verdict_rejects_on_error_and_failed_runs_regardless_of_speed():
    pred = {"min_speedup": 1.3, "vs": "parent_best", "max_error": 1e-3}
    j = judge(pred, met(0.45, err=1.3e-2), 0.642, BAND, 1e-3, None)
    assert j.verdict == "rejected" and [c.passed for c in j.checks] == [True, False]
    assert judge(pred, Metrics(error="timeout"), 0.642, BAND, 1e-3, None).verdict == "rejected"


def test_verdict_adversarial_is_provisional_until_a_precision_bound():
    pred = {"min_speedup": 1.2, "vs": "parent_best", "max_error": 1e-3}
    m = met(0.5, err=5e-4, adv=3e-3)
    j = judge(pred, m, 0.642, BAND, 1e-3, None)
    assert (j.verdict, j.provisional) == ("confirmed", True)
    j = judge(pred, m, 0.642, BAND, 1e-3, 1e-3)
    assert j.verdict == "rejected" and j.checks[-1].name == "adversarial max error"


def test_speed_tolerance_makes_fp16_provisional_and_precision_rejects_it():
    pred = {"min_speedup": 1.3, "vs": "parent_best", "max_error": 2e-2}
    fp16 = met(0.45, err=1.3e-2, adv=7.1e-2)
    j = judge(pred, fp16, 0.642, BAND, 2e-2, None)
    assert (j.verdict, j.provisional) == ("confirmed", True) and j.checks[1].predicted == "< 2.0e-2"
    j = judge(pred, fp16, 0.642, BAND, 1e-3, 1e-3)
    assert j.verdict == "rejected" and [c.passed for c in j.checks] == [True, False, False]
    assert (judge(pred, met(0.6), 0.642, BAND, 2e-2, None).provisional) is False


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
                                                          "verdict", "summary", "branch_status", "findings_exported", "session_finished"]
    v = of(o, "verdict")[0]
    assert v["verdict"] == "confirmed" and v["checks"][0]["observed"] == "2.95x"
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
    pm = of(o, "post_mortem")[0]["post_mortem"]
    assert pm["hypothesis"] == "H1" and [s["k"] for s in pm["sweep"]] == [256, 512, 1024, 2048, 4096]
    assert "acc=fp16" in pm["pivot"]
    assert ("speed", "rejected") in [(e["branch"], e["status"]) for e in of(o, "branch_status")]
    pivot_prompt = next(m for model, tag, m in o.ask.calls if tag.startswith("speed2-"))
    assert "acc=fp16" in pivot_prompt[1]["content"] and "under 1e-03" in pivot_prompt[1]["content"]
    assert of(o, "session_finished")[0]["ledger"][1]["history"] == ["confirmed", "rejected"]
    assert_contract(o.log.events)


def test_harness_variant_tests_fp16_when_the_agent_avoids_it(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05), default="no json"),
             main_iters=1, branch_iters=3)
    run_with(o, [(1, "speed", "Go faster.")])
    variant = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["params"]["acc"] == "fp16")
    assert "Harness variant" in variant["note"]
    v = next(e for e in of(o, "verdict") if e["hypothesis"] == variant["id"])
    assert (v["verdict"], v["provisional"]) == ("confirmed", True)


def test_precision_directive_rejects_a_provisional_result_without_a_new_run(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"block_n": 64}, 1.2), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05)),
             main_iters=1, branch_iters=1)
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
    assert of(o, "summary")[0]["summary"]["uncertainties"] == ["one shape only"]


def test_unsupported_commands_are_logged(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1)
    o.command({"type": "approve_hybrid", "branches": ["a", "b"]})
    o.run()
    assert any("approve_hybrid" in e["text"] for e in of(o, "agent_log"))


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

    with TestClient(create(factory)) as client:
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
            while "directive_added" not in seen:
                seen.append(json.loads(ws.receive_text())["type"])
            assert "hypothesis_proposed" in seen


def test_pivots_stop_at_three_branches_per_lens(tmp_path):
    fp16 = {"acc": "fp16"}
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), *[h(P | fp16 | {"warps": w}) for w in (1, 2, 8, 16)], default="no json"),
             main_iters=1, branch_iters=1)
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
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.5), default="no json"), main_iters=1, branch_iters=3)
    run_with(o, [(1, "speed", "Faster.")])
    ev = o.log.events
    v1 = next(i for i, e in enumerate(ev) if e["type"] == "verdict" and e["hypothesis"] == "H1")
    assert ev[v1]["verdict"] == "rejected" and ev[v1]["checks"][0]["pass"] is False
    nxt = next(i for i, e in enumerate(ev) if i > v1 and e["type"] == "hypothesis_proposed")
    between = [e["type"] for e in ev[v1:nxt]]
    assert "post_mortem" not in between and "branch_created" not in between
    assert ev[nxt]["hypothesis"]["branch"] == "speed"


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
    assert of(o, "hypothesis_proposed")[0]["hypothesis"]["note"] == "Context from Papiers note cnv_1"
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


def test_compile_failure_is_a_miss_not_a_fork(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"block_k": 128}, 1.1), default="no json"), main_iters=1, branch_iters=2)
    run_with(o, [(1, "speed", "Faster.")])
    v1 = next(e for e in of(o, "verdict") if e["hypothesis"] == "H1")
    assert v1["verdict"] == "rejected" and v1["checks"][0]["name"] == "compiles and runs"
    assert not [e for e in of(o, "post_mortem") if e["post_mortem"]["hypothesis"] == "H1"]


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
    assert "inside the noise band" in req["question"] and req["evidence"][0].startswith("r2:")
    ev = o.log.events
    i = ev.index(of(o, "help_requested")[0])
    assert ev[i + 1] == {**ev[i + 1], "type": "branch_status", "branch": "speed", "status": "waiting_on_human"}
    assert of(o, "verdict")[1]["verdict"] == "inconclusive"
    # no answer: the deadline passes and the branch proceeds on the default
    pc = of(o, "plan_change")[0]
    assert pc["help"] == "hq1" and "default assumption" in pc["text"]
    assert ("speed", "running") in [(e["branch"], e["status"]) for e in ev[i:] if e["type"] == "branch_status"]
    assert_contract(ev)


def test_suggestion_answer_becomes_the_next_hypothesis_and_is_exported(tmp_path):
    o = help_session(tmp_path, h(P | {"gelu": "approx"}, 1.05))
    drive(o, [(judged(o, 1), SPEED), (asked(o), {"type": "answer_help", "help": "hq1", "option": "other", "note": "Try the tanh approximate GELU.", "by": "speed"})])
    ans = of(o, "help_answered")[0]
    assert ans == {**ans, "help": "hq1", "option": "other", "note": "Try the tanh approximate GELU.", "by": "speed"}
    assert "suggests" in of(o, "plan_change")[0]["text"]
    nxt = next(e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["t"] >= ans["t"])
    assert nxt["note"] == "Suggested by the Speed lens in answer to hq1" and nxt["params"]["gelu"] == "approx"
    prompt = next(m for model, tag, m in o.ask.calls if tag.startswith("speed-2"))
    assert "Try the tanh approximate GELU." in prompt[1]["content"]
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
    assert "accepts" in of(o, "plan_change")[0]["text"]


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


def test_seeded_variant_still_runs_on_speed_when_fp16_was_only_tried_on_main(tmp_path):
    fp16_main = P | {"acc": "fp16", "warps": 2}
    o = make(tmp_path, Asker(h(fp16_main, 2.0, "naive"), h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), default="no json"), main_iters=2, branch_iters=3)
    run_with(o, [(2, "speed", "Faster.")])
    on_speed = [e["hypothesis"] for e in of(o, "hypothesis_proposed") if e["hypothesis"]["branch"] == "speed"]
    variant = next(x for x in on_speed if x["params"]["acc"] == "fp16")
    assert "Harness variant" in variant["note"] and variant["prediction"]["min_speedup"] == 1.2
