"""Cross-session repeat guard: fingerprinted [RWR] notes, the refusal of a repeated experiment, and the human's confirm card. The papiers CLI
is mocked, no test reaches Papiers or needs a token."""

import json
import threading
import time

import pytest
from test_agent import Asker, Engine, P, assert_contract, h, make, of, types
from test_writeback import NOTES, verdict

from agent import fingerprint as fp
from agent import writeback as wb
from agent.papiers import Library, parse_note
from harness.kernel import NAIVE

BASE = NAIVE.dict()
START = {k: v for k, v in P.items() if BASE[k] != v}  # the change from the naive kernel to P
DEFAULT = {"standard": 1e-3, "adversarial": None, "set_by": "default"}
SPEED = {"standard": 2e-2, "adversarial": None, "set_by": "speed"}


def prior(verdict="confirmed", change=None, base=BASE, bound=DEFAULT, rounds=7, run="20261018-201500/H1"):
    note = f"[RWR] run {run} | {verdict} | 1.54x vs >= 1.20x | Reason." + fp.tail(change or START, base, bound, rounds, "Title")
    return {"id": f"hl_{verdict}", "doc_id": "doc_mp", "document_title": "MIXED PRECISION TRAINING"} | parse_note(note)


def hyp_event(hid="H0", params=P, base=BASE, **kw):
    return {"t": 0, "type": "hypothesis_proposed", "hypothesis": {"id": hid, "claim": "Approximate GELU, tanh form, cuts latency.", "params": params,
                                                                   "base": base, "papiers": "hl_1"} | kw}


# ---- the note ----

def test_the_note_carries_a_fingerprint_that_round_trips():
    events = [{"t": 0, "type": "papiers_context", "query": "q", "note": NOTES["hl_1"]}, hyp_event(params=BASE | {"gelu": "approx"}),
              verdict("H0", "confirmed", [], "Prediction holds.") | {"checks": [{"name": "speedup vs best", "predicted": ">= 1.20x", "observed": "1.5x", "pass": True}]}]
    (it,), _ = wb.plan(events, "s1")
    assert it["note"].endswith(" | params: gelu=approx | base: block_m=16 block_n=16 block_k=16 warps=1 stages=1 acc=fp32 gelu=exact "
                               "| kernel: matmul_bias_gelu | title: Approximate GELU | bounds: std 1.0e-3 adv none | rounds: 7")
    got = parse_note(it["note"])
    assert got["fp"] == {"params": {"gelu": "approx"}, "base": BASE, "kernel": "matmul_bias_gelu", "title": "Approximate GELU",
                         "bounds": {"std": "1.0e-3", "adv": "none"}, "rounds": 7}
    assert got["run"] == "s1/H0" and got["verdict"] == "confirmed" and got["reason"] == "Prediction holds."


def test_a_retest_note_records_its_rounds_and_a_precision_bound_both_batches():
    events = [{"t": 0, "type": "papiers_context", "query": "q", "note": NOTES["hl_1"]}, hyp_event(rounds=14),
              verdict("H0", "inconclusive", [{"name": "speedup vs best", "predicted": ">= 1.10x", "observed": "1.07x", "pass": None}], "Noise.")
              | {"bound": {"standard": 1e-3, "adversarial": 1e-3, "set_by": "precision"}}]
    (it,), _ = wb.plan(events, "s1")
    assert parse_note(it["note"])["fp"]["rounds"] == 14 and parse_note(it["note"])["fp"]["bounds"] == {"std": "1.0e-3", "adv": "1.0e-3"}


def test_old_notes_still_parse_and_are_context_only():
    old = "[RWR] run s0/H1 | rejected | std 1.3e-2 vs < 1.0e-3 | standard max error 1.3e-2 missed < 1.0e-3."
    p = {"id": "hl_old", "doc_id": "d"} | parse_note(old)
    assert "fp" not in p and p["verdict"] == "rejected"
    assert fp.blocking([p], BASE, P, DEFAULT) == (None, None)
    assert fp.status(p, DEFAULT) == "context, old note without a fingerprint"
    assert parse_note(old + " | params: gelu=approx") == parse_note(old)  # a truncated tail is ignored, not an error


# ---- the guard ----

def run(tmp_path, priors, asker, engine=None, **kw):
    o = make(tmp_path, asker, engine, main_iters=1, read_back=lambda: priors, **kw)
    o.run()
    return o


@pytest.mark.parametrize("word", ["confirmed", "rejected", "inconclusive", "provisional"])
def test_a_matching_proposal_is_refused_before_any_gpu_time_for_every_verdict(tmp_path, word):
    engine = Engine()
    asker = Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 2.0, "naive"))
    o = run(tmp_path, [prior(word)], asker, engine)
    (e,) = of(o, "prior_refused")
    assert e["branch"] == "main" and e["prior"]["run"] == "20261018-201500/H1" and e["prior"]["verdict"] == word and e["prior"]["measured"] == "1.54x"
    assert "already ran" in e["text"] and "20261018-201500/H1" in e["text"] and "1.54x vs >= 1.20x" in e["text"]
    assert [x["params"]["stages"] for x in (e["hypothesis"] for e in of(o, "hypothesis_proposed"))] == [4]
    assert [c[0]["params"]["stages"] for c in engine.calls[2:]] == [4]  # no run for the refused proposal
    assert "Invalid: " + e["text"] in asker.calls[1][2][-1]["content"]
    assert any(x["text"].startswith("Refused a proposal on main: it repeats") for x in of(o, "agent_log"))
    assert_contract(o.log.events)


def test_the_inconclusive_retest_exception_needs_more_rounds_than_the_prior_run(tmp_path):
    engine = Engine()
    retest = json.loads(h(P, 2.0, "naive")) | {"retest_rounds": 14}
    o = run(tmp_path, [prior("inconclusive")], Asker(json.dumps(retest)), engine)
    assert of(o, "prior_refused") == []
    assert of(o, "hypothesis_proposed")[0]["hypothesis"]["rounds"] == 14
    assert engine.calls[2][0]["rounds"] == 14
    assert any("retests 20261018-201500/H1" in e["text"] and "14 rounds" in e["text"] for e in of(o, "agent_log"))

    same = json.loads(h(P, 2.0, "naive")) | {"retest_rounds": 7}
    o = run(tmp_path / "b", [prior("inconclusive")], Asker(json.dumps(same), h(P | {"stages": 4}, 2.0, "naive")))
    assert of(o, "prior_refused") == [] and "retest_rounds must be an integer above 7" in [c[2][-1]["content"] for c in o.ask.calls][1]

    o = run(tmp_path / "c", [prior("confirmed")], Asker(json.dumps(retest), h(P | {"stages": 4}, 2.0, "naive")))
    assert len(of(o, "prior_refused")) == 1  # more rounds never reopen a confirmed or rejected verdict


def test_a_retest_with_more_rounds_than_the_cap_is_refused_by_the_parser():
    assert hyp_parse(json.loads(h(P)) | {"retest_rounds": 15})[0] is None
    assert hyp_parse(json.loads(h(P)) | {"retest_rounds": 14})[0]["rounds"] == 14
    assert hyp_parse(json.loads(h(P)))[0]["rounds"] is None


def hyp_parse(d):
    from agent import hypothesis as hyp
    return hyp.parse(json.dumps(d))


def test_a_prior_verdict_under_different_bounds_does_not_block_and_is_labelled(tmp_path):
    p = prior("confirmed", bound={"standard": 2e-2, "adversarial": None})
    asker = Asker(h(P, 2.0, "naive"))
    o = run(tmp_path, [p], asker)
    assert of(o, "prior_refused") == [] and of(o, "verdict")[0]["verdict"] == "confirmed"
    text = asker.calls[0][2][1]["content"]
    assert "context, bounds differ (now std 1.0e-3 adv none)" in text and "std 2.0e-2 adv none" in text


def test_a_different_base_config_does_not_block(tmp_path):
    o = run(tmp_path, [prior("confirmed", base=BASE | {"warps": 2})], Asker(h(P, 2.0, "naive")))
    assert of(o, "prior_refused") == []


def test_the_prompt_table_and_the_logged_section_are_the_same_text(tmp_path):
    asker = Asker(h(P, 2.0, "naive"))
    o = run(tmp_path, [prior("rejected"), prior("confirmed", bound={"standard": 2e-2, "adversarial": None}, run="20261017-101500/H2")], asker)
    (e,) = of(o, "prior_verdicts")
    sent = asker.calls[0][2][1]["content"]
    assert e["section"] in sent and e["count"] == 2
    lines = e["section"].splitlines()
    assert lines[0].startswith("Prior verdicts from earlier sessions, read from Papiers (past results, not evidence, never cite")
    assert "refuses, before any run, a proposal that makes the same change from the same start config" in lines[1]
    assert lines[2] == "| title | change | from | verdict | numbers | bounds | session | status |"
    assert lines[4].endswith("| blocks |") and "m=128 n=128 k=32 warps=4 stages=3" in lines[4] and "20261018-201500/H1" in lines[4]  # blocking rows first
    assert lines[5].endswith("| context, bounds differ (now std 1.0e-3 adv none) |")


def test_a_bound_directive_that_changes_which_rows_block_logs_the_new_section(tmp_path):
    asker = Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), h(P | {"stages": 4, "warps": 8}, 1.05), default="no json")
    o = make(tmp_path, asker, main_iters=1, branch_iters=2, read_back=lambda: [prior("rejected", bound=SPEED)])
    from test_agent import run_with
    run_with(o, [(1, "speed", "Push throughput harder.")])
    sections = [(e.get("branch"), e["section"]) for e in of(o, "prior_verdicts")]
    assert sections[0][0] is None and "context, bounds differ" in sections[0][1]
    assert sections[1][0] == "speed" and sections[1][1].splitlines()[4].endswith("| blocks |")


def test_zero_priors_log_n0_and_send_no_section(tmp_path):
    o = run(tmp_path, [], Asker(h(P, 2.0, "naive")))
    (e,) = of(o, "prior_verdicts")
    assert (e["count"], e["items"], e["section"]) == (0, [], "")
    assert any(x["text"].startswith("Prior verdicts: 0 read from Papiers") for x in of(o, "agent_log"))
    assert "Prior verdicts" not in o.ask.calls[0][2][1]["content"]


def test_ignoring_prior_verdicts_skips_the_read_and_the_guard(tmp_path):
    reads = []
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, read_back=lambda: reads.append(1) or [prior("confirmed")])
    o.ignore_prior = True
    o.run()
    assert reads == [] and of(o, "prior_refused") == []
    (e,) = of(o, "prior_verdicts")
    assert (e["count"], e["ignored"], e["section"]) == (0, True, "")
    assert of(o, "verdict")[0]["verdict"] == "confirmed"


def test_the_seeded_fp16_variant_is_skipped_when_an_earlier_session_ran_it_under_the_speed_bound(tmp_path):
    from test_agent import run_with
    seeded = prior("rejected", change={"acc": "fp16"}, base=P, bound=SPEED)
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"stages": 4}, 1.05), default="no json"), main_iters=1, branch_iters=1, read_back=lambda: [seeded])
    run_with(o, [(1, "speed", "Push throughput harder.")])
    assert all(x["hypothesis"]["params"]["acc"] == "fp32" for x in of(o, "hypothesis_proposed"))
    (e,) = of(o, "prior_refused")
    assert e["seed"] is True and e["branch"] == "speed"
    assert any("Skipped the seeded FP16 variant on speed" in x["text"] and "Ignore prior verdicts" in x["text"] for x in of(o, "agent_log"))


# ---- human steering ----

def drive(o, hunch, answer, wait=lambda o: bool(of(o, "prior_duplicate"))):
    t = threading.Thread(target=o.run)
    t.start()
    sent = False
    while t.is_alive():
        if not sent and any(x.judgment for x in o.hyps):
            o.command({"type": "add_directive", "lens": "speed", "text": hunch})
            sent = True
        if sent and wait(o) and answer:
            o.command(answer.pop(0))
        time.sleep(0.01)


TANH = prior("confirmed", change={"gelu": "approx"}, base=P, bound=SPEED)


def test_a_hunch_matching_a_prior_verdict_warns_and_runs_only_after_confirmation(tmp_path):
    asker = Asker(h(P, 2.0, "naive"), h(P | {"gelu": "approx"}, 1.02), default="no json")
    o = make(tmp_path, asker, main_iters=1, branch_iters=2, read_back=lambda: [TANH])
    drive(o, "try the tanh GELU approximation", [{"type": "confirm_prior", "help": "d1", "run": True}])
    (w,) = of(o, "prior_duplicate")
    assert w["help"] == "d1" and w["prior"]["run"] == "20261018-201500/H1" and "already ran this" in w["text"] and "Confirm to run it anyway" in w["text"]
    assert of(o, "prior_refused") == []
    ran = [x["hypothesis"]["params"]["gelu"] for x in of(o, "hypothesis_proposed")]
    assert ran == ["exact", "exact", "approx"]  # H0, the seeded FP16 variant, then the confirmed hunch
    assert types(o).index("prior_duplicate") < max(i for i, t in enumerate(types(o)) if t == "hypothesis_proposed")
    assert_contract(o.log.events)


def test_a_skipped_hunch_never_runs_and_is_reported(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), default="no json"), main_iters=1, branch_iters=2, read_back=lambda: [TANH])
    drive(o, "try the tanh GELU approximation", [{"type": "confirm_prior", "help": "d1", "run": False}])
    assert [x["hypothesis"]["params"]["gelu"] for x in of(o, "hypothesis_proposed")] == ["exact", "exact"]
    assert any(e["help"] == "d1" and "skipped after the prior verdict warning" in e["text"] for e in of(o, "plan_change"))
    assert o.hunches["d1"] == "not run: skipped after the prior verdict warning"


def test_an_unanswered_hunch_stays_held_and_a_stray_confirmation_is_ignored(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), default="no json"), main_iters=1, branch_iters=2, read_back=lambda: [TANH])
    drive(o, "try the tanh GELU approximation", [{"type": "confirm_prior", "help": "d9", "run": True}])
    assert [x["hypothesis"]["params"]["gelu"] for x in of(o, "hypothesis_proposed")] == ["exact", "exact"]
    assert "d1" in o.held and "no confirmation arrived" in o.hunches["d1"]
    assert any("nothing is waiting on it" in e["text"] for e in of(o, "agent_log"))


def test_a_hunch_under_other_bounds_runs_without_a_card(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"gelu": "approx"}, 1.02), default="no json"), main_iters=1, branch_iters=1,
             read_back=lambda: [prior("confirmed", change={"gelu": "approx"}, base=P, bound={"standard": 5e-3, "adversarial": None})])
    drive(o, "try the tanh GELU approximation", [], wait=lambda o: False)
    assert of(o, "prior_duplicate") == []


# ---- lookup and server ----

def test_the_library_returns_fingerprints_from_the_notes_it_reads():
    from test_agent import CLI
    note = prior("confirmed")
    note_text = (f"[RWR] run {note['run']} | confirmed | 1.54x vs >= 1.20x | Reason." + fp.tail(START, BASE, DEFAULT, 7, "Title"))
    read = {"items": [{"id": "hl_rwr", "ok": True, "content": "x", "metadata": {"documentId": "doc_mp", "annotationText": note_text}}]}
    search = {"results": [{"kind": "document", "id": "doc_mp", "title": "T", "evidence": [{"kind": "highlight", "id": "hl_rwr", "snippet": note_text}]}]}
    (p,) = Library(CLI({("search", "highlight", None): search, ("read", None, None): read})).prior_verdicts()
    assert p["fp"]["params"] == START and p["fp"]["bounds"] == {"std": "1.0e-3", "adv": "none"}


def test_the_server_accepts_the_confirm_and_start_options():
    from server.app import valid
    assert valid({"type": "confirm_prior", "help": "d1", "run": True}) and valid({"type": "confirm_prior", "help": "d1", "run": False})
    assert not valid({"type": "confirm_prior", "help": "d1"}) and not valid({"type": "confirm_prior", "run": True})
    assert valid({"type": "start", "ignore_prior": True})


def test_the_start_command_sets_the_ignore_switch(tmp_path):
    from test_writeback import Orch, client
    o = Orch(tmp_path, [])
    o.ignore_prior = False
    started = []
    o.run = lambda: started.append(o.ignore_prior)
    with client(o) as c, c.websocket_connect("/ws") as ws:
        ws.send_text(json.dumps({"type": "start", "ignore_prior": True}))
        for _ in range(100):
            if started:
                break
            time.sleep(0.02)
    assert started == [True]
