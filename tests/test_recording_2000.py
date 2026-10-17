"""Regression tests from the frozen 20:00 UTC recording (demo/replay.jsonl, runs/sessions/20261017-155752). Each test rebuilds the state
the recording reached from its logged proposals and measured metrics."""

import json
from pathlib import Path

from test_agent import BAND, CLI, make, of

from agent import hypothesis as hyp
from agent.orchestrator import NAIVE, Branch, Hyp, claim_text
from agent.papiers import Library
from agent.verdict import judge
from harness.kernel import Params
from harness.metrics import Metrics

EVENTS = [json.loads(x) for x in (Path(__file__).parent.parent / "demo" / "replay.jsonl").read_text().splitlines() if x.strip()]
PROPS = {e["hypothesis"]["id"]: e["hypothesis"] for e in EVENTS if e["type"] == "hypothesis_proposed"}
RUNS = {e["run"]["hypothesis"]: e["run"] for e in EVENTS if e["type"] == "run_finished"}
DIRS = [e["directive"] for e in EVENTS if e["type"] == "directive_added"]
BASE = next(e for e in EVENTS if e["type"] == "session_started")["baselines"]
BRANCHES = [("main", [], "both"), ("speed", ["main"], "speed"), ("speed2", ["speed"], "speed"), ("prec", ["main"], "precision")]
ALL = [f"H{i}" for i in range(8)]
DEFAULT = {"standard": 1e-3, "adversarial": None, "set_by": "default"}


def state(tmp_path, ids):
    o = make(tmp_path, None, max_hyps=8)
    o.naive, o.eager = (Metrics.from_dict(BASE[k]) for k in ("naive", "eager"))
    for d in DIRS:
        o.directives.append(d | hyp.directive(d["lens"], d["text"]) | {"at": -1000})
    for bid, parents, lens in BRANCHES:
        o.branches[bid] = Branch(bid, parents, bid, lens, "rejected" if bid == "speed" else "done")
    o.branches["speed2"].pivot = Params(**PROPS["H0"]["params"])
    for i in ids:
        x = PROPS[i]
        m = Metrics.from_dict(RUNS[i]["metrics"])
        h = Hyp(i, x["branch"], x["claim"], x["prediction"], Params(**x["params"]), run=RUNS[i]["id"], metrics=m)
        h.base = Params(**PROPS["H0"]["params"]) if i == "H1" else None
        h.bonus = i == "H4"
        h.ref = "naive" if x["prediction"]["vs"] == "naive" else "parent"
        h.speedup, h.noise = m.speedups[h.ref], max(BAND, m.spreads.get(h.ref, 0.0))
        bound = o.bounds(o.branches[h.branch].lens)
        o.set_verdict(h, judge(x["prediction"] | {"max_error": bound["standard"]}, m, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
        o.hyps.append(h)
    return o


def test_a_claim_from_the_naive_start_names_the_config_without_the_delta_list(tmp_path):
    h0 = PROPS["H0"]
    claim = claim_text(NAIVE, Params(**h0["params"]), h0["prediction"], DEFAULT)
    assert claim.startswith("128x128x64 tiles, 4 warps, 4 stages, FP32 accumulation, exact GELU is at least 2.00x faster than naive, "
                            "with standard max error under 1.0e-3")
    assert "to 128" not in claim
    assert claim_text(Params(**h0["params"]), Params(**PROPS["H2"]["params"]), PROPS["H2"]["prediction"], DEFAULT).startswith("Stages 4 to 3, giving ")
    o = state(tmp_path, ["H0"])
    o.hyps[0].claim = claim
    o.summarize(o.hyps[0], o.branches["main"], "")
    assert o.summaries["main"]["tried"] == claim


def test_an_inconclusive_best_run_is_rerun_first_and_every_next_step_is_a_sentence(tmp_path):
    o = state(tmp_path, ALL)
    for h in o.hyps:
        if h.judgment.verdict != "confirmed":
            o.post_mortem(h)
    o.finish()
    report = of(o, "final_report")[0]["report"]
    assert report["best"]["hypothesis"] == "H4" and report["best"]["verdict"] == "inconclusive"
    assert report["next"][0].startswith("Rerun H4 (gelu exact to approx) with more timing rounds")
    assert len(report["next"]) == 3 and all(x.endswith(".") and ": register" not in x for x in report["next"])
    assert o.pms["H7"]["pivot"].startswith("Revert block_k 64 to 128, warps 4 to 8, stages 4 to 2, since register pressure is not what limits this config")


def test_no_hybrid_is_logged_once_per_kind_of_reason(tmp_path):
    o = state(tmp_path, ALL[:6])
    runs, o.hyps = o.hyps, o.hyps[:3]
    o.maybe_hybrid()
    for h in runs[3:]:  # H4 and H5 each give a new donor whose merge already ran, the recording logged one line per donor
        o.hyps.append(h)
        o.maybe_hybrid()
    logs = [e["text"] for e in of(o, "agent_log") if e["text"].startswith("No hybrid")]
    assert len(logs) == 1 and "already ran as" in logs[0]


def test_the_post_mortem_states_the_bound_the_verdict_was_judged_under(tmp_path):
    o = state(tmp_path, ALL)
    assert PROPS["H2"]["prediction"]["max_error"] == 0.02
    o.post_mortem(o.hyps[2])
    assert o.pms["H2"]["expected"] == "Speedup of at least 1.02x vs the best so far with standard max error under 1.0e-3 and adversarial max error under 1.0e-3."


def test_tiling_queries_take_a_listed_highlight_before_a_conversation_and_skip_passages():
    listed = {"items": [{"id": "hl_t", "kind": "highlight", "preview": "larger tiles reuse each loaded block", "metadata": {"documentTitle": "Triton"}}]}
    convo = {"results": [{"kind": "conversation", "id": "cnv_9", "title": "Chat", "snippet": "about tiles"}]}
    cli = CLI({("list", "highlight", None): listed, ("search", "conversation", None): convo, ("read", None, None): {"items": []}})
    assert [x["id"] for x in Library(cli).retrieve("tile size shared memory")] == ["hl_t"]
    assert not [a for a in cli.calls if "conversation" in a]
    cli = CLI({("search", "conversation", None): convo})
    assert [x["kind"] for x in Library(cli).retrieve("tile size shared memory")] == ["conversation"]
    assert not [a for a in cli.calls if "passage" in a]
