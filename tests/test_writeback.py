"""Papiers write-back and read-back, with the papiers CLI mocked throughout. No test reaches Papiers or needs a token."""

import json
import subprocess

import pytest
from test_agent import CLI, Asker, P, assert_contract, h, make, of

from agent import writeback as wb
from agent.events import Log
from agent.papiers import Library, citable, parse_note

PASSAGE = 'To maintain accuracy, the "partial products" accumulate into an FP32 value, which is converted to FP16 before writing to memory again.'
NOTES = {
    "hl_1": {"id": "hl_1", "kind": "highlight", "title": "MIXED PRECISION TRAINING", "document_title": "MIXED PRECISION TRAINING", "excerpt": "x",
             "doc_id": "doc_mp", "page": 3, "family": "pdf", "anchor": PASSAGE},
    "hl_web": {"id": "hl_web", "kind": "highlight", "title": "Triton tutorial", "excerpt": "x", "doc_id": "doc_web", "family": "web_clip",
               "anchor": "Short web passage on tiles."},
    "cnv_1": {"id": "cnv_1", "kind": "conversation", "title": "Chat", "excerpt": "x"},
    "hl_nodoc": {"id": "hl_nodoc", "kind": "highlight", "title": "T", "excerpt": "x", "page": 2, "anchor": "text"},
    "hl_nopage": {"id": "hl_nopage", "kind": "highlight", "title": "T", "excerpt": "x", "doc_id": "doc_a", "family": "pdf", "anchor": "text"},
    "hl_notext": {"id": "hl_notext", "kind": "highlight", "title": "T", "excerpt": "x", "doc_id": "doc_a", "page": 2},
    "hl_cut": {"id": "hl_cut", "kind": "highlight", "title": "T", "excerpt": "x", "doc_id": "doc_a", "page": 2, "anchor": "text", "truncated": True},
}
SPEED = {"name": "speedup vs best", "predicted": ">= 1.20x", "observed": "1.54x (noise +/- 0.03x)", "pass": True}
STD_FAIL = {"name": "standard max error", "predicted": "< 1.0e-3", "observed": "1.3e-2", "pass": False}
STD_OK = {"name": "standard max error", "predicted": "< 2.0e-2", "observed": "1.3e-2", "pass": True}
NOISE = {"name": "speedup vs best", "predicted": ">= 1.10x", "observed": "1.07x (noise +/- 0.05x)", "pass": None}


def verdict(hid, word, checks, reason, provisional=False):
    return {"t": 0, "type": "verdict", "hypothesis": hid, "verdict": word, "provisional": provisional, "reason": reason, "checks": checks,
            "bound": {"standard": 1e-3, "adversarial": None, "set_by": "default"}}


def session(cited):
    """A finished run's events: hypothesis id -> the Papiers id it cited, H1 flips from provisional to rejected."""
    out = [{"t": 0, "type": "papiers_context", "query": "q", "note": n} for n in NOTES.values()]
    out += [{"t": 0, "type": "hypothesis_proposed", "hypothesis": {"id": hid, "papiers": c} if c else {"id": hid}} for hid, c in cited.items()]
    for hid in cited:
        if hid == "H1":
            out.append(verdict(hid, "confirmed", [SPEED, STD_OK], "Predictions hold under the 2.0e-2 tolerance.", provisional=True))
            out.append(verdict(hid, "rejected", [SPEED, STD_FAIL], "standard max error 1.3e-2 missed < 1.0e-3."))
        elif hid == "H2":
            out.append(verdict(hid, "inconclusive", [NOISE, STD_OK | {"predicted": "< 1.0e-3", "observed": "5.0e-4"}],
                               "The speedup is inside the noise band around the predicted threshold."))
        elif hid == "H3":
            out.append(verdict(hid, "confirmed", [SPEED, STD_OK], "Predictions hold under the 2.0e-2 tolerance. Standard error 1.3e-2 is over "
                                                                  "the strict 1.0e-3 bound | which no lens enforces yet.", provisional=True))
        else:
            out.append(verdict(hid, "confirmed", [SPEED, STD_OK | {"predicted": "< 1.0e-3", "observed": "4.9e-4"}], "Prediction holds."))
    return out + [{"t": 0, "type": "session_finished", "ledger": [], "best": {}}]


class Call:
    """Stands in for subprocess.run: exit 1 for documents in fail, records every argument list."""

    def __init__(self, fail=(), stderr="error: anchor text not found on page"):
        self.fail, self.stderr, self.calls = set(fail), stderr, []

    def __call__(self, args, **kw):
        self.calls.append((args, kw))
        code = 1 if args[2] in self.fail else 0
        return subprocess.CompletedProcess(args, code, "" if code else '{"ok": true}', self.stderr if code else "")


@pytest.fixture
def exe(monkeypatch):
    monkeypatch.setattr(wb.papiers, "binary", lambda: "/opt/papiers")
    return "/opt/papiers"


# ---- items from verdicts ----

def test_items_come_from_the_final_verdicts_with_the_note_format_and_colours():
    pending, skipped = wb.plan(session({"H0": "hl_1", "H1": "hl_1", "H2": "hl_web", "H3": "hl_1"}), "s1")
    by = {it["id"]: it for it in pending}
    assert skipped == []
    assert by["H0"]["note"] == "[RWR] run s1/H0 | confirmed | 1.54x, std 4.9e-4 vs >= 1.20x, < 1.0e-3 | Prediction holds."
    assert by["H1"]["note"] == "[RWR] run s1/H1 | rejected | std 1.3e-2 vs < 1.0e-3 | standard max error 1.3e-2 missed < 1.0e-3."
    assert by["H2"]["note"] == "[RWR] run s1/H2 | inconclusive | 1.07x vs >= 1.10x | The speedup is inside the noise band around the predicted threshold."
    assert by["H3"]["verdict"] == "provisional" and by["H3"]["note"].startswith("[RWR] run s1/H3 | provisional | 1.54x, std 1.3e-2 vs >= 1.20x, < 2.0e-2 | ")
    assert all(it["note"].startswith("[RWR] ") and it["note"].count(" | ") == 3 for it in pending)  # a | in the reason never adds a field
    assert len(by["H3"]["note"].split(" | ")[-1]) <= wb.REASON_CHARS
    assert {k: by[k]["color"] for k in by} == {"H0": "#5fb85f", "H1": "#e05d5d", "H2": "#ffd400", "H3": "#ffd400"}
    assert wb.COLORS["inconclusive"] == wb.COLORS["provisional"]


def test_document_page_and_passage_come_from_the_stored_highlight():
    (it,), _ = wb.plan(session({"H0": "hl_1"}), "s1")
    assert (it["doc_id"], it["pages"], it["doc"], it["passage"]) == ("doc_mp", "3", "MIXED PRECISION TRAINING", PASSAGE)
    assert it["start"] == "To maintain accuracy, the \"partial products\" accumulate into"
    assert it["end"] == "converted to FP16 before writing to memory again."
    assert PASSAGE.startswith(it["start"]) and PASSAGE.endswith(it["end"])
    (web,), _ = wb.plan(session({"H0": "hl_web"}), "s1")
    assert web["pages"] is None and web["end"] is None  # a webpage has no pages, a short passage needs no end anchor


def test_verdicts_without_a_usable_passage_are_skipped_with_the_reason():
    _, skipped = wb.plan(session({"H0": None, "H1": "cnv_1", "H2": "hl_nodoc", "H3": "hl_nopage", "H4": "hl_notext", "H5": "hl_cut", "H6": "hl_gone"}), "s1")
    assert [(x["id"], x["verdict"], x["reason"]) for x in skipped] == [
        ("H0", "confirmed", "no Papiers passage cited"),
        ("H1", "rejected", "cited a conversation, not a document passage"),
        ("H2", "inconclusive", "no document id in the stored data"),
        ("H3", "provisional", "no page in the stored data"),
        ("H4", "confirmed", "no exact passage text in the stored data"),
        ("H5", "confirmed", "the stored passage text is truncated"),
        ("H6", "confirmed", "the cited item hl_gone is not in the run log"),
    ]


def test_the_frozen_demo_replay_has_no_stored_locations_so_everything_is_skipped():
    from pathlib import Path
    events = wb.load(Path(__file__).parents[1] / "demo" / "replay.jsonl")
    p = wb.preview(events, "demo")
    assert p["finished"] and p["pending"] == [] and len(p["skipped"]) == 8


# ---- apply ----

def test_the_cli_gets_an_argument_list_never_a_shell_string(tmp_path, exe):
    call = Call()
    wb.apply(session({"H0": "hl_1", "H1": "hl_web"}), "s1", ["H0", "H1"], tmp_path / wb.LEDGER, call)
    (a0, kw), (a1, _) = call.calls
    assert a0 == [exe, "highlight", "doc_mp", "--pages", "3", "--start", "To maintain accuracy, the \"partial products\" accumulate into",
                  "--end", "converted to FP16 before writing to memory again.",
                  "--note", "[RWR] run s1/H0 | confirmed | 1.54x, std 4.9e-4 vs >= 1.20x, < 1.0e-3 | Prediction holds.", "--color", "#5fb85f"]
    assert a1[:5] == [exe, "highlight", "doc_web", "--start", "Short web passage on tiles."] and "--pages" not in a1 and "--end" not in a1
    assert kw["capture_output"] and kw["timeout"] == wb.TIMEOUT_S and "shell" not in kw


def test_applying_twice_never_writes_twice(tmp_path, exe):
    events, path, call = session({"H0": "hl_1", "H1": "hl_web"}), tmp_path / wb.LEDGER, Call()
    assert [r["status"] for r in wb.apply(events, "s1", ["H0"], path, call)] == ["written"]
    assert [r["status"] for r in wb.apply(events, "s1", ["H0", "H1"], path, call)] == ["already_written", "written"]
    assert len(call.calls) == 2
    book = json.loads(path.read_text())
    assert set(book["written"]) == {"doc_mp|To maintain accuracy, the \"partial products\" accumulate into|s1/H0", "doc_web|Short web passage on tiles.|s1/H1"}
    assert [x["written"] for x in wb.preview(events, "s1", path)["pending"]] == [True, True]


def test_one_failure_does_not_stop_the_others_and_nothing_is_retried(tmp_path, exe):
    call = Call(fail={"doc_mp"}, stderr="warning: slow\nerror: anchor text not found on page 3 (token pap_live_abcdefabcdef_secret)")
    got = wb.apply(session({"H0": "hl_1", "H1": "hl_web"}), "s1", ["H0", "H1"], tmp_path / wb.LEDGER, call)
    assert [(r["id"], r["status"], r.get("error")) for r in got] == [
        ("H0", "failed", "error: anchor text not found on page 3 (token [token])"), ("H1", "written", None)]
    assert got[0]["code"] == 1 and "pap_live" not in json.dumps(got) and "pap_live" not in (tmp_path / wb.LEDGER).read_text()
    assert len(call.calls) == 2
    assert [r["status"] for r in wb.apply(session({"H0": "hl_1"}), "s1", ["H0"], tmp_path / wb.LEDGER, Call())] == ["written"]


def test_a_timeout_is_a_failed_item(tmp_path, exe):
    def slow(args, **kw):
        raise subprocess.TimeoutExpired(args, kw["timeout"])

    assert wb.apply(session({"H0": "hl_1"}), "s1", ["H0"], tmp_path / wb.LEDGER, slow) == [
        {"id": "H0", "status": "failed", "error": f"timed out after {wb.TIMEOUT_S} s", "code": None}]


def test_writes_per_run_are_capped(tmp_path, exe, monkeypatch):
    monkeypatch.setattr(wb, "MAX_WRITES", 2)
    events = session({"H0": "hl_1", "H1": "hl_1", "H2": "hl_web"})
    got = wb.apply(events, "s1", ["H0", "H1", "H2"], tmp_path / wb.LEDGER, Call())
    assert [r["status"] for r in got] == ["written", "written", "failed"] and got[2]["error"] == "the cap of 2 writes per run is reached"


def test_unknown_or_skipped_ids_are_rejected_before_anything_runs(tmp_path, exe):
    call = Call()
    with pytest.raises(ValueError, match="unknown item ids: H9, H1"):
        wb.apply(session({"H0": "hl_1", "H1": "cnv_1"}), "s1", ["H0", "H9", "H1"], tmp_path / wb.LEDGER, call)
    assert call.calls == [] and not (tmp_path / wb.LEDGER).exists()


# ---- API ----

class Orch:
    """The server's view of a session: its log, whether it is done, and stop()."""

    def __init__(self, tmp_path, events, done=True):
        self.session, self.done, self.record, self.max_hyps = "sessions/s1", done, False, None
        self.log = Log(tmp_path / "events.jsonl")
        for e in events:
            self.log.emit(e["type"], **{k: v for k, v in e.items() if k not in ("t", "type")})

    def stop(self):
        pass


def client(orch):
    from fastapi.testclient import TestClient

    from server.app import create
    return TestClient(create(lambda: orch, checks={}, autostart=False))


def test_api_previews_and_applies_the_finished_live_session(tmp_path, exe, monkeypatch):
    call = Call()
    monkeypatch.setattr(subprocess, "run", call)
    o = Orch(tmp_path, session({"H0": "hl_1", "H1": "cnv_1"}))
    with client(o) as c:
        p = c.get("/papiers/writes").json()
        assert p["apply"] is True and p["session"] == "s1" and [x["id"] for x in p["pending"]] == ["H0"]
        assert p["skipped"] == [{"id": "H1", "verdict": "rejected", "reason": "cited a conversation, not a document passage"}]
        assert set(p["pending"][0]) >= {"id", "verdict", "doc", "pages", "passage", "note", "color", "written"}
        r = c.post("/papiers/writes", json={"session": "s1", "ids": ["H0"]})
        assert r.status_code == 200 and [x["status"] for x in r.json()["results"]] == ["written"]
        assert c.post("/papiers/writes", json={"session": "s1", "ids": ["H0"]}).json()["results"][0]["status"] == "already_written"
        assert c.post("/papiers/writes", json={"session": "s1", "ids": ["H7"]}).status_code == 400
        assert c.post("/papiers/writes", json={"session": "other", "ids": ["H0"]}).status_code == 404
    assert len(call.calls) == 1
    assert [e["results"] for e in of(o, "papiers_written")] == [[{"id": "H0", "status": "written"}], [{"id": "H0", "status": "already_written"}]]
    assert (tmp_path / wb.LEDGER).exists()


def test_api_refuses_apply_for_a_replay_a_mock_log_or_an_unfinished_session(tmp_path, exe, monkeypatch):
    call = Call()
    monkeypatch.setattr(subprocess, "run", call)
    with client(Orch(tmp_path / "a", session({"H0": "hl_1"}))) as c:
        r = c.get("/papiers/writes", params={"file": "/demo/replay.jsonl"}).json()
        assert r["apply"] is False and r["reason"].startswith("A replay is read-only") and r["pending"] == [] and len(r["skipped"]) == 8
        assert c.get("/papiers/writes", params={"file": "/pyproject.toml"}).status_code == 404
        assert c.get("/papiers/writes", params={"file": "/demo/../pyproject.toml"}).status_code == 404
    mock = [e | {"mock": True} for e in session({"H0": "hl_1"})]
    with client(Orch(tmp_path / "b", mock)) as c:
        assert c.get("/papiers/writes").json()["apply"] is False
        r = c.post("/papiers/writes", json={"session": "s1", "ids": ["H0"]})
        assert r.status_code == 403 and r.json()["detail"] == "Mock data never writes to Papiers."
    with client(Orch(tmp_path / "c", session({"H0": "hl_1"})[:-1], done=False)) as c:
        assert c.get("/papiers/writes").status_code == 409
        assert c.post("/papiers/writes", json={"session": "s1", "ids": ["H0"]}).status_code == 403
    assert call.calls == []


# ---- read-back ----

RWR = "[RWR] run 20261018-201500/H1 | rejected | std 1.3e-2 vs < 1.0e-3 | standard max error 1.3e-2 missed < 1.0e-3."


def test_rwr_notes_parse_and_anything_else_does_not():
    assert parse_note(RWR) == {"run": "20261018-201500/H1", "verdict": "rejected", "measured": "std 1.3e-2", "bound": "< 1.0e-3",
                               "reason": "standard max error 1.3e-2 missed < 1.0e-3."}
    (it,), _ = wb.plan(session({"H0": "hl_1"}), "s1")
    assert parse_note(it["note"])["measured"] == "1.54x, std 4.9e-4" and parse_note(it["note"])["bound"] == ">= 1.20x, < 1.0e-3"
    assert parse_note("This is why our FP16 accumulator fails at large K.") is None
    assert parse_note("[RWR] free text") is None and parse_note(None) is None


SEARCH = {"results": [{"kind": "document", "id": "doc_mp", "title": "MIXED PRECISION TRAINING", "source": {"family": "pdf"}, "evidence": [
    {"kind": "highlight", "id": "hl_rwr", "snippet": "FP16 accumulates into FP32"}, {"kind": "highlight", "id": "hl_me", "snippet": "FP16 accumulates"}]}]}
READ = {"items": [
    {"id": "hl_rwr", "ok": True, "content": "FP16 accumulates into FP32", "metadata": {"documentId": "doc_mp", "documentTitle": "MIXED PRECISION TRAINING",
                                                                                     "pageNumber": 3, "annotationText": RWR}},
    {"id": "hl_me", "ok": True, "content": "FP16 accumulates into FP32", "metadata": {"documentId": "doc_mp", "documentTitle": "MIXED PRECISION TRAINING",
                                                                                    "pageNumber": 3, "annotationText": "My own note."}}]}


def test_prior_verdicts_are_read_from_rwr_annotations_and_never_retrieved_as_context():
    cli = CLI({("search", "highlight", None): SEARCH, ("read", None, None): READ})
    lib = Library(cli)
    assert lib.prior_verdicts() == [{"id": "hl_rwr", "doc_id": "doc_mp", "document_title": "MIXED PRECISION TRAINING"} | parse_note(RWR)]
    assert ["search", "[RWR]", "--kind", "highlight", "--limit", "50"] in cli.calls
    got = lib.retrieve("accumulation precision FP16")
    assert [x["id"] for x in got] == ["hl_me"]
    assert (got[0]["doc_id"], got[0]["page"], got[0]["family"]) == ("doc_mp", 3, "pdf")
    # an [RWR] highlight that was not found at start is still dropped once its annotation is read
    assert [x["id"] for x in Library(CLI({("search", "highlight", None): SEARCH, ("read", None, None): READ})).retrieve("accumulation")] == ["hl_me"]


def test_prior_lookup_reads_at_most_the_capped_number_of_pages(monkeypatch):
    from agent import papiers
    monkeypatch.setattr(papiers, "PRIOR_READS", 1)
    listed = {"items": [{"id": f"hl_{i}", "kind": "highlight", "preview": "p"} for i in range(50)]}
    search = {"results": [{"kind": "document", "id": "doc", "evidence": [{"kind": "highlight", "id": f"hl_s{i}", "snippet": "s"} for i in range(50)]}]}
    cli = CLI({("search", "highlight", None): search, ("list", "highlight", None): listed, ("read", None, None): {"items": []}})
    assert Library(cli).prior_verdicts() == []
    reads = [c for c in cli.calls if c[0] == "read"]
    assert len(reads) == 1 and len([a for a in reads[0] if a.startswith("hl_")]) == 50


def test_the_evidence_check_rejects_rwr_notes():
    assert not citable({"id": "hl_rwr", "annotation": RWR})
    assert not citable({"id": "hl_x", "annotation": "plain"}, {"hl_x"})
    assert citable({"id": "hl_me", "annotation": "My own note."}) and citable({"id": "cnv_1"})


PRIOR = [{"id": "hl_rwr", "doc_id": "doc_mp", "document_title": "MIXED PRECISION TRAINING"} | parse_note(RWR)]
RWR_ITEM = {"id": "hl_rwr", "kind": "highlight", "title": "MIXED PRECISION TRAINING", "document_title": "MIXED PRECISION TRAINING",
            "excerpt": "FP16 accumulates into FP32", "annotation": RWR}
MINE = {"id": "hl_me", "kind": "highlight", "title": "Triton", "document_title": "Triton", "excerpt": "Tiles share memory."}


def test_prior_verdicts_reach_the_prompt_in_their_own_section_and_the_log(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, read_back=lambda: PRIOR, context=lambda q: [RWR_ITEM, MINE])
    o.run()
    (e,) = of(o, "prior_verdicts")
    assert e["count"] == 1 and e["items"] == PRIOR and "error" not in e
    assert [x["type"] for x in o.log.events].index("prior_verdicts") == 2  # right after session_started
    text = o.ask.calls[0][2][1]["content"]
    assert e["section"] in text and text.count("Prior verdicts from earlier sessions, read from Papiers") == 1
    assert "| rejected | std 1.3e-2 vs < 1.0e-3 |" in text
    assert "context, old note without a fingerprint" in text
    assert of(o, "hypothesis_proposed")[0]["hypothesis"]["papiers"] == "hl_me"  # the [RWR] item came first and was refused as evidence
    assert_contract(o.log.events)


def test_an_rwr_note_alone_is_never_cited(tmp_path):
    o = make(tmp_path, Asker(h(P, 2.0, "naive")), main_iters=1, read_back=list, context=lambda q: [RWR_ITEM])
    o.run()
    hy = of(o, "hypothesis_proposed")[0]["hypothesis"]
    assert "papiers" not in hy and "Context from" not in (hy.get("note") or "")
    assert all(pm.get("note") != "hl_rwr" for pm in (e["post_mortem"] for e in of(o, "post_mortem")))


def test_no_prior_verdicts_is_logged_as_zero_and_a_failed_lookup_says_so(tmp_path):
    o = make(tmp_path / "a", Asker(h(P, 2.0, "naive")), main_iters=1, read_back=list)
    o.run()
    assert [(e["count"], e["items"]) for e in of(o, "prior_verdicts")] == [(0, [])]
    assert "Prior verdicts" not in o.ask.calls[0][2][1]["content"]

    def down():
        raise subprocess.CalledProcessError(1, "papiers")

    o = make(tmp_path / "b", Asker(h(P, 2.0, "naive")), main_iters=1, read_back=down)
    o.run()
    assert [(e["count"], e["error"]) for e in of(o, "prior_verdicts")] == [(0, "Papiers unavailable (CalledProcessError)")]
    assert of(o, "verdict")[0]["verdict"] == "confirmed"
