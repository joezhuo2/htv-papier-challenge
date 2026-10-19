"""Final verdicts written back to Papiers as highlights on the passage each hypothesis cited, only after a run finished and only for
the items a human approved. The document id, page and passage text all come from what the agent read during the run, as logged in its
papiers_context events; nothing here composes a quote. The Markdown export in agent.export is separate and unchanged."""

import json
import re
import subprocess
import time

from agent import papiers

COLORS = {"confirmed": "#5fb85f", "rejected": "#e05d5d", "inconclusive": "#ffd400", "provisional": "#ffd400"}
MAX_WRITES = 20
ANCHOR_WORDS = 8
REASON_CHARS = 80
TIMEOUT_S = 30
LEDGER = "papiers-writes.json"
WEB = {"web_clip", "tweet"}  # source families without pages
KINDS = {"hl": "highlight", "pas": "passage", "psg": "passage", "nte": "note", "cnv": "conversation"}
LABELS = {"standard max error": "std", "adversarial max error": "adv", "compiles and runs": "runs"}
TOKEN = re.compile(r"pap_(?:live|test)_[A-Za-z0-9_]+")


def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def short(text, n=REASON_CHARS):
    text = " ".join(text.replace("|", "/").split())
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0].rstrip(",;:.")


def figures(checks):
    return [(c["observed"].split()[0] if c["name"].startswith("speedup") else f"{LABELS[c['name']]} {c['observed']}", c["predicted"]) for c in checks]


def measured(v, word):
    """(measured, bound): the failed check for a rejection, the speedup for an inconclusive verdict, speedup and standard error otherwise."""
    checks = v["checks"]
    speed = [c for c in checks if c["name"].startswith("speedup")]
    picked = ([c for c in checks if c["pass"] is False][:1] if word == "rejected" else speed or checks[:1] if word == "inconclusive"
              else speed + [c for c in checks if c["name"] == "standard max error"])
    got = figures(picked)
    return ", ".join(m for m, _ in got), ", ".join(b for _, b in got)


def note(run, word, v):
    m, b = measured(v, word)
    return f"[RWR] run {run} | {word} | {m} vs {b} | {short(v['reason'])}"


def build(hid, v, cited, notes, session):
    """(item, None) for a writable verdict, (None, reason) for a skipped one."""
    if not cited:
        return None, "no Papiers passage cited"
    n = notes.get(cited)
    if not n:
        return None, f"the cited item {cited} is not in the run log"
    kind = n.get("kind") or KINDS.get(cited.split("_")[0], "item")
    if kind not in ("highlight", "passage"):
        return None, f"cited a {kind}, not a document passage"
    if not n.get("doc_id"):
        return None, "no document id in the stored data"
    text = n.get("anchor")
    if not text or "[[src:" in text:
        return None, "no exact passage text in the stored data"
    if n.get("truncated"):
        return None, "the stored passage text is truncated"
    if n.get("page") is None and n.get("family") not in WEB:
        return None, "no page in the stored data"
    w = text.split()
    word = "provisional" if v["provisional"] else v["verdict"]
    run = f"{session}/{hid}"
    start = " ".join(w[:ANCHOR_WORDS])
    return {"id": hid, "verdict": word, "doc": n.get("document_title") or n["doc_id"], "doc_id": n["doc_id"],
            "pages": None if n.get("page") is None else str(n["page"]), "passage": text, "start": start,
            "end": " ".join(w[-ANCHOR_WORDS:]) if len(w) > 2 * ANCHOR_WORDS else None, "note": note(run, word, v), "color": COLORS[word],
            "key": f"{n['doc_id']}|{start}|{run}"}, None


def plan(events, session):
    """(pending, skipped) from the final verdict of every judged hypothesis."""
    notes = {e["note"]["id"]: e["note"] for e in events if e["type"] == "papiers_context" and "note" in e}
    cited = {e["hypothesis"]["id"]: e["hypothesis"].get("papiers") for e in events if e["type"] == "hypothesis_proposed"}
    final = {e["hypothesis"]: e for e in events if e["type"] == "verdict"}
    pending, skipped = [], []
    for hid, v in final.items():
        it, why = build(hid, v, cited.get(hid), notes, session)
        if it:
            pending.append(it)
        else:
            skipped.append({"id": hid, "verdict": "provisional" if v["provisional"] else v["verdict"], "reason": why})
    return pending, skipped


def ledger(path):
    return json.loads(path.read_text()) if path.exists() else {"written": {}, "attempts": []}


def preview(events, session, path=None):
    """The dry run: what would be written, what is skipped and why. path is the run's ledger, when it has one."""
    pending, skipped = plan(events, session)
    done = ledger(path)["written"] if path else {}
    return {"session": session, "finished": any(e["type"] == "session_finished" for e in events),
            "pending": [it | {"written": it["key"] in done} for it in pending], "skipped": skipped}


def args(exe, it):
    out = [exe, "highlight", it["doc_id"]]
    if it["pages"]:
        out += ["--pages", it["pages"]]
    out += ["--start", it["start"]]
    if it["end"]:
        out += ["--end", it["end"]]
    return out + ["--note", it["note"], "--color", it["color"]]


def scrub(text):
    return TOKEN.sub("[token]", text or "")


def write(exe, it, call):
    try:
        p = call(args(exe, it), capture_output=True, text=True, timeout=TIMEOUT_S, env=papiers.env(exe), check=False)
    except subprocess.TimeoutExpired:
        return {"status": "failed", "error": f"timed out after {TIMEOUT_S} s", "code": None}
    out = {"code": p.returncode, "stdout": scrub(p.stdout)[-500:], "stderr": scrub(p.stderr)[-500:]}
    if p.returncode == 0:
        return out | {"status": "written"}
    last = (out["stderr"].strip().splitlines() or [f"exit code {p.returncode}"])[-1]
    return out | {"status": "failed", "error": short(last, 160)}


def apply(events, session, ids, path, call=None):
    """Writes the approved items one by one and records each result in the ledger at path, so applying twice never writes twice.
    One failure does not stop the others, nothing is retried. Raises ValueError for ids that are not pending items."""
    pending, _ = plan(events, session)
    by = {it["id"]: it for it in pending}
    unknown = [i for i in ids if i not in by]
    if unknown:
        raise ValueError(f"unknown item ids: {', '.join(map(str, unknown))}")
    exe = papiers.binary()
    if not exe:
        raise FileNotFoundError("papiers CLI not found, install it with Node 22 or set PAPIERS_BIN")
    book = ledger(path)
    out = []
    for i in dict.fromkeys(ids):
        it = by[i]
        if it["key"] in book["written"]:
            r = {"id": i, "status": "already_written"}
        elif len(book["written"]) >= MAX_WRITES:
            r = {"id": i, "status": "failed", "error": f"the cap of {MAX_WRITES} writes per run is reached"}
        else:
            r = {"id": i} | write(exe, it, call or subprocess.run)
            if r["status"] == "written":
                book["written"][it["key"]] = {"id": i, "doc_id": it["doc_id"], "pages": it["pages"], "note": it["note"], "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        book["attempts"].append(r | {"at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        path.write_text(json.dumps(book, indent=1))
        out.append(r)
    return out
