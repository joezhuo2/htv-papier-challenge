"""Papiers context through the papiers CLI (Node 22). Reads only: verdicts are written back by agent.writeback, on a human's approval.

retrieve() follows the papiers-research skill shipped in the npm package: the user's own judgement (highlights, annotations,
notes) outranks everything else, then passages in annotated documents, then any passage, then conversations. Tile, stage, warp and shared
memory queries go from highlights and notes straight to conversations."""

import glob
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from agent import fingerprint

# lower ranks first: an annotated highlight carries the user's own words, a note is theirs entirely
RANK = {"annotated highlight": 0, "note": 1, "highlight": 2, "passage": 3, "conversation": 4}
TEX = re.compile(r"\$\$?(.+?)\$\$?|\\\((.+?)\\\)|\\\[(.+?)\\\]", re.DOTALL)
TILING = {"tile", "tiles", "stage", "stages", "warp", "warps", "shared", "memory"}
# the note prefix of a verdict written by an earlier session: context for the proposer, never evidence for a new claim
PRIOR = "[RWR]"
PRIOR_READS = 2  # read calls at session start, of up to 50 highlights each
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "with"}


def binary():
    found = os.environ.get("PAPIERS_BIN") or shutil.which("papiers")
    if found:
        return found
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v2*/bin/papiers")))
    return nvm[-1] if nvm else None


def env(exe):
    """The CLI runs under Node from its own install, and finds PAPIERS_TOKEN in the inherited environment."""
    return os.environ | {"PATH": f"{Path(exe).parent}{os.pathsep}{os.environ.get('PATH', '')}"}


def cli(args, timeout=30):
    """Runs papiers ARGS --json and returns the parsed envelope. A failure (exit 1 or 2, or a timeout) raises."""
    exe = binary()
    if not exe:
        raise FileNotFoundError("papiers CLI not found, install it with Node 22 or set PAPIERS_BIN")
    out = subprocess.run([exe, *args, "--json"], capture_output=True, text=True, timeout=timeout, env=env(exe), check=True)
    return json.loads(out.stdout)


def results(stdout):
    """papiers search --json output to [{id, kind, title, excerpt}]."""
    d = json.loads(stdout)
    return [{"id": r["id"], "kind": r.get("kind", ""), "title": r.get("title") or r["id"], "excerpt": (r.get("snippet") or "").strip()}
            for r in d.get("results", [])]


def search(query, limit=3, timeout=30):
    """Returns matching notes, highlights, passages or conversations from the linked account."""
    return results(json.dumps(cli(["search", query, "--limit", str(limit)], timeout)))


def hits(envelope, kind):
    """Flattens a search page to (id, document title, snippet) for one evidence kind. Documents nest their highlight and passage
    hits under evidence[], notes and conversations are standalone results."""
    out = []
    for r in envelope.get("results", []):
        if r.get("kind") == kind:
            out.append((r["id"], r.get("title") or "", (r.get("snippet") or "").strip()))
        for e in r.get("evidence") or []:
            if e.get("kind") == kind:
                out.append((e["id"], r.get("title") or "", (e.get("snippet") or "").strip()))
    return out


def parents(envelope):
    """evidence id -> (document id, source family) for the hits a search nests under their document."""
    return {e["id"]: (r["id"], (r.get("source") or {}).get("family")) for r in envelope.get("results", []) if r.get("kind") == "document"
            for e in r.get("evidence") or []}


def parse_note(text):
    """An [RWR] note to {run, verdict, measured, bound, reason}, plus fp (see agent.fingerprint) when it carries one. None for anything else.
    The reason never holds a pipe, the writer replaces them."""
    head, verdict, figures, reason, *rest = (text or "").strip().split(" | ") + [None] * 4
    run, scored = re.fullmatch(r"\[RWR\] run (\S+)", head), figures and re.fullmatch(r"(.+?) vs (.+)", figures, re.DOTALL)
    if not (run and scored and reason is not None and re.fullmatch(r"\w+", verdict or "")):
        return None
    fp = fingerprint.parse_tail([s for s in rest if s])
    return {"run": run[1], "verdict": verdict, "measured": scored[1], "bound": scored[2], "reason": reason} | ({"fp": fp} if fp else {})


def citable(n, prior=()):
    """The claim-evidence check: an [RWR] highlight records an earlier verdict, so it never backs a new claim."""
    return not (n.get("annotation") or "").startswith(PRIOR) and n["id"] not in prior


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def untex(text):
    """Drops LaTeX math delimiters ($x$, $$x$$, \\(x\\), \\[x\\]) and keeps their content, marks are shown as plain text."""
    return TEX.sub(lambda m: next(g for g in m.groups() if g is not None), text).replace("$", "").strip() if text else text


def item(id, kind, title, text, annotation=None, src=None):
    """src is where the passage sits, for a write-back: doc_id, page, family, the raw unstripped text as anchor, and truncated."""
    kind = "highlight" if kind == "annotated highlight" else kind
    annotation, text = untex(annotation), untex(text)
    d = {"id": id, "kind": kind, "document_title": title, "title": title or id, "excerpt": text}
    if kind == "highlight":
        d["highlight_text"] = text
    return d | ({"annotation": annotation} if annotation else {}) | {k: v for k, v in (src or {}).items() if v is not None}


class Library:
    """Per-session retrieval. run(args) -> envelope is the CLI, injectable for tests. The highlight and note lists are fetched once."""

    def __init__(self, run=cli):
        self.run = run
        self.marks = None
        self.docs = None
        self.parent = {}  # evidence id -> (document id, source family), from search pages
        self.prior = set()  # ids of the [RWR] highlights found at session start

    def document(self, text):
        """The library document a conversation is about: the title sharing the most words with its text, at least two, else None.
        The CLI does not link a conversation to its document, and its own title is a segment name like "Conversation overview"."""
        if self.docs is None:
            self.docs = [x.get("title") or "" for x in self.run(["list", "--kind", "document", "--limit", "50"]).get("items", [])]
        w = words(text)
        score, title = max(((len(w & words(t)), t) for t in self.docs), default=(0, None))
        return title if score >= 2 else None

    def listed(self):
        if self.marks is None:
            self.marks = [(x["id"], x["kind"], (x.get("metadata") or {}).get("documentTitle") or x.get("title") or "", x.get("preview") or "")
                          for kind in ("highlight", "note") for x in self.run(["list", "--kind", kind, "--limit", "50"]).get("items", [])]
        return self.marks

    def annotate(self, ids):
        """id -> (full text, annotation, src) from one read: annotationText, documentId and pageNumber only come back on a read."""
        if not ids:
            return {}
        out = {}
        for x in self.run(["read", *ids, "--format", "plain", "--max-chars-per-item", "1200"]).get("items", []):
            if not x.get("ok", True):
                continue
            meta, text = x.get("metadata") or {}, x.get("content") or ""
            doc, family = self.parent.get(x["id"], (None, None))
            src = {"doc_id": meta.get("documentId") or doc, "page": meta.get("pageNumber"), "family": family, "anchor": text.strip() or None,
                   "truncated": True if x.get("truncated") else None}
            out[x["id"]] = (text.strip(), meta.get("annotationText"), src)
        return out

    def located(self, id):
        doc, family = self.parent.get(id, (None, None))
        return {"doc_id": doc, "family": family}

    def retrieve(self, query, limit=2):
        """[{id, kind, document_title, title, excerpt, highlight_text?, annotation?}], the user's judgement first, [] when nothing matches."""
        with ThreadPoolExecutor(2) as pool:
            pages = list(pool.map(self.run, [["search", query, "--kind", k, "--limit", "5"] for k in ("highlight", "note")]))
        for page in pages:
            self.parent |= parents(page)
        found = [(h, k) for page, k in zip(pages, ("highlight", "note")) for h in hits(page, k)]
        q = words(query)
        # tiling queries rarely share two words with a mark and passages about tiling are generic, so any listed mark that mentions tiles,
        # stages, warps or shared memory is a candidate and the only fallback is a conversation
        tiled = bool(q & TILING)
        # the search index can miss a mark the user made, so listed highlights and notes sharing words with the query are candidates too
        listed = [((i, t, p), k) for i, k, t, p in self.listed() if (words(p) & TILING if tiled else len(q & words(p)) >= 2)]
        found = found or listed
        if found:
            ranked = self.rank(found)
            if not any(x[4] for x in ranked) and listed:
                # no annotated highlight among the hits: an annotated listed one that matches the query is preferred
                ranked = self.rank(found + listed)
            return [item(i, k, t, text, note, src) for k, i, t, text, note, src in ranked[:limit]]
        passages = [] if tiled else [(["--kind", "passage", "--filter", "annotated.eq=true"], "passage"), (["--kind", "passage"], "passage")]
        for args, kind in [*passages, (["--kind", "conversation"], "conversation")]:
            page = self.run(["search", query, *args, "--limit", str(limit)])
            self.parent |= parents(page)
            got = hits(page, kind)[:limit]
            if got and kind == "passage":
                full = self.annotate([i for i, _, _ in got])
                return [item(i, kind, t, p, None, full[i][2] if i in full else self.located(i)) for i, t, p in got]
            if got:
                return [item(i, kind, self.document(p) or t, p) for i, t, p in got]
        return []

    def rank(self, found):
        """[(kind, id, title, text, annotation, src)], annotated highlights first, then notes, then plain highlights, in search order within
        each. [RWR] highlights are dropped: they are prior verdicts, read separately by prior_verdicts."""
        seen, marks = set(self.prior), []
        for (i, t, p), k in found:
            if i not in seen:
                seen.add(i)
                marks.append((i, k, t, p))
        full = self.annotate([i for i, k, _, _ in marks if k == "highlight"][:10])
        ranked = []
        for i, k, t, p in marks:
            text, note, src = full.get(i, (p, None, self.located(i)))
            if not (note or "").startswith(PRIOR):
                ranked.append(("annotated highlight" if note else k, i, t, text or p, note, src))
        return sorted(ranked, key=lambda x: RANK[x[0]])

    def prior_verdicts(self):
        """[{id, doc_id, document_title, run, verdict, measured, bound, reason}] from the [RWR] highlights earlier sessions wrote. The CLI does
        not say whether search matches annotation text, so the candidates are a search for the prefix plus the listed highlights, and only a
        read (at most PRIOR_READS of them) shows the annotation that decides."""
        found = hits(self.run(["search", PRIOR, "--kind", "highlight", "--limit", "50"]), "highlight")
        ids = list(dict.fromkeys([i for i, _, _ in found] + [i for i, k, _, _ in self.listed() if k == "highlight"]))
        out = []
        for n in range(0, min(len(ids), 50 * PRIOR_READS), 50):
            for x in self.run(["read", *ids[n:n + 50], "--format", "plain", "--max-chars-per-item", "1200"]).get("items", []):
                meta = x.get("metadata") or {}
                p = x.get("ok", True) and parse_note(meta.get("annotationText"))
                if p:
                    out.append({"id": x["id"], "doc_id": meta.get("documentId"), "document_title": meta.get("documentTitle")} | p)
        self.prior = {p["id"] for p in out}
        return out

    def counts(self):
        """Highlights, notes and documents the CLI can see, '50+' when a page is full."""
        out = {}
        for kind in ("highlight", "note", "document"):
            env = self.run(["list", "--kind", kind, "--limit", "50"])
            n = len(env.get("items", []))
            out[kind] = f"{n}+" if (env.get("page") or {}).get("nextCursor") else n
        return out
