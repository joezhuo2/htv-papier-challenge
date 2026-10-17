"""Read-only Papiers context through the papiers CLI (Node 22). The CLI cannot write, findings are exported as files instead.

retrieve() follows the papiers-research skill shipped in the npm package: the user's own judgement (highlights, annotations,
notes) outranks everything else, then passages in annotated documents, then any passage, then conversations."""

import glob
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# lower ranks first: an annotated highlight carries the user's own words, a note is theirs entirely
RANK = {"annotated highlight": 0, "note": 1, "highlight": 2, "passage": 3, "conversation": 4}
STOP = {"a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "with"}


def binary():
    found = os.environ.get("PAPIERS_BIN") or shutil.which("papiers")
    if found:
        return found
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v2*/bin/papiers")))
    return nvm[-1] if nvm else None


def cli(args, timeout=30):
    """Runs papiers ARGS --json and returns the parsed envelope. A failure (exit 1 or 2, or a timeout) raises."""
    exe = binary()
    if not exe:
        raise FileNotFoundError("papiers CLI not found, install it with Node 22 or set PAPIERS_BIN")
    env = os.environ | {"PATH": f"{Path(exe).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    out = subprocess.run([exe, *args, "--json"], capture_output=True, text=True, timeout=timeout, env=env, check=True)
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


def words(text):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP and len(w) > 2}


def item(id, kind, title, text, annotation=None):
    kind = "highlight" if kind == "annotated highlight" else kind
    d = {"id": id, "kind": kind, "document_title": title, "title": title or id, "excerpt": text}
    if kind == "highlight":
        d["highlight_text"] = text
    return d | ({"annotation": annotation} if annotation else {})


class Library:
    """Per-session retrieval. run(args) -> envelope is the CLI, injectable for tests. The highlight and note lists are fetched once."""

    def __init__(self, run=cli):
        self.run = run
        self.marks = None

    def listed(self):
        if self.marks is None:
            self.marks = [(x["id"], x["kind"], (x.get("metadata") or {}).get("documentTitle") or x.get("title") or "", x.get("preview") or "")
                          for kind in ("highlight", "note") for x in self.run(["list", "--kind", kind, "--limit", "50"]).get("items", [])]
        return self.marks

    def annotate(self, ids):
        """id -> (full text, annotation) from one read: annotationText only comes back on a read."""
        if not ids:
            return {}
        env = self.run(["read", *ids, "--format", "plain", "--max-chars-per-item", "1200"])
        return {x["id"]: ((x.get("content") or "").strip(), (x.get("metadata") or {}).get("annotationText")) for x in env.get("items", []) if x.get("ok", True)}

    def retrieve(self, query, limit=2):
        """[{id, kind, document_title, title, excerpt, highlight_text?, annotation?}], the user's judgement first, [] when nothing matches."""
        with ThreadPoolExecutor(2) as pool:
            pages = list(pool.map(self.run, [["search", query, "--kind", k, "--limit", "5"] for k in ("highlight", "note")]))
        found = [(h, k) for page, k in zip(pages, ("highlight", "note")) for h in hits(page, k)]
        if not found:
            # the search index can miss a mark the user made, so match the listed highlights and notes on shared words
            q = words(query)
            found = [((i, t, p), k) for i, k, t, p in self.listed() if len(q & words(p)) >= 2]
        if found:
            seen, marks = set(), []
            for (i, t, p), k in found:
                if i not in seen:
                    seen.add(i)
                    marks.append((i, k, t, p))
            full = self.annotate([i for i, k, _, _ in marks if k == "highlight"][:limit * 2])
            ranked = []
            for i, k, t, p in marks:
                text, note = full.get(i, (p, None))
                ranked.append(("annotated highlight" if note else k, i, t, text or p, note))
            ranked.sort(key=lambda x: RANK[x[0]])
            return [item(i, k, t, text, note) for k, i, t, text, note in ranked[:limit]]
        for args, kind in ((["--kind", "passage", "--filter", "annotated.eq=true"], "passage"), (["--kind", "passage"], "passage"),
                           (["--kind", "conversation"], "conversation")):
            got = hits(self.run(["search", query, *args, "--limit", str(limit)]), kind)
            if got:
                return [item(i, kind, t, p) for i, t, p in got[:limit]]
        return []

    def counts(self):
        """Highlights, notes and documents the CLI can see, '50+' when a page is full."""
        out = {}
        for kind in ("highlight", "note", "document"):
            env = self.run(["list", "--kind", kind, "--limit", "50"])
            n = len(env.get("items", []))
            out[kind] = f"{n}+" if (env.get("page") or {}).get("nextCursor") else n
        return out
