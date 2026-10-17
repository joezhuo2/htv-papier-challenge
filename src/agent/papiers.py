"""Read-only Papiers context through the papiers CLI (Node 22). The CLI cannot write, findings are exported as files instead."""

import glob
import json
import os
import shutil
import subprocess
from pathlib import Path


def binary():
    found = os.environ.get("PAPIERS_BIN") or shutil.which("papiers")
    if found:
        return found
    nvm = sorted(glob.glob(os.path.expanduser("~/.nvm/versions/node/v2*/bin/papiers")))
    return nvm[-1] if nvm else None


def results(stdout):
    """papiers search --json output to [{id, kind, title, excerpt}]."""
    d = json.loads(stdout)
    return [{"id": r["id"], "kind": r.get("kind", ""), "title": r.get("title") or r["id"], "excerpt": (r.get("snippet") or "").strip()}
            for r in d.get("results", [])]


def search(query, limit=3, timeout=30):
    """Returns matching notes, highlights, passages or conversations from the linked account."""
    exe = binary()
    if not exe:
        raise FileNotFoundError("papiers CLI not found, install it with Node 22 or set PAPIERS_BIN")
    env = os.environ | {"PATH": f"{Path(exe).parent}{os.pathsep}{os.environ.get('PATH', '')}"}
    out = subprocess.run([exe, "search", query, "--limit", str(limit), "--json"], capture_output=True, text=True, timeout=timeout, env=env, check=True)
    return results(out.stdout)
