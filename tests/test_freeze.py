"""scripts/freeze.sh, run on a copy inside a temporary repo root."""

import json
import shutil
import subprocess
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "freeze.sh"
FINAL = {"t": 90, "type": "final_report", "report": {"best": {"run": "r2", "hypothesis": "H1", "verdict": "inconclusive", "vs_naive": 2.6, "vs_eager": 0.97}}}
DONE = {"t": 125000, "type": "session_finished", "ledger": [], "best": {"run": "r2"}}


def repo(tmp_path, events, exports=None):
    (tmp_path / "scripts").mkdir(parents=True)
    shutil.copy(SCRIPT, tmp_path / "scripts")
    s = tmp_path / "runs" / "sessions" / "s1"
    (s / "exports").mkdir(parents=True)
    (s / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for name, text in (exports or {}).items():
        (s / "exports" / name).write_text(text)
    return tmp_path


def freeze(root, *args):
    return subprocess.run(["bash", str(root / "scripts" / "freeze.sh"), "s1", *args], check=False, capture_output=True, text=True)


def test_freezes_and_reports(tmp_path):
    root = repo(tmp_path, [{"t": 0, "type": "session_started"}, FINAL, DONE], {"ledger.md": "# ledger\n"})
    stale = root / "docs" / "examples" / "exports"
    stale.mkdir(parents=True)
    (stale / "old.md").write_text("old")
    r = freeze(root)
    assert r.returncode == 0, r.stderr
    assert (root / "demo" / "replay.jsonl").read_text().count("\n") == 3
    assert sorted(p.name for p in stale.iterdir()) == ["ledger.md"]
    assert "clean" in r.stdout and "events: 3" in r.stdout and "2m 05s" in r.stdout
    assert "best: r2" in r.stdout and "verdict: inconclusive" in r.stdout


def test_fallback_keeps_replay_and_exports(tmp_path):
    root = repo(tmp_path, [FINAL, DONE], {"ledger.md": "new"})
    (root / "demo").mkdir()
    (root / "demo" / "replay.jsonl").write_text("keep")
    r = freeze(root, "--as", "fallback.jsonl")
    assert r.returncode == 0, r.stderr
    assert (root / "demo" / "replay.jsonl").read_text() == "keep"
    assert (root / "demo" / "fallback.jsonl").exists()
    assert not (root / "docs" / "examples" / "exports").exists()


def test_rejects_unfinished_or_reportless(tmp_path):
    assert freeze(repo(tmp_path / "a", [DONE, FINAL])).returncode != 0
    assert freeze(repo(tmp_path / "b", [DONE])).returncode != 0
    assert not (tmp_path / "a" / "demo").exists()


def test_fails_on_secret(tmp_path):
    root = repo(tmp_path, [FINAL, DONE], {"note.md": "GEMINI_API_KEY=AIzaXXXX\n"})
    r = freeze(root)
    assert r.returncode != 0 and "clean" not in r.stdout
