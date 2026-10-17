"""scripts/results_table.py against the frozen demo replay, and docs/results.md matches it."""

import runpy
from pathlib import Path

ROOT = Path(__file__).parent.parent
table = runpy.run_path(str(ROOT / "scripts" / "results_table.py"))["table"]


def test_frozen_table():
    rows = table(ROOT / "demo" / "replay.jsonl").splitlines()[2:]
    assert [r.split(" | ")[0].strip("|* ") for r in rows] == [f"r{i}" for i in range(1, 9)]
    assert "**r5**" in rows[4] and rows[4].endswith("| inconclusive |")
    assert sum(r.endswith("| inconclusive |") for r in rows) == 5


def test_results_doc_is_current():
    assert table(ROOT / "demo" / "replay.jsonl") in (ROOT / "docs" / "results.md").read_text()
