"""Golden session: a fixed fake session's event types and payload keys must match tests/fixtures/golden_session.jsonl.
Regenerate deliberately with GOLDEN_UPDATE=1 uv run pytest tests/test_golden.py, then check the UI golden test in src/ui/test."""

import json
import os
from pathlib import Path

from test_agent import Asker, P, h, make, run_with
from test_prior_guard import prior

FIXTURE = Path(__file__).parent / "fixtures" / "golden_session.jsonl"


def shape(e, depth=0):
    if isinstance(e, dict):
        return {k: shape(v, depth + 1) for k, v in sorted(e.items()) if k != "t"} if depth < 2 else sorted(e)
    return type(e).__name__


def session(tmp_path):
    # an earlier session's verdict for stages 4 under the Precision bound: the Precision branch's second proposal is refused by it
    earlier = prior("confirmed", change={"stages": 4}, base=P, bound={"standard": 1e-3, "adversarial": 1e-3})
    o = make(tmp_path, Asker(h(P, 2.0, "naive"), h(P | {"acc": "fp16"}), h(P | {"stages": 4}, 1.05), h(P | {"gelu": "approx"}, 1.05)), main_iters=1,
             branch_iters=1, read_back=lambda: [earlier],
             context=lambda q: [{"id": "cnv_1", "title": "Triton tutorial", "excerpt": "Accumulate in FP32."}])
    run_with(o, [(1, "speed", "Push throughput harder."), (2, "precision", "Max error under 1e-3 on both batches.")])
    return o.log.events


def test_golden_session_matches_fixture(tmp_path):
    events = session(tmp_path)
    if os.environ.get("GOLDEN_UPDATE") == "1":
        FIXTURE.write_text("".join(json.dumps(e) + "\n" for e in events))
    golden = [json.loads(line) for line in FIXTURE.read_text().splitlines()]
    assert [e["type"] for e in events] == [e["type"] for e in golden]
    assert [shape(e) for e in events] == [shape(e) for e in golden]
