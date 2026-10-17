import json

import pytest

from agent.events import Log
from agent.orchestrator import Orchestrator
from harness import noise, runner
from harness.data import BOUND
from harness.kernel import NAIVE
from agent.verdict import judge
from test_agent import P, Asker, h, of, run_with, assert_contract

pytestmark = pytest.mark.gpu


def test_sweep_target_shows_fp16_error_over_the_bound_and_growing_with_k(tmp_path):
    m = runner.run({"target": "sweep", "params": P | {"acc": "fp16"}, "ks": [256, 1024, 4096]}, runs=tmp_path)
    errs = [m.errors[f"k{k}"] for k in (256, 1024, 4096)]
    assert all(e > BOUND for e in errs) and errs[0] < errs[-1], errs
    safe = runner.run({"target": "sweep", "params": P, "ks": [256, 4096]}, runs=tmp_path)
    assert all(v < BOUND for v in safe.errors.values()), safe.errors


def test_fp16_error_is_deterministic_passes_speed_tolerance_three_times_then_fails_precision(tmp_path):
    fp16 = P | {"acc": "fp16"}
    refs = {"parent": {"target": "triton", "params": P}, "naive": {"target": "triton", "params": NAIVE.dict()}, "eager": {"target": "eager"}}
    pred = {"min_speedup": 1.2, "vs": "parent_best", "max_error": 2e-2}
    runs = [runner.run({"target": "triton", "params": fp16, "refs": refs}, runs=tmp_path) for _ in range(3)]
    print(f"fp16 standard {runs[0].max_error:.3e} (margin {1 - runs[0].max_error / 2e-2:.0%} under 2e-2), adversarial {runs[0].adv_error:.3e}, "
          f"speedup vs fp32 parent {[(round(m.speedups['parent'], 3), round(m.spreads['parent'], 4)) for m in runs]}")
    assert len({(m.max_error, m.adv_error) for m in runs}) == 1, "fixed seed inputs must give identical errors"
    for m in runs:
        j = judge(pred, m, m.speedups["parent"], max(noise.band(), m.spreads["parent"]), 2e-2, None)
        assert (j.verdict, j.provisional) == ("confirmed", True), j.reason
    m = runs[0]
    j = judge(pred, m, m.speedups["parent"], max(noise.band(), m.spreads["parent"]), 1e-3, 1e-3)
    assert j.verdict == "rejected" and [c.passed for c in j.checks][1:] == [False, False]


def test_real_failure_path_with_scripted_hypotheses(tmp_path):
    engine = lambda spec, run_id: runner.run(spec, run_id=run_id, runs=tmp_path)
    o = Orchestrator(engine, Asker(h(P, 2.0, "naive"), h(P | {"acc": "fp16"}, 1.2), default="no json"), Log(tmp_path / "events.jsonl"),
                     "s", "fast", "strong", noise.band(), main_iters=1, branch_iters=1, idle_s=0.5)
    run_with(o, [(1, "speed", "Push throughput harder."), (2, "precision", "Max error under 1e-3 on both batches.")])
    h1 = [e for e in of(o, "verdict") if e["hypothesis"] == "H1"]
    ev = o.log.events
    flip = ev.index(h1[1])
    directive = next(i for i, e in enumerate(ev) if e["type"] == "directive_added" and e["directive"]["lens"] == "precision")
    assert not [e for e in ev[directive:flip] if e["type"] == "run_started"], "the flip must come from logged metrics, without a new run"
    assert [(e["verdict"], e["provisional"]) for e in h1] == [("confirmed", True), ("rejected", False)]
    assert h1[0]["checks"][0]["pass"] is True, "FP16 accumulation should be faster on this GPU"
    sweep = of(o, "post_mortem")[0]["post_mortem"]["sweep"]
    assert len(sweep) == 5 and all(s["adv_error"] > BOUND for s in sweep)
    assert_contract(o.log.events)
