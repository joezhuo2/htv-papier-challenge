import json

import pytest

from harness.kernel import NAIVE


def test_fake_llm_scripts(fake_llm):
    fake_llm.script("malformed_json", "valid_hypothesis", TimeoutError())
    with pytest.raises(json.JSONDecodeError):
        json.loads(fake_llm.complete([]))
    assert json.loads(fake_llm.complete([]))["params"]["block_m"] == 128
    with pytest.raises(TimeoutError):
        fake_llm.complete([])


def test_fake_engine_lookup(fake_engine):
    m = fake_engine.run(NAIVE)
    assert m.correct_ok and m.latency_ms == 1.891 and fake_engine.runs == [NAIVE]
