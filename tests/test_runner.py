import pytest

from harness import runner
from harness.kernel import NAIVE

pytestmark = pytest.mark.gpu


def test_naive_passes(tmp_path):
    m = runner.run({"target": "triton", "params": NAIVE.dict()}, runs=tmp_path)
    assert m.compiled_ok and m.correct_ok, m


def test_timeout_kills(tmp_path):
    m = runner.run({"target": "triton", "params": NAIVE.dict()}, timeout=0.01, runs=tmp_path)
    assert m.timed_out and m.error == "timeout"


def test_invalid_params_reported(tmp_path):
    m = runner.run({"target": "triton", "params": NAIVE.dict() | {"block_m": 24}}, runs=tmp_path)
    assert not m.compiled_ok and "block_m" in m.error
