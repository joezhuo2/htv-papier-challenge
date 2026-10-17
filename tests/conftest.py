import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

from harness.metrics import Metrics

FIXTURES = Path(__file__).parent / "fixtures"


def pytest_collection_modifyitems(config, items):
    if "slow" in (config.option.markexpr or ""):
        return
    keep = [i for i in items if "slow" not in i.keywords]
    config.hook.pytest_deselected(items=[i for i in items if "slow" in i.keywords])
    items[:] = keep


class RateLimited(Exception):
    status_code = 429


class FakeLLM:
    """Returns scripted responses in order. A script entry is a fixture name, a raw string, or an exception instance to raise."""

    def __init__(self, responses):
        self.responses = responses
        self.queue = []
        self.calls = []

    def script(self, *entries):
        self.queue.extend(entries)
        return self

    def complete(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        if not self.queue:
            raise AssertionError("FakeLLM ran out of scripted responses")
        entry = self.queue.pop(0)
        if isinstance(entry, BaseException):
            raise entry
        return self.responses.get(entry, entry)


class FakeEngine:
    """Returns scripted Metrics keyed by params, and records every run."""

    def __init__(self, table):
        self.table = {self.key(row["params"]): row["metrics"] for row in table}
        self.runs = []

    @staticmethod
    def key(params):
        return tuple(sorted((asdict(params) if is_dataclass(params) else params).items()))

    def set(self, params, **metrics):
        self.table[self.key(params)] = metrics

    def run(self, params):
        self.runs.append(params)
        p = asdict(params) if is_dataclass(params) else params
        return Metrics.from_dict(self.table[self.key(params)] | {"params": p, "run_id": f"fake{len(self.runs)}"})


@pytest.fixture
def fake_llm():
    return FakeLLM(json.loads((FIXTURES / "llm_responses.json").read_text()))


@pytest.fixture
def fake_engine():
    return FakeEngine(json.loads((FIXTURES / "metrics.json").read_text()))
