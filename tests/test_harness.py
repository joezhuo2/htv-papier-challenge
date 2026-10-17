import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from harness import runner
from harness.metrics import CONTRACT, Metrics, tflops

TYPES_TS = Path(__file__).parents[1] / "src/contract/types.ts"


def ts_metrics_fields():
    block = re.search(r"interface Metrics \{(.*?)\}", TYPES_TS.read_text(), re.DOTALL).group(1)
    return dict(re.findall(r"^\s*(\w+): (\w+)", block, re.MULTILINE))


def test_contract_fields_match_types_ts():
    assert set(ts_metrics_fields()) == set(CONTRACT)


def test_metrics_json_roundtrip():
    m = Metrics(compiled_ok=True, correct_ok=True, latency_ms=0.64, tflops=26.6, max_error=4.9e-4, adv_error=5.2e-4, vram_mb=96.0,
                errors={"standard": 4.9e-4}, params={"block_m": 128})
    back = Metrics.from_dict(json.loads(json.dumps(m.dict())))
    assert back == m
    pytypes = {"number": float, "boolean": bool}
    for k, t in ts_metrics_fields().items():
        assert isinstance(back.contract()[k], pytypes[t]), k


def test_failed_metrics_defaults():
    c = Metrics(error="boom").contract()
    assert c["compiled_ok"] is False and c["correct_ok"] is False and c["latency_ms"] is None


def test_from_dict_ignores_unknown():
    assert Metrics.from_dict({"latency_ms": 1.0, "label": "x"}).latency_ms == 1.0


def test_tflops_formula():
    assert tflops(2048, 2048, 2048, 1.0) == pytest.approx(2 * 2048**3 / 1e-3 / 1e12)
    assert tflops(512, 512, 16384, 0.5) == pytest.approx(2 * 512 * 512 * 16384 / 0.5e-3 / 1e12)


SLOW = "import subprocess, time; subprocess.Popen(['sleep', '317.25']); print('ready', flush=True); time.sleep(300)"


def test_timeout_kills_process_group(tmp_path):
    m = runner.run({"target": "triton"}, timeout=1, cmd=[sys.executable, "-c", SLOW], runs=tmp_path)
    assert m.timed_out and m.error == "timeout"
    assert m.wall_s - m.startup_s < runner.TIMEOUT_S
    assert subprocess.run(["pgrep", "-f", "sleep 317.25"], capture_output=True, check=False).returncode == 1


def test_startup_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "STARTUP_S", 1)
    m = runner.run({"target": "triton"}, cmd=[sys.executable, "-c", "import time; time.sleep(300)"], runs=tmp_path)
    assert m.timed_out and m.error == "startup timeout" and m.wall_s < 5


def test_crash_reported(tmp_path):
    m = runner.run({"target": "triton"}, cmd=[sys.executable, "-c", "print('ready', flush=True); raise SystemExit(3)"], runs=tmp_path)
    assert not m.timed_out and not m.compiled_ok and m.error.startswith("exit 3")


def test_evidence_written(tmp_path):
    cmd = [sys.executable, "-c", "import sys, json; print('ready', flush=True); sys.stdin.read(); print(json.dumps({'compiled_ok': True, 'latency_ms': 1.5}))"]
    m = runner.run({"target": "triton"}, run_id="abc", cmd=cmd, runs=tmp_path)
    assert m.compiled_ok and m.latency_ms == 1.5
    assert {p.name for p in (tmp_path / "abc").iterdir()} == {"spec.json", "stdout.txt", "stderr.txt", "metrics.json"}
    assert json.loads((tmp_path / "abc/metrics.json").read_text())["run_id"] == "abc"
