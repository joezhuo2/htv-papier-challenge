import json
import statistics
import subprocess
import sys
import time

import pytest

from harness import data, experiment, kernel, runner
from harness.kernel import NAIVE, Params
from harness.noise import BAND_FILE, CONFIG

pytestmark = pytest.mark.gpu


def test_naive_all_cases():
    m = experiment.measure({"target": "triton", "params": NAIVE.dict()})
    assert m.compiled_ok and m.correct_ok
    assert set(m.errors) == {name for name, *_ in data.CASES}
    assert all(e < data.BOUND for e in m.errors.values()), m.errors
    assert m.adv_error < data.BOUND


def test_wrong_bias_fails(monkeypatch):
    launch = kernel.launch
    monkeypatch.setattr(kernel, "launch", lambda p, x, w, b: launch(p, x, w, b + 0.01))
    m = experiment.measure({"target": "triton", "params": NAIVE.dict()})
    assert m.compiled_ok and not m.correct_ok


def test_metrics_types():
    m = experiment.measure({"target": "triton", "params": NAIVE.dict()})
    c = m.contract()
    assert all(isinstance(c[k], float) for k in ("latency_ms", "tflops", "max_error", "adv_error", "vram_mb"))
    assert all(isinstance(c[k], bool) for k in ("compiled_ok", "correct_ok"))


def test_register_and_spill_counts_are_recorded():
    m = experiment.measure({"target": "triton", "params": Params(256, 256, 64, 8, 2).dict()})
    assert m.n_regs > 0 and m.n_spills > 0, (m.n_regs, m.n_spills)
    assert experiment.measure({"target": "eager"}).n_regs is None


def test_noise_band(tmp_path):
    lat = [runner.run({"target": "triton", "params": CONFIG.dict()}, runs=tmp_path).latency_ms for _ in range(10)]
    med = statistics.median(lat)
    cv = statistics.stdev(lat) / statistics.mean(lat)
    max_dev = max(abs(v - med) for v in lat) / med
    band = {"params": CONFIG.dict(), "n": len(lat), "latencies_ms": lat, "median_ms": med, "cv": cv, "max_dev": max_dev,
            "band": max(3 * cv, max_dev)}
    BAND_FILE.write_text(json.dumps(band, indent=2) + "\n")
    print(f"\nnoise: median {med:.4f} ms, cv {cv:.2%}, spread {min(lat):.4f} to {max(lat):.4f} ms, band +/- {band['band']:.2%}")
    assert cv < 0.10


HANG = ("import time, torch; x = torch.empty(512 * 2**20, dtype=torch.uint8, device='cuda'); "
        "print('ready', flush=True); time.sleep(600)  # hang-marker-8841")


def gpu_used_mb():
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
    return int(out.stdout.split()[0])


def test_hang_killed(tmp_path):
    before = gpu_used_mb()
    m = runner.run({"target": "triton"}, cmd=[sys.executable, "-c", HANG], runs=tmp_path)
    assert m.timed_out and m.error == "timeout"
    assert m.wall_s - m.startup_s < runner.TIMEOUT_S + 2
    assert subprocess.run(["pgrep", "-f", "hang-marker-8841"], capture_output=True, check=False).returncode == 1
    time.sleep(1)
    assert gpu_used_mb() < before + 256, "the killed child's 512 MB allocation was not released"


SWEEP = [Params(block_m=64, block_n=64, block_k=32, warps=4, stages=3), Params(block_m=128, block_n=64, block_k=32, warps=4, stages=2),
         Params(block_m=128, block_n=128, block_k=32, warps=4, stages=2), Params(block_m=128, block_n=128, block_k=32, warps=4, stages=3)]


def test_headroom(tmp_path):
    eager = runner.run({"target": "eager"}, runs=tmp_path)
    naive = runner.run({"target": "triton", "params": NAIVE.dict()}, runs=tmp_path)
    best = min((runner.run({"target": "triton", "params": p.dict()}, runs=tmp_path) for p in SWEEP), key=lambda m: m.latency_ms)
    vs_naive, vs_eager = naive.latency_ms / best.latency_ms, eager.latency_ms / best.latency_ms
    print(f"\nheadroom: best {best.params} {best.latency_ms:.4f} ms, {vs_naive:.2f}x vs naive, {vs_eager:.2f}x vs eager")
    assert best.correct_ok and vs_naive > 2


@pytest.mark.slow
def test_thermal_drift():
    x, w, b = data.make(*data.CASES[0][1:])
    fn = lambda: kernel.run(CONFIG, x, w, b)
    t0, samples = time.monotonic(), []
    while time.monotonic() - t0 < 180:
        samples.append((time.monotonic() - t0, experiment.time_ms(fn)[0]))
    first = statistics.median(v for t, v in samples if t < 30)
    last = statistics.median(v for t, v in samples if t > 150)
    drift = (last - first) / first
    print(f"\ndrift: {len(samples)} samples, first 30 s median {first:.4f} ms, last 30 s median {last:.4f} ms, drift {drift:+.2%}")
    assert abs(drift) < 0.10
