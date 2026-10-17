"""Child process entry point: measures one target and prints one JSON line. Run via harness.runner."""

import json
import statistics
import sys

import torch

from harness import data, kernel
from harness.metrics import Metrics, tflops

WARMUP = 10
REPS = 50


def time_ms(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times), statistics.stdev(times)


def measure(spec):
    if spec["target"] == "eager":
        fn, params = data.eager, None
    else:
        params = kernel.Params(**spec.get("params", {}))
        fn = lambda x, w, b: kernel.run(params, x, w, b)

    m = Metrics(target=spec["target"], params=params.dict() if params else None)
    for i, (name, M, N, K, scale) in enumerate(data.CASES):
        x, w, b = data.make(M, N, K, scale)
        ref = data.reference(x, w, b)
        torch.cuda.reset_peak_memory_stats()
        try:
            y = fn(x, w, b)
            torch.cuda.synchronize()
        except Exception as e:
            m.compiled_ok = False
            m.error = f"{type(e).__name__}: {e}"[:2000]
            return m
        m.compiled_ok = True
        err = (y.float() - ref).abs()
        m.abs_errors[name] = err.max().item()
        m.errors[name] = (err / ref.abs().clamp_min(1)).max().item()
        if i == 0:
            m.vram_mb = torch.cuda.max_memory_allocated() / 2**20
            m.latency_ms, m.latency_std_ms = time_ms(lambda: fn(x, w, b))
            m.tflops = tflops(M, N, K, m.latency_ms)
            m.max_abs_out = ref.abs().max().item()

    # The default gate is the standard batch only. Adversarial gating comes from the Precision lens.
    m.max_error = m.errors[data.CASES[0][0]]
    m.adv_error = max(v for k, v in m.errors.items() if k != data.CASES[0][0])
    m.correct_ok = m.max_error < data.BOUND
    return m


if __name__ == "__main__":
    torch.zeros(1, device="cuda")
    print("ready", flush=True)
    print(json.dumps(measure(json.loads(sys.stdin.read())).dict()))
