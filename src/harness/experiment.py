"""Child process entry point: measures one target and prints one JSON line. Run via harness.runner."""

import json
import statistics
import sys

import torch

from harness import data, kernel
from harness.metrics import Metrics, tflops

WARMUP = 10
REPS = 50
ROUNDS = 7
ROUND_REPS = 20


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


def round_ms(fn, reps=ROUND_REPS):
    times = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e))
    return statistics.median(times)


def interleaved(fns):
    """Times every target in the same process and the same window, rotating the order each round so no target always runs
    first or last. Returns each target's per-round median latencies."""
    for fn in fns.values():
        for _ in range(WARMUP):
            fn()
    torch.cuda.synchronize()
    names = list(fns)
    per = {n: [] for n in names}
    for r in range(ROUNDS):
        for n in names[r % len(names):] + names[:r % len(names)]:
            per[n].append(round_ms(fns[n]))
    return per


def call(f, x, w, b):
    return lambda: f(x, w, b)


def ref_fn(ref):
    if ref["target"] == "eager":
        return data.eager
    p = kernel.Params(**ref["params"])
    return lambda x, w, b: kernel.run(p, x, w, b)


def sweep(spec):
    """Max error of one config across K on the adversarial magnitude, keyed k<K> in errors. Used by post-mortems."""
    params = kernel.Params(**spec["params"])
    m = Metrics(target="sweep", params=params.dict(), compiled_ok=True)
    for K in spec["ks"]:
        x, w, b = data.make(512, 512, K, data.CASES[2][4])
        ref = data.reference(x, w, b)
        y = kernel.run(params, x, w, b).float()
        m.errors[f"k{K}"] = ((y - ref).abs() / ref.abs().clamp_min(1)).max().item()
    m.adv_error = max(m.errors.values())
    return m


def measure(spec):
    if spec["target"] == "sweep":
        return sweep(spec)
    m = Metrics(target=spec["target"])
    if spec["target"] == "eager":
        fn, params = data.eager, None
    else:
        params = kernel.Params(**spec.get("params", {}))
        m.params = params.dict()

        def fn(x, w, b):
            y, ck = kernel.launch(params, x, w, b)
            m.n_regs, m.n_spills = ck.n_regs, ck.n_spills
            return y

    for i, (name, M, N, K, scale) in enumerate(data.CASES):
        x, w, b = data.make(M, N, K, scale)
        ref = data.reference(x, w, b)
        torch.cuda.reset_peak_memory_stats()
        try:
            y = fn(x, w, b)
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001, any kernel failure (compile, launch, resources) is a measured result
            m.compiled_ok = False
            m.error = f"{type(e).__name__}: {e}"[:2000]
            return m
        m.compiled_ok = True
        err = (y.float() - ref).abs()
        m.abs_errors[name] = err.max().item()
        m.errors[name] = (err / ref.abs().clamp_min(1)).max().item()
        if i == 0:
            m.vram_mb = torch.cuda.max_memory_allocated() / 2**20
            m.max_abs_out = ref.abs().max().item()
            if spec.get("refs"):
                # speedups come from per-round ratios against references timed in the same window, with their spread
                fns = {"candidate": call(fn, x, w, b)} | {k: call(ref_fn(r), x, w, b) for k, r in spec["refs"].items()}
                per = interleaved(fns)
                m.latency_ms = statistics.median(per["candidate"])
                m.latency_std_ms = statistics.stdev(per["candidate"])
                for k in spec["refs"]:
                    ratios = [a / c for a, c in zip(per[k], per["candidate"])]
                    med = statistics.median(ratios)
                    m.speedups[k] = med
                    m.spreads[k] = (max(ratios) - min(ratios)) / 2 / med
                    m.ref_ms[k] = statistics.median(per[k])
            else:
                m.latency_ms, m.latency_std_ms = time_ms(call(fn, x, w, b))
            m.tflops = tflops(M, N, K, m.latency_ms)

    # The default gate is the standard batch only. Adversarial gating comes from the Precision lens.
    m.max_error = m.errors[data.CASES[0][0]]
    m.adv_error = max(v for k, v in m.errors.items() if k != data.CASES[0][0])
    m.correct_ok = m.max_error < data.BOUND
    return m


if __name__ == "__main__":
    torch.zeros(1, device="cuda")
    print("ready", flush=True)
    print(json.dumps(measure(json.loads(sys.stdin.read())).dict()))
