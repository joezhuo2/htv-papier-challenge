from dataclasses import asdict, dataclass, field, fields

CONTRACT = ("latency_ms", "tflops", "max_error", "adv_error", "vram_mb", "compiled_ok", "correct_ok")


@dataclass
class Metrics:
    """One run's measurements. The CONTRACT fields match Metrics in src/contract/types.ts, the rest is evidence."""

    compiled_ok: bool = False
    correct_ok: bool = False
    latency_ms: float | None = None
    tflops: float | None = None
    max_error: float | None = None
    adv_error: float | None = None
    vram_mb: float | None = None
    n_regs: int | None = None  # registers per thread in the compiled kernel
    n_spills: int | None = None  # 32-bit words spilled to local memory per thread
    latency_std_ms: float | None = None
    max_abs_out: float | None = None
    errors: dict[str, float] = field(default_factory=dict)
    abs_errors: dict[str, float] = field(default_factory=dict)
    speedups: dict[str, float] = field(default_factory=dict)  # vs references timed in the same window: naive, eager, parent
    spreads: dict[str, float] = field(default_factory=dict)  # relative half-range of the per-round ratios
    ref_ms: dict[str, float] = field(default_factory=dict)
    target: str = "triton"
    params: dict | None = None
    error: str | None = None
    run_id: str | None = None
    startup_s: float | None = None
    wall_s: float | None = None
    timed_out: bool = False

    @classmethod
    def from_dict(cls, d):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def dict(self):
        return asdict(self)

    def contract(self):
        d = {k: getattr(self, k) for k in CONTRACT}
        d |= {"speedups": self.speedups, "spreads": self.spreads} if self.speedups else {}
        return d | ({"n_regs": self.n_regs, "n_spills": self.n_spills} if self.n_regs is not None else {})


def tflops(M, N, K, latency_ms):
    return 2 * M * N * K / (latency_ms * 1e-3) / 1e12
