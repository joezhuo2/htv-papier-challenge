"""Programmatic verdicts. The LLM states the prediction, this module alone decides whether it held."""

from dataclasses import dataclass, field

from harness.data import BOUND
from harness.metrics import Metrics


@dataclass
class Check:
    name: str
    predicted: str
    observed: str
    passed: bool | None  # None when the observed value sits inside the noise band

    def dict(self):
        return {"name": self.name, "predicted": self.predicted, "observed": self.observed, "pass": self.passed}


@dataclass
class Judgment:
    verdict: str
    provisional: bool
    reason: str
    checks: list[Check] = field(default_factory=list)


def sci(x):
    return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e")


def judge(prediction: dict, m: Metrics, base_ms: float | None, band: float, bound: float, adv_bound: float | None, strict=BOUND) -> Judgment:
    """prediction is {min_speedup, vs, max_error}. base_ms is the latency the speedup is measured against.
    bound is the standard batch bound set by the branch's lens (the Speed lens tolerates more than the strict 1e-3).
    adv_bound is the adversarial bound from an active Precision directive, or None when no lens gates it yet."""
    if not m.compiled_ok or m.latency_ms is None:
        why = m.error or "no measurement"
        return Judgment("rejected", False, f"The run failed: {why[:200]}", [Check("compiles and runs", "yes", "no", False)])

    checks = []
    if base_ms:
        speedup, target = base_ms / m.latency_ms, prediction["min_speedup"]
        vs = "naive" if prediction["vs"] == "naive" else "best"
        passed = True if speedup >= target * (1 + band) else False if speedup < target * (1 - band) else None
        observed = f"{speedup:.2f}x" + (f" (noise +/- {target * band:.2f}x)" if passed is None else "")
        checks.append(Check(f"speedup vs {vs}", f">= {target:.2f}x", observed, passed))

    checks.append(Check("standard max error", f"< {sci(bound)}", sci(m.max_error), m.max_error < bound))
    if adv_bound is not None:
        checks.append(Check("adversarial max error", f"< {sci(adv_bound)}", sci(m.adv_error), m.adv_error < adv_bound))

    failed = [c for c in checks if c.passed is False]
    if failed:
        return Judgment("rejected", False, "; ".join(f"{c.name} {c.observed} missed {c.predicted}" for c in failed) + ".", checks)
    if any(c.passed is None for c in checks):
        return Judgment("inconclusive", False, "The speedup is inside the noise band around the predicted threshold.", checks)
    # Confirmed only under a looser lens tolerance, or with adversarial error no lens gates yet: provisional.
    if adv_bound is None and (m.max_error >= strict or m.adv_error >= strict):
        return Judgment("confirmed", True, f"Predictions hold under the {sci(bound)} tolerance. Standard error {sci(m.max_error)} and adversarial error "
                                           f"{sci(m.adv_error)} are not both under the strict {sci(strict)} bound, which no lens enforces yet.", checks)
    return Judgment("confirmed", False, "Prediction holds.", checks)
