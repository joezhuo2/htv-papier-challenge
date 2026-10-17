"""The research loop: hypotheses with stated predictions, constrained runs, code computed verdicts, branches per lens, post-mortems.

Branches run one at a time on the single GPU. Commands from the UI (directives, help answers, hybrid approval) are applied the moment
they arrive by a listener thread. The loop holds the state lock except while it waits on a model or the GPU, and every GPU run
takes the GPU lock, so a post-mortem sweep never overlaps a timed run.
"""

import json
import queue
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

from agent import export as ex
from agent import hypothesis as hyp
from agent.events import Log
from agent.papiers import words
from agent.verdict import Judgment, judge, sci
from harness.data import BOUND
from harness.kernel import (
    ACC_LIMIT,
    NAIVE,
    SMEM_LIMIT,
    Params,
    acc_per_thread,
    fit_stages,
    smem_bytes,
)
from harness.metrics import Metrics
from harness.runner import TIMEOUT_S

SWEEP_KS = [256, 512, 1024, 2048, 4096]
LENS_IDS = {"speed": "speed", "precision": "prec"}
# Papiers queries name the change under test, the lens topic is the fallback before a proposal exists
FACETS = [(("acc",), "accumulation precision FP16"), (("gelu",), "GELU tanh approximation"), (TILE_KEYS := ("block_m", "block_n", "block_k", "warps", "stages"), "tile size shared memory")]
QUERIES = {"both": "tile size shared memory", "speed": "accumulation precision FP16", "precision": "accumulation error grows with length"}
MAX_PER_LENS = 3
DIRECTIVE_GAP_S = 10.0
MAX_DIRECTIVES = 6
TILE = TILE_KEYS
SLOW = 0.5  # a run below this speedup vs its parent gets a register post-mortem
PAPIERS_ID = re.compile(r"\b(?:cnv|hl|nte|doc|psg|pas)_[0-9a-f]{8,}\b")


@dataclass
class Branch:
    id: str
    parents: list[str]
    label: str
    lens: str
    status: str = "running"
    iters: int = 0
    misses: int = 0
    fails: int = 0  # consecutive runs that failed to compile or run
    budget: int | None = None  # iterations, when not the main or branch default (a hybrid runs once)
    pivot: Params | None = None  # for a pivot branch, the rejected config with what its post-mortem ruled out dropped

    def dict(self):
        return {"id": self.id, "parents": self.parents, "label": self.label, "lens": self.lens, "status": self.status}


@dataclass
class Hyp:
    id: str
    branch: str
    claim: str
    prediction: dict
    params: Params
    note: str | None = None
    run: str | None = None
    metrics: Metrics | None = None
    speedup: float | None = None
    noise: float = 0.0
    judgment: Judgment | None = None
    history: list[str] = field(default_factory=list)
    bounds: list[dict] = field(default_factory=list)
    base: Params | None = None  # the config a harness variant is compared against, when it is not the branch best
    ref: str = "naive"  # which same-window reference the speedup was taken against
    bonus: bool = False  # ran in a reserved slot (a presenter hunch or the approved hybrid), not counted against the cap
    papiers: str | None = None  # the id of the Papiers item cited in the note

    def dict(self):
        d = {"id": self.id, "branch": self.branch, "claim": self.claim, "prediction": self.prediction, "params": self.params.dict()}
        return d | ({"note": self.note} if self.note else {}) | ({"papiers": self.papiers} if self.papiers else {})


@dataclass
class Hint:
    """What a human asked a branch to test next: a bound (delta None), a concrete change (delta), or a vague one ({}).
    params is set when the change was built in code at the moment it arrived, earlier is the run that tested it before."""
    text: str
    cite: str
    ref: str  # the directive or help id
    delta: dict | None
    params: Params | None = None
    earlier: "Hyp | None" = None
    base: Params | None = None  # the current best params the change was applied to


def open_help(x):
    return not (x["answered"] or x["expired"] or x["cancelled"])


def bound_owner(bound):
    return {"speed": "Speed lens", "precision": "Precision lens"}.get(bound["set_by"], "default")


def tiles_text(p: Params):
    return f"{p.block_m}x{p.block_n}x{p.block_k} tiles, {p.warps} warps, {p.stages} stages"


def params_text(p: Params):
    return f"{tiles_text(p)}, {p.acc.upper()} accumulation, {p.gelu} GELU"


def facet(a: Params, b: Params):
    """The Papiers query for the change from a to b: precision first, then the GELU formula, then tiling. None when nothing changed."""
    keys = {k for k in a.dict() if getattr(a, k) != getattr(b, k)}
    return next((q for ks, q in FACETS if keys & set(ks)), None)


def cite_text(n):
    """Names the cited item by its document, the id goes in Hyp.papiers instead of the prose."""
    title = (n.get("document_title") or n["title"]).strip()
    kind = n.get("kind") or {"hl": "highlight", "nte": "note", "cnv": "conversation"}.get(n["id"].split("_")[0], "passage")
    return (f"Context from your {kind} in \"{title}\"" if kind in ("highlight", "note")
            else f"Context from the Papiers {kind} \"{title}\"")


def claim_text(start: Params, p: Params, pred, bound):
    """A claim built from the final parameters, used whenever the harness adjusted the proposed ones."""
    vs = "naive" if pred["vs"] == "naive" else "the best configuration on this branch"
    return (f"{change(start, p).capitalize()}, giving {params_text(p)}, is at least {pred['min_speedup']:.2f}x faster than {vs}, "
            f"with standard max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")


def change(a: Params, b: Params):
    diff = [f"{k} {getattr(a, k)} to {getattr(b, k)}" for k in a.dict() if getattr(a, k) != getattr(b, k)]
    return ", ".join(diff) or "no parameter change"


def direction(k):
    """block_m and block_n move together as the output tile, every other knob is its own direction."""
    return "tile" if k in ("block_m", "block_n") else k


def value_text(k, v):
    return f"{v.upper()} accumulation" if k == "acc" else f"{v} GELU" if k == "gelu" else f"{k} {v}"


def setting(p: Params, k):
    return value_text(k, getattr(p, k))


def pivot_keys(p: Params, base: Params):
    """What a post-mortem rules out: the precision or GELU change when there is one, otherwise every changed parameter."""
    return [k for k in p.dict() if getattr(p, k) != getattr(base, k) and k in ("acc", "gelu")] or [k for k in p.dict() if getattr(p, k) != getattr(base, k)]


def ruled_out(p: Params, base: Params):
    return ", ".join(setting(p, k) for k in pivot_keys(p, base)) or f"the configuration {params_text(p)}"


def failure_cause(p: Params, m: Metrics):
    """Templated post-mortem for a run that did not compile or run, with the shared memory arithmetic when Triton ran out of it."""
    err = m.error or ""
    if "shared memory" in err:
        need = smem_bytes(p)
        fits = next((s for s in range(p.stages - 1, 0, -1) if smem_bytes(replace(p, stages=s)) <= SMEM_LIMIT), None)
        cause = (f"Shared memory: (stages {p.stages} - 1) x ({p.block_m}x{p.block_k} + {p.block_k}x{p.block_n}) x 2 bytes = {need} bytes, over the "
                 f"{SMEM_LIMIT} byte limit, so Triton refused to compile it.")
        pivot = (f"Keep the tiles with {fits} stages ({smem_bytes(replace(p, stages=fits))} bytes)." if fits and fits > 1
                 else "Halve block_k or the larger output tile so a pipelined config fits.")
        return {"cause": cause, "pivot": pivot}
    if m.timed_out:
        return {"cause": f"The run hit the {TIMEOUT_S} s harness timeout and was killed before it produced a measurement.",
                "pivot": "Go back to the best configuration on this branch and change one parameter at a time."}
    return {"cause": f"The run failed before measuring: {err.strip().splitlines()[-1][:200] if err.strip() else 'no error text'}.",
            "pivot": "Go back to the best configuration on this branch and change one parameter at a time."}


def register_cause(p: Params, m: Metrics, speedup, parent: Metrics | None):
    """Templated post-mortem for a run far slower than its parent, from the compiled kernel's register and spill counts."""
    per = acc_per_thread(p)
    regs = (f"The compiled kernel uses {m.n_regs} registers per thread and spills {m.n_spills} 32-bit words ({m.n_spills * 4} bytes) per thread to local memory"
            if m.n_regs is not None else "The run did not record its register count")
    vs = f", against {parent.n_regs} registers and {parent.n_spills} spilled words for the parent" if parent and parent.n_regs is not None else ""
    tile = f"Its {p.block_m}x{p.block_n} {p.acc.upper()} accumulator over {p.warps} warps ({p.warps * 32} threads) is {per} elements per thread"
    why = (f"{tile}, against at most {ACC_LIMIT[p.acc]} that fit beside the operands, so the inner loop reads and writes local memory"
           if per > ACC_LIMIT[p.acc] else f"{tile}, within the {ACC_LIMIT[p.acc]} limit, so the accumulator alone does not explain the slowdown")
    cause = f"{regs}{vs}. {why}. It runs at {speedup:.2f}x of the parent."
    return {"cause": cause, "pivot": f"Keep the accumulator at or under {ACC_LIMIT[p.acc]} elements per thread: a smaller output tile or more warps."}


KNOB_EFFECT = {
    ("stages", 1): "a deeper pipeline holds more shared memory per block, so fewer blocks fit on each SM, without hiding more load latency here",
    ("stages", -1): "a shallower pipeline leaves less load latency hidden behind the math",
    ("warps", 1): "more warps split the same tile thinner, so each thread does less math per load",
    ("warps", -1): "fewer warps leave less parallelism to hide memory latency",
    ("block_k", 1): "a longer K step loads more per iteration and raises shared memory per stage",
    ("block_k", -1): "a shorter K step means more loop iterations and more synchronization per output tile",
    ("block_m", 1): "a taller output tile raises registers and shared memory per block", ("block_m", -1): "a shorter output tile reuses each loaded column less",
    ("block_n", 1): "a wider output tile raises registers and shared memory per block", ("block_n", -1): "a narrower output tile reuses each loaded row less",
}


def miss_cause(p: Params, base: Params, m: Metrics, parent: Metrics, speedup, pred, ref, noise=None):
    """Templated post-mortem for a speed miss: the likely cause from registers, spills and the parameter diff. noise is the decision
    band of an inconclusive run, whose speedup sits inside it."""
    vs = "naive" if ref == "naive" else "its parent"
    regs = (f"{parent.n_regs} to {m.n_regs} registers and {parent.n_spills} to {m.n_spills} spilled words per thread"
            if m.n_regs is not None and parent.n_regs is not None else
            f"{m.n_regs} registers and {m.n_spills} spilled words per thread" if m.n_regs is not None else "no register count recorded")
    per, limit = acc_per_thread(p), ACC_LIMIT[p.acc]
    knobs = [k for k in TILE if getattr(p, k) != getattr(base, k)]
    if m.n_spills and m.n_spills > (parent.n_spills or 0):
        why = f"it spills {m.n_spills} words per thread to local memory, so the inner loop waits on memory"
        pivot = f"Keep the accumulator at or under {limit} elements per thread ({per} now): a smaller output tile or more warps."
    elif per > limit:
        why = f"{per} {p.acc.upper()} accumulator elements per thread is over the {limit} that fit in registers"
        pivot = f"Keep the accumulator at or under {limit} elements per thread: a smaller output tile or more warps."
    elif m.n_regs and parent.n_regs and m.n_regs > 1.15 * parent.n_regs:
        why = f"registers per thread rose from {parent.n_regs} to {m.n_regs}, so fewer warps fit on each SM"
        pivot = f"Revert {change(base, p)} and try a change that does not add registers."
    else:
        effects = [KNOB_EFFECT[(k, 1 if getattr(p, k) > getattr(base, k) else -1)] for k in knobs]
        why = (f"registers and spills barely moved, so the change itself is the likely cause: {'; '.join(effects)}" if effects
               else f"registers and spills barely moved, so {change(base, p)} itself is the likely cause")
        pivot = f"Revert {change(base, p)} and change a knob the ledger has not tried yet."
    if noise is not None:
        gain = (f"It ran at {speedup:.2f}x vs {vs}, inside the +/- {noise * 100:.1f}% noise band around the predicted {pred['min_speedup']:.2f}x, "
                "so this run cannot tell whether the change clears the prediction")
        if not (m.n_spills and m.n_spills > (parent.n_spills or 0)) and per <= limit:
            pivot = f"Rerun {change(base, p)} with more timing rounds, or keep the parent and change a knob the ledger has not tried yet."
    elif speedup >= 1:
        gain = f"It ran at {speedup:.2f}x vs {vs}, faster but short of the predicted {pred['min_speedup']:.2f}x, so the change helps less than claimed"
    else:
        gain = f"It ran at {speedup:.2f}x vs {vs}, slower than what it was timed against"
    return {"cause": f"{gain}. Changed {change(base, p)} ({regs}). Likely cause: {why}.", "pivot": pivot}


def assess(h):
    """(confidence, confidence_reason, uncertainties, would_change_mind) from the verdict, its checks and the run's measured spread."""
    m, j = h.metrics, h.judgment
    shape = "Measured on one shape (2048 x 2048 x 2048) only."
    if m.latency_ms is None:
        return ("low", "The run produced no measurement, so nothing about speed or error is known for this config.",
                ["No timing or error was measured.", shape], "A run of the same config that compiles and measures.")
    spread = m.spreads.get(h.ref, 0.0)
    speed = next((c for c in j.checks if c.name.startswith("speedup")), None)
    target = h.prediction["min_speedup"]
    vs = "naive" if h.ref == "naive" else "the parent"
    err_fail = [c for c in j.checks if c.passed is False and "error" in c.name]
    unsure = [shape]
    if not any(c.name.startswith("adversarial") for c in j.checks):
        unsure.append(f"Adversarial error {sci(m.adv_error)} is measured but not gated by any lens yet.")
    band = f"this run's measured spread of +/- {spread * 100:.1f}% (decision band +/- {h.noise * 100:.1f}%)"
    if err_fail:
        c = err_fail[0]
        return ("high", f"{c.name.capitalize()} {c.observed} against {c.predicted}. Errors are deterministic with fixed-seed inputs, so a rerun gives the same value.",
                unsure, f"An error under {c.predicted.lstrip('< ')} on the same inputs, which would mean the harness changed.")
    if j.verdict == "inconclusive":
        unsure.insert(0, f"The speedup is within {band}.")
        return ("low", f"{h.speedup:.2f}x vs {vs} sits inside {band} around the {target:.2f}x threshold.",
                unsure, f"A same-window rerun clearly above {target * (1 + h.noise):.2f}x or below {target * (1 - h.noise):.2f}x vs {vs}.")
    if j.provisional:
        return ("low", f"It passes only under the Speed tolerance: adversarial error {sci(m.adv_error)} has no Precision bound yet.",
                unsure, f"A Precision bound under {sci(m.adv_error)} on the adversarial batch, which this run would fail.")
    margin = h.speedup / target - 1 if speed else 0.0
    if j.verdict == "confirmed":
        level = "high" if margin > 2 * h.noise else "medium"
        return (level, f"{h.speedup:.2f}x vs {vs} clears the {target:.2f}x threshold by {margin * 100:.1f}%, against {band}; every error check passed.",
                unsure, f"A same-window rerun below {target * (1 - h.noise):.2f}x vs {vs}.")
    level = "high" if -margin > 2 * h.noise else "medium"
    return (level, f"{h.speedup:.2f}x vs {vs} misses the {target:.2f}x threshold by {-margin * 100:.1f}%, outside {band}.",
            unsure, f"A same-window rerun at or above {target * (1 + h.noise):.2f}x vs {vs}.")


def errs(h):
    """The error checks this hypothesis failed under its latest verdict."""
    return [c for c in h.judgment.checks if c.passed is False and "error" in c.name]


class Stopped(Exception):
    pass


class Orchestrator:
    """engine(spec, run_id) -> Metrics runs one experiment (harness.runner.run in production).
    ask(messages, model, tag) -> str is the LLM. fast and strong are model ids.
    context(query) -> [{id, title, excerpt}] reads Papiers, optional. headroom is harness.noise.headroom(), quoted in the prompt.
    record marks a recording session (set with set_record): uncapped by the live demo cap, it never finishes on idle, it waits
    directive_wait_s for a lens directive whenever nothing is runnable and then continues on main as a flagged default, and it
    saves its log as a replay."""

    def __init__(self, engine, ask, log: Log, session: str, fast: str, strong: str, band: float,
                 main_iters=2, branch_iters=3, idle_s=120.0, context=None, help_deadline_s=45.0, max_hyps=None, retry_s=30.0, headroom=None, directive_wait_s=90.0):
        self.engine, self.ask, self.log, self.session = engine, ask, log, session
        self.fast, self.strong, self.band = fast, strong, band
        self.main_iters, self.branch_iters, self.idle_s = main_iters, branch_iters, idle_s
        self.branches: dict[str, Branch] = {}
        self.hyps: list[Hyp] = []
        self.directives: list[dict] = []
        self.avoid: list[str] = []
        self.commands = queue.Queue()
        self.stopped = False
        self.runs = 0
        self.context = context
        self.read: dict[str, list] = {}
        self.help_deadline_s = help_deadline_s
        self.max_hyps, self.retry_s = max_hyps, retry_s
        self.headroom = headroom
        self.record = False
        self.directive_wait_s = directive_wait_s
        self.defaulted: set[str] = set()  # branches reopened by the no-directive default, flagged on their hypotheses
        self.waiting = False
        self.announced = False  # the waiting line is logged once per session
        self.retry = threading.Event()
        self.applied = threading.Event()  # set by the listener when a command arrived, so an idle session stays open
        self.helps: dict[str, dict] = {}  # id -> {help, branch, t, answered, expired}
        self.hints: dict[str, Hint] = {}  # branch -> what a human asked it to test next
        self.extra: dict[str, int] = {}  # branch -> iterations added by a help answer
        self.summaries: dict[str, dict] = {}
        self.summarized: set[str] = set()  # hypotheses whose latest verdict has a summary
        self.pms: dict[str, dict] = {}
        self.promises: list[dict] = []  # extra iterations a plan_change promised: {ref, branch, n, start}
        self.conflicts = 0
        self.hybrid: dict | None = None  # {branches, params, base, text, t, state: proposed, approved, expired, done}
        self.skipped = ""  # the last logged reason no hybrid could be proposed
        self.unmerged = ""  # why no run qualified for a hybrid so far, logged at the end if none was proposed
        self.hunches: dict[str, str] = {}  # directive or help id of a concrete hunch -> the hypothesis that tested it, or "not run: <why>"
        self.taken: Hint | None = None  # the hint the current proposal used, restored if its branch is rejected before the run
        self.reserved: set[str] = set()  # slots past the cap: directive ids of hunches still to test, and "hybrid" once approved
        self.lock = threading.RLock()
        self.gpu = threading.Lock()
        self.done = False
        self.debug = log.path.with_name("debug.log")

    # ---- public, thread safe ----

    def command(self, cmd: dict):
        if cmd["type"] == "stop":
            # acts at once instead of between iterations, so an in-flight model call does not delay it
            if not self.stopped:
                self.log.emit("agent_log", text="Stopped by the presenter. Finishing the session with what is logged so far.")
            self.stop()
            return
        self.commands.put(cmd)
        if cmd["type"] == "retry":
            self.retry.set()  # wakes a loop that is waiting out a model failure

    def note(self, text):
        """Diagnostics that do not belong in the replay (provider failover, file paths) go to debug.log next to the events."""
        with self.debug.open("a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    @contextmanager
    def released(self):
        """Lets the listener apply commands while this thread waits on a model or the GPU."""
        try:
            self.lock.release()
        except RuntimeError:  # not held, as in tests that call a step directly
            yield
            return
        try:
            yield
        finally:
            self.lock.acquire()

    def gpu_run(self, spec, run_id):
        with self.released(), self.gpu:
            return self.engine(spec, run_id)

    def listen(self):
        """Applies commands as they arrive and enforces help and hybrid deadlines on the wall clock."""
        while not self.done:
            try:
                cmd = self.commands.get(timeout=0.2)
            except queue.Empty:
                cmd = None
            with self.lock:
                if self.done:
                    return
                if cmd:
                    self.handle(cmd)
                self.check_deadlines()

    def stop(self):
        self.stopped = True
        self.retry.set()

    def set_record(self, max_hyps):
        """Recording mode: the main branch stops after H0 so the session waits for a lens directive."""
        self.record, self.max_hyps, self.main_iters = True, max_hyps, 1

    def expecting(self):
        """Whether a lens has yet to send a directive, the only reason to keep an idle session open."""
        return bool({"speed", "precision"} - {d["lens"] for d in self.directives})

    def base_capped(self):
        return self.max_hyps is not None and sum(not h.bonus for h in self.hyps) >= self.max_hyps

    def owed(self):
        """Branches holding a reserved slot: a presenter hunch still to test, or the approved hybrid."""
        out = {bid for bid, hint in self.hints.items() if hint.ref in self.reserved}
        return out | ({"merge"} if "hybrid" in self.reserved else set())

    def capped(self):
        return self.base_capped() and not any(self.branches[b].status in ("running", "waiting_on_human") for b in self.owed() if b in self.branches)

    def spend(self, h, hint):
        """Marks h as the hypothesis that used its hint's reserved slot, if the hint had one."""
        if h and hint and hint.ref in self.reserved:
            h.bonus = True
            self.reserved.discard(hint.ref)

    def run(self):
        with self.lock:
            ok = self.start()
        if not ok:
            return
        threading.Thread(target=self.listen, daemon=True).start()
        idle = time.monotonic()
        while True:
            with self.lock:
                if self.stopped:
                    break
                self.rehome_hints()
                b = self.next_branch()
                if b:
                    self.waiting = False
                    self.iterate(b)
                    idle = time.monotonic()
                    continue
                busy = self.pending() or self.applied.is_set()
                self.applied.clear()
                if busy:
                    idle = time.monotonic()
                if self.capped():
                    self.maybe_hybrid()
                    if not (self.hybrid and self.hybrid["state"] == "proposed"):
                        break  # help deadlines are not waited out, but a proposed hybrid waits for approval or its deadline
                if not busy and not self.expecting():
                    # nothing runs, nothing waits on a human, and both lenses have spoken: waiting longer only adds an idle tail
                    self.maybe_hybrid()
                    if not self.pending():
                        unused = f" with {self.max_hyps - sum(not h.bonus for h in self.hyps)} of its {self.max_hyps} hypotheses unused" if self.max_hyps else ""
                        self.log.emit("agent_log", text=f"Every branch is done and both lenses have sent a directive, so the session ends{unused}.")
                        break
                if self.record:
                    if self.base_capped():
                        idle = time.monotonic()
                    elif not self.waiting:
                        self.waiting = True
                        if not self.announced:
                            self.announced = True
                            self.log.emit("agent_log", text=f"Waiting for a lens directive. With none in {self.directive_wait_s:.0f} s I continue on main with the default.")
                    elif time.monotonic() - idle > self.directive_wait_s:
                        self.proceed_default()
                        idle = time.monotonic()
                elif time.monotonic() - idle > self.idle_s:
                    break
            time.sleep(0.1)
        with self.lock:
            self.finish()
            self.done = True

    # ---- session ----

    def start(self):
        """Measures the baselines. Returns False, with the reason in the agent log, when the session cannot start."""
        self.log.t0 = self.log.clock()  # session time starts at Start, not when the server built the session
        self.log.emit("agent_log", text="Measuring baselines: PyTorch eager and the naive Triton kernel.")
        self.eager = self.gpu_run({"target": "eager"}, f"{self.session}/eager")
        self.naive = self.gpu_run({"target": "triton", "params": NAIVE.dict()}, f"{self.session}/naive")
        failed = next((name for name, m in (("eager", self.eager), ("naive", self.naive)) if m.error), None)
        if failed:
            cause = getattr(self, failed).error.strip().splitlines()[-1][:240]
            self.log.emit("agent_log", text=f"Cannot start: the {failed} baseline did not run ({cause}). Check the GPU, then press New session.")
            return False
        if not self.naive.correct_ok:
            self.log.emit("agent_log", text=f"Cannot start: the naive kernel fails the correctness gate (max error {sci(self.naive.max_error)}).")
            return False
        main = Branch("main", [], "Baseline tuning", "both")
        self.branches["main"] = main
        self.log.emit("session_started", baselines={"eager": self.eager.contract(), "naive": self.naive.contract()}, branch=main.dict())
        self.log.emit("agent_log", text=f"Baselines measured: naive {self.naive.latency_ms:.3f} ms, eager {self.eager.latency_ms:.3f} ms "
                                        f"(eager standard error {sci(self.eager.max_error)}). Starting from the naive configuration.")
        if self.record:
            self.log.emit("agent_log", text=f"Recording mode: the full loop for up to {self.max_hyps} hypotheses or until Stop. "
                                            "Directives and help answers are accepted at any point.")
        return True

    def finish(self):
        for h in self.hyps:
            if h.judgment and h.id not in self.summarized:
                self.summarize(h, self.branches[h.branch], "Session stopped before the next step." if self.stopped else "Session finished.")
        ledger = [{"hypothesis": h.id, "verdict": h.judgment.verdict, "history": h.history} for h in self.hyps if h.judgment]
        best = self.fastest()
        lat = best.metrics.latency_ms if best else self.naive.latency_ms
        # one metric for the best run everywhere: the baselines' latency over the fastest run within every bound, shown with its verdict
        out = ({"run": best.run, "hypothesis": best.id, "verdict": best.judgment.verdict} if best else {"run": "naive"}) \
            | {"vs_naive": round(self.naive.latency_ms / lat, 2), "vs_eager": round(self.eager.latency_ms / lat, 2)}
        if ledger:
            self.export([("ledger", f"Hypothesis ledger {self.hyps[0].id} to {self.hyps[-1].id}", "ledger.md",
                          ex.ledger_md(self.hyps, out if best else None, self.naive.latency_ms, self.eager.latency_ms))]
                        + [("summary", f"Branch summary: {bid}", f"summary-{bid}.md", ex.summary_md(sm)) for bid, sm in self.summaries.items()])
        why = (f"the {self.max_hyps} hypothesis cap was reached" if self.base_capped() else "the session was stopped" if self.stopped
               else "the session went idle")
        for hid, x in self.helps.items():
            if open_help(x):
                x["expired"] = True
                self.log.emit("plan_change", help=hid, text=f"Session ended with {hid} unanswered: {why}, so {x['branch']} gets no more iterations.")
        if not self.hybrid and self.unmerged:
            self.log.emit("agent_log", text=self.unmerged)
        if self.hybrid and self.hybrid["state"] == "proposed":
            self.log.emit("agent_log", text=f"Session ended with the hybrid of {' and '.join(self.hybrid['branches'])} unapproved: {why}.")
        for pr in self.promises:
            left = pr["n"] - (sum(h.branch == pr["branch"] for h in self.hyps) - pr["start"])
            if left <= 0:
                continue
            b = self.branches[pr["branch"]]
            why = (f"{b.id} closed ({b.status})" if b.status not in ("running", "waiting_on_human")
                   else f"The {self.max_hyps} hypothesis cap ended the session" if self.base_capped()
                   else "The session was stopped" if self.stopped else "The session went idle")
            self.log.emit("plan_change", help=pr["ref"], text=f"{why} before {left} of the {pr['n']} iterations promised in {pr['ref']} ran on {b.id}.")
        for b in self.branches.values():
            if b.status in ("running", "waiting_on_human"):
                self.close(b, "done")
        self.log.emit("final_report", report=self.final_report(best, out))
        replay = self.log.path.with_name("replay.jsonl")
        self.note(f"replay: ?mode=replay&file=/runs/{self.session}/replay.jsonl (written after session_finished)")
        self.log.emit("session_finished", ledger=ledger, best=out)
        # a replay file only ever holds a complete session
        tmp = replay.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in self.log.events))
        tmp.replace(replay)

    def final_report(self, best, out):
        """Computed from the log: the best confirmed run, what was ruled out, what to try next, what needs a human, and each typed hunch's outcome."""
        done = [h for h in self.hyps if h.judgment]
        ruled = [f"{h.id} on {h.branch}, {change(self.baseline(h)[0], h.params)}: {self.outcome(h)}" for h in done if h.judgment.verdict == "rejected"]
        nxt = [f"Rerun {h.id} ({change(self.baseline(h)[0], h.params)}) with more timing rounds: {self.outcome(h)} sits inside the noise band."
               for h in done if h.judgment.verdict == "inconclusive"][-2:]
        nxt += list(dict.fromkeys(self.pms[h.id]["pivot"] for h in reversed(done) if h.id in self.pms))[:3 - len(nxt)]
        human = [f"{hid} ({x['help']['target'].capitalize()} lens): {x['help']['question']}" for hid, x in self.helps.items()
                 if not x["answered"] and not x["cancelled"]]
        if self.hybrid and self.hybrid["state"] in ("proposed", "expired"):
            human.append(f"Approve or drop the hybrid of {' and '.join(self.hybrid['branches'])}: {params_text(self.hybrid['params'])}.")
        for h in done:
            fails = errs(h)
            if h.history[:-1] and "confirmed" in h.history[:-1] and fails and h.speedup:
                human.append(f"{h.id} ran {h.speedup:.2f}x faster but fails {' and '.join(c.name for c in fails)} ({', '.join(c.observed for c in fails)}) under "
                             f"the {bound_owner(h.bounds[-1])} bound. Decide whether that error is acceptable for this workload.")
        hunches = []
        for d in self.directives:
            if hyp.hunch_change(d["text"]) is None:
                continue
            got = self.hunches.get(d["id"]) or ("not run: the session ended before it ran" if d["id"] in self.reserved else "not run")
            x = next((h for h in self.hyps if h.id == got), None)
            result = (f"{x.id} on {x.branch} ran {params_text(x.params)}: {self.outcome(x)}" if x and x.judgment
                      else f"{x.id} was proposed but not measured" if x else got)
            hunches.append({"directive": d["id"], "lens": d["lens"], "text": d["text"], "outcome": result})
        for hid, x in self.helps.items():
            note = (x.get("cmd") or {}).get("note")
            if x["answered"] and x["cmd"]["option"] == "other" and note:
                y = next((h for h in self.hyps if h.id == self.hunches.get(hid) and h.judgment), None)
                hunches.append({"directive": hid, "lens": x["cmd"]["by"], "text": note,
                                "outcome": f"{y.id} on {y.branch} ran {params_text(y.params)}: {self.outcome(y)}" if y else self.hunches.get(hid, "not run")})
        best_run = best and {"run": best.run, "hypothesis": best.id, "verdict": best.judgment.verdict, "branch": best.branch, "params": best.params.dict(),
                             "latency_ms": best.metrics.latency_ms} | {k: out[k] for k in ("vs_naive", "vs_eager")}
        return {"best": best_run, "ruled_out": ruled, "next": nxt, "human": human, "hunches": hunches}

    def proceed_default(self):
        main = self.branches["main"]
        self.defaulted.add(main.id)
        self.extra[main.id] = self.extra.get(main.id, 0) + 2
        main.misses = 0
        self.log.emit("agent_log", text=f"No lens directive in {self.directive_wait_s:.0f} s, so I am proceeding on the default assumption: two more tuning "
                                        "iterations on main under the default 1e-3 bound. These hypotheses are flagged as default, not human directed.")
        self.reopen(main)
        self.waiting = False

    # ---- commands ----

    def handle(self, cmd):
        self.applied.set()
        if cmd["type"] == "add_directive":
            self.add_directive(cmd["lens"], cmd["text"])
        elif cmd["type"] == "answer_help":
            self.answer(cmd)
        elif cmd["type"] == "approve_hybrid":
            self.approve_hybrid(cmd.get("branches") or [])
        elif cmd["type"] in ("retry", "start"):
            self.retry.set()

    def add_directive(self, lens, text):
        now = self.log.clock() - self.log.t0
        recent = [x for x in self.directives if x["lens"] == lens and now - x["at"] < DIRECTIVE_GAP_S]
        if recent or len(self.directives) >= MAX_DIRECTIVES:
            why = f"another {lens} directive arrived less than {DIRECTIVE_GAP_S:.0f} s ago" if recent else f"the session already has {MAX_DIRECTIVES} directives"
            self.log.emit("agent_log", text=f"Ignored a {lens} directive ({why}): {text[:80]}")
            return
        d = {"id": f"d{len(self.directives) + 1}", "lens": lens, "text": text, **hyp.directive(lens, text)}
        self.defaulted.clear()
        self.directives.append(d | {"at": now})
        self.log.emit("directive_added", directive=d)
        if lens == "speed" and any(x["lens"] == "precision" for x in self.directives):
            self.log.emit("agent_log", text=f"Speed lens directive recorded. The Precision bound of {sci(self.bounds('speed')['standard'])} on the standard and "
                                            f"adversarial batches still applies to every branch, so the Speed tolerance of {sci(d['threshold'])} does not loosen it.")
        elif lens == "speed":
            self.log.emit("agent_log", text=f"Speed lens bound assigned: max error under {sci(d['threshold'])} on the standard batch for Speed branches "
                                            f"(default {sci(hyp.SPEED_TOLERANCE)}, ceiling {sci(hyp.SPEED_CEILING)}).")
        else:
            self.log.emit("agent_log", text=f"Precision bound assigned: {sci(d['threshold'])} on the standard and adversarial batches for every branch. "
                                            "Re-evaluating logged runs, no new runs needed.")
        delta = hyp.hunch_change(text)
        name = "Speed" if lens == "speed" else "Precision"
        open_ = [b for b in self.branches.values() if b.lens == lens and b.status in ("running", "waiting_on_human")]
        if open_:
            target = open_[-1]
        elif self.stopped or (delta is None and self.base_capped()):
            # nothing would run on a new branch, so none is opened; a Precision bound still re-judges the logged runs below
            why = "the session was stopped" if self.stopped else f"the session reached its {self.max_hyps} hypothesis cap"
            self.log.emit("plan_change", help=d["id"], text=f"No new {name} branch for {d['id']} because {why}"
                                                           + (". Its bound applies to the logged runs." if lens == "precision" else "."))
            target = None
        else:
            parent = self.best_run()
            target = self.fork(parent.branch if parent else "main", lens, f"{name} lens", f"{name} lens directive {d['id']}: {text}")
        if target and delta is None:
            self.bound_only(target, d)
        elif target:
            self.hunch(target, text, f"Hunch from the {lens.capitalize()} lens in directive {d['id']}", d["id"], lens, delta)
        if lens == "precision":
            self.reevaluate(d)

    def bound_only(self, b, d):
        """A directive that names no concrete change is a constraint: the next hypothesis is attributed to its bound, not called a hunch."""
        if self.capped() or self.stopped:
            why = "the session was stopped" if self.stopped else f"the session reached its {self.max_hyps} hypothesis cap"
            self.log.emit("plan_change", help=d["id"], text=f"{d['id']} sets a bound, not a specific change. No new hypotheses run on {b.id} because {why}, "
                                                           "so the bound applies to the logged runs only.")
            return
        lens = d["lens"].capitalize()
        self.hints.setdefault(b.id, Hint(d["text"], f"Tested under the {lens} lens bound set by {d['id']}", d["id"], None))
        bound = self.bounds(b.lens)
        self.log.emit("plan_change", help=d["id"], text=f"{d['id']} sets a bound, not a specific change, so the next hypotheses on {b.id} are chosen by the agent "
                                                       f"and judged against max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")

    def tried_with(self, delta):
        """Measured hypotheses that set this hunch's values and changed at least one of them from what they were timed against."""
        return [x for x in self.hyps if x.judgment and x.metrics and x.metrics.latency_ms and all(getattr(x.params, k) == v for k, v in delta.items())
                and set(self.knobs(x)) & set(delta)]

    def tested(self, delta):
        """The latest hypothesis that tested this hunch in isolation (it changed no other knob), or None."""
        hits = [x for x in self.tried_with(delta) if set(self.knobs(x)) <= set(delta)]
        return hits[-1] if hits else None

    def plan(self, b, delta):
        """(params, earlier, why_not, dup) for a concrete hunch on b: its change applied to b's current best params, built in code. earlier is
        the run that tested the change before, why_not says why it cannot run. dup is the hypothesis that already has that exact config
        (finished or being measured), or True when it is queued, so not even the model may run it."""
        start = self.start_point(b.id)[0]
        e = self.tested(delta)
        try:
            p = fit_stages(replace(start, **delta))
        except ValueError as err:
            return None, e, f"it names an invalid value ({err})", None
        if acc_per_thread(p) > ACC_LIMIT[p.acc]:
            return None, e, (f"on {b.id}'s current best it gives {params_text(p)}, which puts {acc_per_thread(p)} accumulator elements on each thread, "
                             f"over the {ACC_LIMIT[p.acc]} register limit"), None
        if p == start:
            return None, e, f"{b.id}'s current best, {params_text(start)}, already has it", None
        dup, why = self.duplicate(p)
        return (None, e, why, dup) if dup else (p, e, None, None)

    def duplicate(self, p):
        """(match, why) when the exact config p, accumulation and GELU included, already ran, is being measured, or is queued as a
        hunch or the hybrid. match is the hypothesis, or True for a queued one. (None, None) when p is new."""
        x = next((x for x in self.hyps if x.params == p), None)
        if x:
            return x, (f"that exact config, {params_text(p)}, already ran as {x.id} on {x.branch} ({self.outcome(x)})" if x.judgment
                       else f"that exact config, {params_text(p)}, is being measured right now as {x.id} on {x.branch}")
        q = next((bid for bid, hint in self.hints.items() if hint.params == p), None)
        if q:
            return True, f"that exact config, {params_text(p)}, is already queued on {q} for {self.hints[q].ref}"
        if self.hybrid and self.hybrid["state"] in ("proposed", "approved") and self.hybrid["params"] == p:
            return True, f"that exact config, {params_text(p)}, is the pending hybrid of {' and '.join(self.hybrid['branches'])}"
        return None, None

    def hunch(self, b, text, cite, ref, lens, delta):
        """A typed hunch gets exactly one plan_change, when it arrives: whether it is a retest, and either what will run (built in code on
        the branch's current best params, in a reserved slot the cap does not count) or why it will not. Never dropped silently."""
        quoted = f"Because the {lens.capitalize()} lens sent \"{text[:120]}\""
        if self.stopped:
            self.log.emit("plan_change", help=ref, text=f"{quoted}, nothing is tested: the session was stopped.")
            return
        p, e, why, dup = self.plan(b, delta) if delta else (None, None, None, None)
        if dup:
            # the exact config is already logged or on its way, so the hunch's outcome is that run's
            if isinstance(dup, Hyp):
                self.hunches[ref] = dup.id
            else:
                self.hunches[ref] = f"not run: {why}"
            self.log.emit("plan_change", help=ref, text=f"{quoted}, I am not running it again on {b.id} because {why}.")
            return
        if not e:
            p, why = None, None  # a first test goes to the model, which can fit the rest of the config around the change
        sp = e and (f", {e.speedup:.2f}x vs {'naive' if e.ref == 'naive' else 'its parent'}" if e.speedup else "")
        mixed = delta and not e and self.tried_with(delta)
        status = (f"This is a retest: it was already tested in {e.id} ({e.judgment.verdict}{sp}) on {e.params.block_m}x{e.params.block_n}x{e.params.block_k} tiles"
                  if e else f"This is not a retest: it has not been tested in isolation, {mixed[-1].id} changed it together with "
                            f"{', '.join(k for k in self.knobs(mixed[-1]) if k not in delta)}" if mixed
                  else "This is not a retest, nothing in the session has tested it yet")
        if why:
            self.hunches[ref] = f"not run: {why}"
            self.log.emit("plan_change", help=ref, text=f"{quoted}, I am not testing it because {why}. {status}.")
            return
        self.hints[b.id] = Hint(text, f"{cite}, a retest of {e.id} on the current best tiles" if e else cite, ref, delta, p, e, self.start_point(b.id)[0])
        self.reserved.add(ref)
        tried = [h for h in self.hyps if self.branches[h.branch].lens == "speed"]
        when = ("after the scheduled FP16 variant" if b.lens == "speed" and not tried and self.seed(b, {json.dumps(h.params.dict(), sort_keys=True) for h in self.hyps}) and not self.base_capped()
                else "next" if b.status == "running" else "as soon as it resumes")
        bound = self.bounds(b.lens)
        start = self.start_point(b.id)[0]
        what = (f"so the run changes only {change(start, p)} on the current best {tiles_text(start)}" if e
                else "so the agent turns it into the next hypothesis on that branch")
        self.log.emit("plan_change", help=ref, text=f"{quoted}, I am testing it {when} on {b.id}, judged against max error under {sci(bound['standard'])}, "
                                                    f"the {bound_owner(bound)} bound. {status}, {what}.")

    def rehome_hints(self):
        for bid, hint in list(self.hints.items()):
            text, ref = hint.text, hint.ref
            b = self.branches[bid]
            if b.status in ("running", "waiting_on_human"):
                continue
            del self.hints[bid]
            if hint.delta is None:
                continue  # a bound applies to every branch of its lens anyway, nothing was left untested
            others = [x for x in self.branches.values() if x.lens == b.lens and x.status in ("running", "waiting_on_human")]
            # the plan_change was logged when the hunch arrived, a move afterwards is a log line
            if others:
                self.hints[others[-1].id] = hint
                self.log.emit("agent_log", text=f"{bid} closed ({b.status}) before testing \"{text[:120]}\" ({ref}), so it runs next on {others[-1].id}.")
            else:
                self.reserved.discard(ref)
                self.hunches[ref] = f"not run: {bid} closed ({b.status}) before it and no other {b.lens} branch was open"
                self.log.emit("agent_log", text=f"{bid} closed ({b.status}) before testing \"{text[:120]}\" ({ref}) and no other {b.lens} branch is open. "
                                                "Send it again as a directive to open one.")

    def reevaluate(self, d):
        """Re-judges every measured run under the bound now active on its branch, so each verdict shows the bound in force. A run that
        flips from confirmed to rejected rejects its branch, with a post-mortem and a pivot fork."""
        speed = [x["id"] for x in self.directives if x["lens"] == "speed"]
        flipped = []
        for h in self.hyps:
            j = h.judgment
            bound = self.bounds(self.branches[h.branch].lens)
            if not (j and h.metrics.latency_ms) or h.bounds[-1] == bound:
                continue
            new = judge(h.prediction, h.metrics, h.speedup, h.noise, bound["standard"], bound["adversarial"])
            self.set_verdict(h, new, bound)
            if j.verdict == "confirmed" and new.verdict == "rejected":
                # only the first flip on a branch rejects it and forks, a second one on the same branch gets its post-mortem
                first = self.branches[h.branch].status != "rejected"
                if first:
                    self.branches[h.branch].status = "rejected"
                    self.log.emit("branch_status", branch=h.branch, status="rejected")
                    self.pivot(h)  # before the post-mortem, so a run that finishes meanwhile already has a successor
                if speed and first:
                    self.conflicts += 1
                    batches = [c.name.split()[0] for c in new.checks if c.passed is False and "error" in c.name]
                    self.log.emit("conflict_detected", directives=speed + [d["id"]],
                                  reason=f"The best candidate for the Speed directive ({h.id}) violates the Precision bound on the {' and '.join(batches)} "
                                         f"batch{'es' if len(batches) > 1 else ''}. Forking instead of overwriting either directive.")
                self.post_mortem(h)
                self.summarize(h, self.branches[h.branch], f"Fork: keep what worked in {h.id} and drop {ruled_out(h.params, self.start_point(h.branch)[0])}.")
                self.export_failure(h, self.branches[h.branch])
                flipped.append(h)
            elif new.verdict == "rejected" and j.verdict != "rejected":
                self.post_mortem(h)
        if flipped:
            self.refresh_helps(flipped)
        self.maybe_hybrid()

    # ---- branches ----

    def fork(self, parent, lens, label, reason):
        base = LENS_IDS.get(lens, "branch")
        n = sum(b.id.startswith(base) for b in self.branches.values())
        b = Branch(base if n == 0 else f"{base}{n + 1}", [parent], label, lens)
        self.branches[b.id] = b
        self.log.emit("branch_created", branch=b.dict(), reason=reason)
        return b

    def pivot(self, h):
        b = self.branches[h.branch]
        if b.lens == "both" or sum(x.lens == b.lens for x in self.branches.values()) >= MAX_PER_LENS:
            return
        base = self.start_point(b.id)[0]
        # one open branch per lens family: an open branch of the lens takes the pivot instead of a duplicate fork
        open_ = [x for x in self.branches.values() if x.lens == b.lens and x.id != b.id and x.status in ("running", "waiting_on_human")]
        if open_:
            self.log.emit("agent_log", text=f"{open_[-1].id} is already open on the {b.lens.capitalize()} lens, so it continues the pivot from {h.id} "
                                            f"(drop {ruled_out(h.params, base)}) instead of a new branch.")
            return
        dropped = pivot_keys(h.params, base)
        new = self.fork(b.id, b.lens, f"{'Speed' if b.lens == 'speed' else 'Precision'} lens, pivot from {h.id}",
                        f"Pivot from the {h.id} post-mortem: drop {ruled_out(h.params, base)} and keep the rest.")
        new.pivot = replace(h.params, **{k: getattr(base, k) for k in dropped})

    def next_branch(self):
        if self.capped() or self.stopped:
            return None
        live = [b for b in self.branches.values() if b.status == "running"]
        if self.base_capped():
            live = [b for b in live if b.id in self.owed()]
        return min(live, key=lambda b: b.iters) if live else None

    def lineage(self, bid):
        seen, stack = [], [bid]
        while stack:
            x = stack.pop()
            if x not in seen:
                seen.append(x)
                stack += self.branches[x].parents
        return seen

    def best_run(self, branch=None):
        """The best confirmed run, ranked by absolute latency. The only ranking used for the session's best, its parents and the hybrid."""
        ok = [h for h in self.hyps if h.judgment and h.judgment.verdict == "confirmed" and not h.judgment.provisional
              and (branch is None or h.branch in self.lineage(branch))]
        return min(ok, key=lambda h: h.metrics.latency_ms) if ok else None

    def fastest(self):
        """The session's result: the lowest latency among measured runs inside every active bound, whatever their verdict, so a run that
        missed its predicted gain but is the fastest correct one is reported (with its verdict). Branch start points stay on best_run."""
        ok = [h for h in self.hyps if h.judgment and h.metrics and h.metrics.latency_ms and self.failed_bound(h) is None]
        return min(ok, key=lambda h: h.metrics.latency_ms, default=None)

    def start_point(self, bid):
        best = self.best_run(bid)
        return (best.params, best.metrics) if best else (NAIVE, self.naive)

    # ---- one iteration ----

    def iterate(self, b: Branch):
        b.iters += 1
        h = self.propose(b)
        if self.stopped:
            return
        if not h:
            self.close(b, "done")
            return
        self.cite(h, self.start_point(b.id)[0])
        if b.status == "rejected":
            self.drop(h, b)
            return
        h.claim, h.note = self.scrub(h.claim), h.note and self.scrub(h.note)
        if b.id in self.defaulted:
            flag = "flagged: default continuation, no lens directive arrived"
            h.note = f"{h.note} ({flag})" if h.note else flag.capitalize()
        self.hyps.append(h)
        self.log.emit("hypothesis_proposed", hypothesis=h.dict())
        best = self.best_run(b.id)
        parent = h.base or (best.params if best else None)
        ref = "parent" if h.prediction["vs"] == "parent_best" and parent else "naive"
        self.measure(h, parent if ref == "parent" else None)
        h.ref = ref
        h.speedup = h.metrics.speedups.get(ref)
        h.noise = max(self.band, h.metrics.spreads.get(ref, 0.0))
        bound = self.bounds(b.lens)
        self.set_verdict(h, judge(h.prediction, h.metrics, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
        v = h.judgment.verdict
        if h.metrics.latency_ms is None:
            self.post_mortem(h)
        b.fails = b.fails + 1 if h.metrics.latency_ms is None else 0
        # no help is asked for, and the branch stops, once the cap has no hypothesis left that is not already promised elsewhere
        spent = self.max_hyps is not None and self.room()[0] == 0 and not self.promised(b.id)
        if b.fails >= 2 and not any(x["branch"] == b.id and open_help(x) for x in self.helps.values()) and not spent:
            target = b.lens if b.lens != "both" else "speed"
            self.summarize(h, b, f"Ask the {target.capitalize()} lens how to continue after two failed runs in a row.")
            self.request_help(h, b, target, failed=True)
            return

        # Only correctness failures end a branch with a post-mortem and a pivot. A missed speedup is a normal miss.
        unsafe = v == "rejected" and any(c.passed is False and "error" in c.name for c in h.judgment.checks)
        if unsafe and b.lens != "both":
            if b.status != "rejected":
                b.status = "rejected"
                self.log.emit("branch_status", branch=b.id, status="rejected")
            self.post_mortem(h)
            self.summarize(h, b, f"Fork: drop {ruled_out(h.params, self.start_point(b.id)[0])} and keep the rest.")
            self.export_failure(h, b)
            self.pivot(h)
            return
        if v in ("rejected", "inconclusive") and h.id not in self.pms:
            self.post_mortem(h)

        b.misses = 0 if v == "confirmed" else b.misses + 1
        if v == "inconclusive" and not any(x["branch"] == b.id for x in self.helps.values()) and not spent:
            target = b.lens if b.lens != "both" else "speed"
            self.summarize(h, b, f"Ask the {target.capitalize()} lens whether to keep tuning or change approach.")
            self.request_help(h, b, target)
            return
        budget = (b.budget or (self.main_iters if b.id == "main" else self.branch_iters)) + self.extra.get(b.id, 0)
        # two misses in a row end a branch, unless a lens owner was promised more iterations on it
        done = b.iters >= budget or (b.misses >= 2 and not self.promised(b.id)) or spent
        if spent and b.id != "merge":
            nxt = "Stop this branch: no hypothesis budget remains for more iterations."
        elif v == "rejected":
            nxt = ("Stop this branch and wait for a lens directive." if done
                   else f"Try a different change on this branch, avoiding what the {h.id} post-mortem ruled out." if h.id in self.pms
                   else "Try a different change on this branch.")
        elif v == "confirmed" and h.judgment.provisional:
            nxt = "Hold and flag the adversarial error until a precision lens is attached."
        elif done:
            nxt = "Stop this branch and wait for a lens directive." if b.id == "main" else "Stop this branch, the iteration budget is spent."
        else:
            nxt = "Refine from the best configuration on this branch." if v == "confirmed" else "Try a different change, this one did not clearly beat the noise band."
        if b.id == "merge":
            nxt = "Keep the hybrid, the lenses' changes add up." if v == "confirmed" else "Keep the faster parent, the hybrid did not beat it."
        self.summarize(h, b, nxt)
        if v == "rejected":
            self.export_failure(h, b)
        if b.id == "merge" and v == "confirmed":
            for pid in b.parents:
                self.close(self.branches[pid], "merged")
        if done:
            self.close(b, "done")
        self.maybe_hybrid()

    def drop(self, h, b):
        """A proposal whose branch was rejected while Papiers was read never runs. Its hint and reserved slot go back, and
        rehome_hints moves the hint to the lens's open branch."""
        hint, self.taken = self.taken, None
        if hint:
            self.hints.setdefault(b.id, hint)
            if h.bonus:
                self.reserved.add(hint.ref)
            if self.hunches.get(hint.ref) == h.id:
                del self.hunches[hint.ref]
        succ = self.successor(b)
        self.log.emit("agent_log", text=f"{b.id} was rejected while its next hypothesis was being prepared, so it is not run"
                                        + (f" and {succ.id} continues the lens." if succ else "."))

    # ---- help requests ----

    def pending(self):
        return any(open_help(x) for x in self.helps.values()) or bool(self.hybrid and self.hybrid["state"] == "proposed")

    def successor(self, b):
        """The open branch that continues a rejected one: its pivot (the newest open child on the same lens), or the open branch of the
        lens that took the pivot instead of a duplicate fork."""
        same = [x for x in self.branches.values() if x.lens == b.lens and x.id != b.id and x.status in ("running", "waiting_on_human")]
        kids = [x for x in same if b.id in x.parents]
        return (kids or same or [None])[-1]

    def live(self, bid):
        """The branch a help answer applies to: the branch itself, or its successor once it was rejected. None when nothing continues it."""
        b = self.branches[bid]
        return (self.successor(b) if b.status == "rejected" else b)

    def evidence(self, lineage):
        return [f"{x.run}: {x.metrics.latency_ms:.3f} ms, " + ("provisionally confirmed" if x.judgment.provisional else x.judgment.verdict)
                + (f", {c.observed}" if (c := next((c for c in x.judgment.checks if c.name.startswith("speedup")), None)) else "")
                for x in reversed(self.hyps) if x.branch in lineage and x.metrics and x.metrics.latency_ms and x.judgment][:3]

    def refresh_helps(self, flipped):
        """After a re-judge flipped these runs, a pending request on a rejected branch moves to its successor (or is withdrawn when none
        is open), and any pending request whose evidence lists a flipped run is reissued with the current verdicts. Same id, so an answer
        still lands, and at most one reissue per request per re-judge, only when the request changed."""
        for hid, x in self.helps.items():
            if not open_help(x):
                continue
            b = self.branches[x["branch"]]
            moved = b.status == "rejected"
            hit = next((h for h in flipped if any(e.startswith(f"{h.run}:") for e in x["help"]["evidence"])), None)
            if not moved and not hit:
                continue
            h = next((h for h in flipped if h.branch == b.id), None) or hit or flipped[-1]
            succ = self.live(b.id)
            if succ is None:
                x["cancelled"] = True
                self.log.emit("help_cancelled", help=hid, reason=f"{b.id} was rejected when {h.id} flipped under the Precision bound and no branch continues it, "
                                                                 "so there is nothing left to tune.")
                continue
            req = dict(x["help"])
            if not x["failed"]:
                req["evidence"] = self.evidence(x["lineage"])
            if moved:
                req["question"] = (f"{req['question']} {b.id} was rejected when {h.id} flipped under the Precision bound, so an answer now applies to {succ.id}, "
                                   "which keeps running meanwhile.")
                req["blocks"] = succ.id
                x["branch"] = succ.id
            if req == x["help"]:
                continue
            x["help"], x["t"] = req, self.log.clock()
            self.log.emit("help_requested", help=req)

    def request_help(self, h: Hyp, b: Branch, target, failed=False):
        lineage = self.lineage(b.id)
        # a re-judge can reject b while h is measured: the request then applies to the successor, which keeps running, and b stays rejected
        live = self.live(b.id)
        if live is None:
            self.log.emit("agent_log", text=f"{b.id} was rejected while {h.id} ran and no branch continues it, so I am not asking for help on it.")
            return
        hid = f"hq{len(self.helps) + 1}"
        if failed:
            runs = [x for x in reversed(self.hyps) if x.branch == b.id and x.metrics and x.metrics.latency_ms is None][:2]
            question = f"Two configs in a row failed to compile or run on {b.id}. Should I go back to the best configuration and try smaller tiles, or change approach?"
            evidence = [f"{x.run}: {x.params.block_m}x{x.params.block_n}x{x.params.block_k}, {x.params.warps} warps, {x.params.stages} stages: "
                        f"{(x.metrics.error or 'failed').strip().splitlines()[-1][:120]}" for x in runs]
            options = [{"id": "tile_search", "label": "Go back to the best config and try smaller tiles"},
                       {"id": "accept", "label": "Stop this branch"}, {"id": "other", "label": "Suggest something else"}]
        else:
            speed = next(c for c in h.judgment.checks if c.name.startswith("speedup"))
            question = f"Branch {b.id} reached {speed.observed.split(' ')[0]} against my {speed.predicted} prediction, inside the noise band. Should I keep tuning or change approach?"
            evidence = self.evidence(lineage)
            options = [{"id": "tile_search", "label": "Keep tuning tiles and warps"},
                       {"id": "accept", "label": f"Accept {speed.observed.split(' ')[0]} and stop this branch"},
                       {"id": "other", "label": "Suggest something else"}]
        if live is not b:
            question += f" {b.id} was rejected while {h.id} ran, so an answer applies to {live.id}, which keeps running meanwhile."
        n, short = self.room()
        req = {
            "id": hid, "target": target,
            "question": question,
            "evidence": evidence,
            "options": options,
            "default": "tile_search", "blocks": live.id, "deadline_s": int(self.help_deadline_s),
            "what_changes": (f"Keep tuning adds {'two more iterations' if n == 2 else 'one more iteration'} on this branch"
                             + (f" ({short})" if short else "") + ", accept closes it, and a suggestion becomes the next hypothesis to test."),
        }
        self.helps[hid] = {"help": req, "branch": live.id, "t": self.log.clock(), "answered": False, "expired": False, "cancelled": False,
                           "lineage": lineage, "failed": failed}
        self.log.emit("help_requested", help=req)
        if live is not b:
            return
        b.status = "waiting_on_human"
        self.log.emit("branch_status", branch=b.id, status="waiting_on_human")
        others = [x.id for x in self.branches.values() if x.status == "running"]
        self.log.emit("agent_log", text=f"Branch {b.id} is blocked on the {target.capitalize()} lens ({hid}). "
                                        + (f"Continuing on {', '.join(others)}." if others else "Nothing else is runnable, waiting for the answer or the deadline."))

    def check_deadlines(self):
        x = self.hybrid
        if x and x["state"] == "proposed" and self.log.clock() - x["t"] >= self.help_deadline_s:
            x["state"] = "expired"
            self.log.emit("agent_log", text=f"No approval for the hybrid of {' and '.join(x['branches'])} within {self.help_deadline_s:.0f} s, so it is not run. "
                                            "Approving it later still runs it.")
        for hid, x in self.helps.items():
            if not open_help(x) or self.log.clock() - x["t"] < x["help"]["deadline_s"] or self.capped():
                continue
            x["expired"] = True
            b = self.live(x["branch"])
            if b is None:
                continue
            n, short = self.room()
            if n == 0:
                self.log.emit("plan_change", help=hid, text=f"No answer to {hid} by the deadline, and {short}, so no iterations fit and I am closing {b.id}.")
                self.close(b, "done")
                continue
            self.promise(hid, b, n)
            self.log.emit("plan_change", help=hid, text=f"No answer to {hid} by the deadline, so I am proceeding on the default assumption: keep tuning tiles and warps on {b.id} "
                                                        f"for {'two more iterations' if n == 2 else 'one more iteration'}"
                                                        + (f", {short}. " if short else ". ")
                                                        + "Results on this branch are flagged assumption pending until an answer arrives.")
            self.reopen(b)

    def promised(self, bid):
        """Iterations a plan_change promised on this branch that have not run yet."""
        return sum(max(0, pr["n"] - (sum(h.branch == bid for h in self.hyps) - pr["start"])) for pr in self.promises if pr["branch"] == bid)

    def promise(self, ref, b, n):
        self.extra[b.id] = self.extra.get(b.id, 0) + n
        self.promises.append({"ref": ref, "branch": b.id, "n": n, "start": sum(h.branch == b.id for h in self.hyps)})

    def room(self, want=2):
        """(iterations a help answer can promise, up to want, what the cap has left as a phrase). The cap minus hypotheses counted against
        it and iterations already promised on any branch; (want, "") when the session is uncapped."""
        if self.max_hyps is None:
            return want, ""
        used = sum(not h.bonus for h in self.hyps)
        owed = sum(self.promised(bid) for bid in {pr["branch"] for pr in self.promises})
        n = max(0, min(want, self.max_hyps - used - owed))
        return n, f"the {self.max_hyps} hypothesis cap has {self.max_hyps - used} left" + (f" and {owed} of those are already promised" if owed else "")

    def answer(self, cmd):
        x = self.helps.get(cmd["help"])
        if x and x["answered"] and x.get("cmd", {}).get("option") == cmd["option"]:
            self.note(f"duplicate answer to {cmd['help']} ignored")  # the same answer delivered twice, for example after a reconnect
            return
        if not x or x["answered"] or x["cancelled"] or cmd["option"] not in {o["id"] for o in x["help"]["options"]}:
            self.log.emit("agent_log", text=f"Ignored an answer to {cmd['help']}: unknown request, unknown option, withdrawn, or already answered.")
            return
        x["answered"], x["cmd"] = True, cmd
        note = (cmd.get("note") or "").strip()
        self.log.emit("help_answered", help=cmd["help"], option=cmd["option"], by=cmd["by"], **({"note": note} if note else {}))
        late = "The answer arrived after the deadline, so I am revising the default. " if x["expired"] else ""
        lens = cmd["by"].capitalize()
        b = self.live(x["branch"])
        if b is None:
            text = f"{late}{x['branch']} was rejected and no branch continues it, so I am not reopening it. The answer is recorded."
            self.log.emit("plan_change", help=cmd["help"], text=text)
            self.export_answer(x["help"], cmd, text)
            return
        if b.id != x["branch"]:
            late += f"{x['branch']} was rejected, so the answer applies to its successor {b.id}. "
            x["branch"] = b.id
        b.misses = 0
        if cmd["option"] == "accept":
            text = f"{late}Because the {lens} lens accepts the current result, I am closing {b.id} and keeping its best configuration."
            self.log.emit("plan_change", help=cmd["help"], text=text)
            if b.status in ("running", "waiting_on_human"):
                self.close(b, "done")
        else:
            other = cmd["option"] == "other" and note
            # a late answer revises the iterations the deadline default already promised, it adds none
            if x["expired"]:
                n = next((pr["n"] for pr in self.promises if pr["ref"] == cmd["help"]), 0)
                short = "the deadline default found no room" if n == 0 else ""
            else:
                n, short = self.room()
            if n == 0:
                text = (f"{late}The {lens} lens wants {'its suggestion tested' if other else 'more tile search'} on {b.id}, but {short}, so no iterations fit "
                        f"and I am closing {b.id} with its best configuration.")
                self.log.emit("plan_change", help=cmd["help"], text=text)
                self.close(b, "done")
                self.export_answer(x["help"], cmd, text)
                return
            more = "two more iterations" if n == 2 else "one more iteration"
            fewer = f" {short[0].upper()}{short[1:]}." if short else ""
            if other:
                delta = hyp.hunch_change(note) or {}
                p, e, why, dup = self.plan(b, delta) if delta else (None, None, None, None)
                if not e and not dup:
                    p, why = None, None
                cite = f"Suggested by the {lens} lens in answer to {cmd['help']}"
                self.hints[b.id] = Hint(note, f"{cite}, a retest of {e.id} on the current best tiles" if e else cite, cmd["help"], delta, p, e,
                                        self.start_point(b.id)[0])
                text = (f"{late}Because the {lens} lens suggests \"{note}\", I am testing that next on {b.id} instead of more tile search, still under the active error bound."
                        + (f" It was already tested in {e.id} ({e.judgment.verdict}), so this is a retest on the current best tiles." if e else "")
                        + (f" As a fixed change it cannot run ({why}), so the agent picks the closest valid version." if why else "") + fewer)
                if why:
                    self.hints[b.id].params = None
            else:
                text = f"{late}Because the {lens} lens wants more tile search, I am giving {b.id} {more}.{fewer}"
            if not x["expired"]:
                self.promise(cmd["help"], b, n)
            self.log.emit("plan_change", help=cmd["help"], text=text)
            self.reopen(b)
        self.export_answer(x["help"], cmd, text)

    # ---- findings export ----

    def export(self, files):
        """files: [(kind, title, name, markdown)]. Written under the session folder, then announced as exported."""
        d = self.log.path.parent / "exports"
        d.mkdir(parents=True, exist_ok=True)
        for _, _, name, text in files:
            (d / name).write_text(text)
        self.log.emit("findings_exported", items=[{"kind": k, "title": t, "file": f"exports/{name}"} for k, t, name, _ in files])

    def export_failure(self, h, b):
        files = [("summary", f"Branch summary: {b.id}", f"summary-{b.id}.md", ex.summary_md(self.summaries[b.id]))]
        if h.id in self.pms:
            files.insert(0, ("negative_result", f"{h.id} rejected: {h.judgment.reason}"[:120], f"{h.id}-negative-result.md",
                             ex.negative_md(h, self.pms[h.id], h.bounds[-1])))
        self.export(files)

    def export_answer(self, req, cmd, plan):
        label = next(o["label"] for o in req["options"] if o["id"] == cmd["option"])
        self.export([("answer", f"{cmd['by'].capitalize()} lens answer to {req['id']}: {label}", f"{req['id']}-answer.md", ex.answer_md(req, cmd, plan))])

    def reopen(self, b):
        if b.status not in ("running", "rejected"):  # a rejected branch is never reopened, its successor continues it
            b.status = "running"
            self.log.emit("branch_status", branch=b.id, status="running")

    def close(self, b, status):
        if b.status == "rejected":
            return
        b.status = status
        self.log.emit("branch_status", branch=b.id, status=status)

    def bounds(self, lens):
        """The contract Bound for a branch of this lens. A Precision directive applies to every branch, the Speed tolerance only to Speed branches."""
        prec = [d["threshold"] for d in self.directives if d["lens"] == "precision"]
        if prec:
            return {"standard": min(prec), "adversarial": min(prec), "set_by": "precision"}
        speed = [d["threshold"] for d in self.directives if d["lens"] == "speed"]
        if lens == "speed" and speed:
            return {"standard": speed[-1], "adversarial": None, "set_by": "speed"}
        return {"standard": BOUND, "adversarial": None, "set_by": "default"}

    def propose(self, b: Branch):
        start, metrics = self.start_point(b.id)
        tried_all = {json.dumps(h.params.dict(), sort_keys=True) for h in self.hyps}
        self.taken = None
        if b.id == "merge" and self.hybrid and self.hybrid["state"] == "approved":
            return self.hybrid_hyp(b)
        # the seeded FP16 variant is the first Speed lens hypothesis, ahead of a pending hunch
        seed = None if self.base_capped() else self.seed(b, tried_all)
        if seed:
            return seed
        hint = self.taken = self.hints.pop(b.id, None)
        if hint and hint.params:
            built = self.built(b, hint)
            if built:
                return built
            hint = None

        directives = [d for d in self.directives if b.lens in ("both", d["lens"])]
        planned = facet(start, replace(start, **{k: v for k, v in hint.delta.items() if k in start.dict()})) if hint and hint.delta else None
        notes = self.papiers(planned or QUERIES[b.lens])
        messages = hyp.prompt(b.dict(), directives, {"params": start.dict(), "metrics": metrics.contract()}, self.ledger(), self.avoid,
                              ([f"Suggestion from a human lens owner, test this next: {hint.text}"] if hint and hint.delta is not None else [])
                              + ([f"This branch pivots to {params_text(b.pivot)}. Keep those parameters unless your claim names every one you change."]
                                 if b.pivot else [])
                              + [f"Papiers {n.get('kind') or 'item'} from \"{n.get('document_title') or n['title']}\": {n['excerpt']}" for n in notes],
                              self.bounds(b.lens)["standard"], self.headroom,
                              self.refused_lines())
        why, first = "", ""
        for attempt in range(3):
            text = self.ask_or_wait(messages, f"{b.id}-{b.iters}-{attempt}", b.id)
            if text is None or b.status == "rejected":
                # a re-judge can reject b while the model answers: the hint goes back and moves to the lens's open branch
                if hint:
                    self.hints[b.id] = hint
                self.taken = None
                if text is not None:
                    self.log.emit("agent_log", text=f"{b.id} was rejected while its next hypothesis was being proposed, so nothing more runs on it.")
                return None
            h, why = hyp.parse(text)
            refused = h and self.vet(start, h["params"], h["claim"])
            missing = hint and hint.delta and h and not h["adjusted"] and [value_text(k, v) for k, v in hint.delta.items() if getattr(h["params"], k) != v]
            if missing and not refused:
                refused = f"the lens asked to test {', '.join(missing)}, which this proposal does not set"
            # a claim that misdescribes the change is sent back, on the last attempt it is rebuilt from the params instead
            wrong = h and not refused and not h["adjusted"] and hyp.claim_errors(h["claim"], start, h["params"])
            if wrong and attempt < 2:
                refused = f"the claim does not match the change it makes ({change(start, h['params'])}): {'; '.join(wrong)}"
            if refused:
                h, why = None, refused
                self.log.emit("agent_log", text=f"Refused a proposal on {b.id}: {why}.")
            if h:
                cite = hint.cite if hint else None
                claim = h["claim"]
                if h["adjusted"] or wrong:
                    # rebuilt from the params that run, keeping any run ids the model cited
                    cited = list(dict.fromkeys(re.findall(r"\bH\d+\b", claim)))
                    claim = claim_text(start, h["params"], h["prediction"], self.bounds(b.lens)) + (f" Revisits {', '.join(cited)}." if cited else "")
                    fix = h["adjusted"] or f"Claim rebuilt from the final params after three mismatched claims, the last one said {'; '.join(wrong)}"
                    cite = f"{cite}. {fix}" if cite else fix
                if b.pivot and any(not hyp.named(claim, k, getattr(h["params"], k)) for k in h["params"].dict()
                                   if getattr(h["params"], k) != getattr(b.pivot, k)):
                    claim += f" Relative to the pivot config ({params_text(b.pivot)}) it changes {change(b.pivot, h['params'])}."
                new = self.new_hyp(b, claim, h["prediction"], h["params"], cite)
                self.spend(new, hint)
                if hint and hint.delta is not None:
                    self.hunches[hint.ref] = new.id
                    if first:
                        self.log.emit("agent_log", text=f"My first reading of \"{hint.text[:120]}\" ({hint.ref}) was not testable ({first}), so {new.id} tests "
                                                        f"the closest valid version: {change(start, new.params)}.")
                return new
            first = first or why
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Invalid: {why}. Reply again with only the JSON object."}]
        p = hint and hint.delta and self.plan(b, hint.delta)[0]
        if p:
            # the model kept ignoring the hunch, so its change is built in code on the branch's best config
            pred = {"min_speedup": 1.02, "vs": "parent_best"}
            new = self.new_hyp(b, claim_text(start, p, pred, self.bounds(b.lens)), pred, p,
                               f"{hint.cite}. Built in code: the model's proposals did not set {', '.join(value_text(k, v) for k, v in hint.delta.items())}")
            self.spend(new, hint)
            self.hunches[hint.ref] = new.id
            self.log.emit("agent_log", text=f"The model did not turn \"{hint.text[:120]}\" ({hint.ref}) into a valid proposal ({why}), so {new.id} tests it as "
                                            f"built in code: {change(start, p)}.")
            return new
        if hint:
            self.reserved.discard(hint.ref)
        if hint and hint.delta is not None:
            self.hunches[hint.ref] = f"not run: no valid configuration ({why})"
            self.log.emit("agent_log", text=f"I could not turn \"{hint.text[:120]}\" ({hint.ref}) into a valid configuration on {b.id} ({why}), so it is not tested.")
        self.log.emit("agent_log", text=f"No valid hypothesis for branch {b.id} after 3 attempts ({why}). Closing the branch.")
        return None

    def outcome(self, h):
        if h.metrics.latency_ms is None:
            return "failed to run"
        sp = f", {h.speedup:.2f}x vs {'naive' if h.ref == 'naive' else 'its parent'}" if h.speedup else ""
        return f"{'provisionally confirmed' if h.judgment.provisional else h.judgment.verdict}{sp}"

    def ledger(self):
        """Every judged run on any branch, for the proposer: what changed, the verdict, the same-window speedup, registers and spills."""
        return [{"id": h.id, "branch": h.branch, "params": h.params.dict(), "change": change(self.baseline(h)[0], h.params),
                 "outcome": self.outcome(h), "n_regs": h.metrics.n_regs, "n_spills": h.metrics.n_spills}
                for h in self.hyps if h.judgment]

    def hurts(self):
        """{(knob, value): hypothesis} for runs that changed only that knob from what they were timed against and failed, broke an error
        bound, or ran measurably slower than their parent. The latest such run wins."""
        out = {}
        for h in self.hyps:
            if not h.judgment:
                continue
            keys = [k for k in h.params.dict() if getattr(h.params, k) != getattr(self.baseline(h)[0], k)]
            bad = (h.metrics.latency_ms is None or any(c.passed is False and "error" in c.name for c in h.judgment.checks)
                   or (h.ref == "parent" and h.speedup is not None and h.speedup < 1 - h.noise))
            if len(keys) == 1 and bad:
                out[(keys[0], getattr(h.params, keys[0]))] = h
        return out

    def blamed(self, h):
        """(changes, lost) for a measured run: the (knob, from, to) changes it made, and whether it lost. A run loses when it breaks an
        active bound, which blames only its precision or GELU change when it made one, or when it is slower than what it was timed
        against. A smaller gain than predicted is not a loss."""
        base = self.baseline(h)[0]
        bad = self.failed_bound(h) is not None
        keys = pivot_keys(h.params, base) if bad else self.knobs(h)
        return [(k, getattr(base, k), getattr(h.params, k)) for k in keys], bad or self.gain(h) < 1.0

    def losing(self, hyps=None):
        """{(knob, from, to): [runs that lost with it]} for the losing directions: a change that lost on its own, or that lost in two or
        more runs together with other changes and never came with a gain inside every bound. Any branch counts. "On its own" treats
        block_m and block_n as one direction, the output tile."""
        lost, won, alone = {}, set(), set()
        for h in self.hyps if hyps is None else hyps:
            if not (h.judgment and h.metrics and h.metrics.latency_ms):
                continue
            changes, bad = self.blamed(h)
            if not bad:
                won.update(changes)
                continue
            for d in changes:
                lost.setdefault(d, []).append(h)
            if len({direction(k) for k, _, _ in changes}) == 1:
                alone.update(changes)
        return {d: xs for d, xs in lost.items() if d in alone or len(xs) >= 2 and d not in won}

    def isolated(self, d, xs):
        """The run that lost with change d and nothing outside its direction, or None when d only lost together with other changes."""
        return next((x for x in xs if {direction(k) for k, _, _ in self.blamed(x)[0]} == {direction(d[0])}), None)

    def refused_lines(self):
        return [f"{k} {a} to {b} (" + (f"lost alone in {e.id}, {self.outcome(e)}" if (e := self.isolated((k, a, b), xs))
                                       else f"lost in {', '.join(x.id for x in xs)}") + ")" for (k, a, b), xs in self.losing().items()]

    def vet(self, start: Params, p: Params, claim):
        """Why a proposal is refused before it runs, or None: an exact repeat, a losing direction (see losing), a rejected config with
        one more knob that already hurt, or a claim that sets a knob to a value shown to hurt without citing the run that showed it."""
        ran = next((x for x in self.hyps if x.params == p), None)
        if ran:
            return f"{params_text(p)} already ran as {ran.id} on {ran.branch} ({self.outcome(ran) if ran.judgment else 'running'})"
        lost = self.losing()
        changes = [(k, getattr(start, k), getattr(p, k)) for k in p.dict() if getattr(start, k) != getattr(p, k)]
        for d in changes:
            if d in lost and (e := self.isolated(d, lost[d])):
                return (f"it changes {d[0]} {d[1]} to {d[2]}, which {e.id} on {e.branch} already tested on its own and lost ({self.outcome(e)}); "
                        "that change is refused on every branch")
        hit = [d for d in changes if d in lost]
        if hit and len({direction(k) for k, _, _ in changes}) > 1:
            return ("it repeats changes that lost before without isolating them: "
                    + ", ".join(f"{k} {a} to {b} (lost in {', '.join(x.id for x in lost[(k, a, b)])})" for k, a, b in hit)
                    + "; test one of them alone or change something else")
        hurt = self.hurts()
        for x in self.hyps:
            if not (x.judgment and x.judgment.verdict == "rejected"):
                continue
            keys = [k for k in p.dict() if getattr(p, k) != getattr(x.params, k)]
            if len(keys) == 1 and (e := hurt.get((keys[0], getattr(p, keys[0])))):
                k = keys[0]
                return (f"it is the rejected {x.id} with only {k} {getattr(x.params, k)} to {getattr(p, k)}, a change {e.id} already showed hurts "
                        f"({self.outcome(e)})")
        for k in p.dict():
            e = hurt.get((k, getattr(p, k)))
            if getattr(start, k) != getattr(p, k) and e and not re.search(rf"\b{e.id}\b", claim):
                return (f"the claim sets {setting(p, k)}, which {e.id} showed hurts ({self.outcome(e)}), without citing {e.id}; "
                        f"cite it and say why this case differs")
        return None

    def built(self, b, hint):
        """A retest planned in code when its hunch arrived: the hunch's change on the params named in its plan_change."""
        ran = next((x for x in self.hyps if x.params == hint.params), None)
        if ran:
            self.reserved.discard(hint.ref)
            self.hunches[hint.ref] = f"not run: {ran.id} ran the same config first"
            self.log.emit("agent_log", text=f"\"{hint.text[:120]}\" ({hint.ref}) is not run: {ran.id} on {ran.branch} ran {params_text(hint.params)} first.")
            return None
        bound = self.bounds(b.lens)
        claim = (f"Retest of {hint.earlier.id}: {change(hint.base, hint.params)} on the current best {tiles_text(hint.base)} is at least 1.02x faster than the same config "
                 f"without it, with standard max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")
        h = self.new_hyp(b, claim, {"min_speedup": 1.02, "vs": "parent_best", "max_error": bound["standard"]}, hint.params, hint.cite)
        h.base = hint.base
        self.spend(h, hint)
        self.hunches[hint.ref] = h.id
        return h

    def correct_runs(self, branches):
        """Measured runs on these branches that pass every bound active now (the strict 1e-3 by default), fastest first."""
        out = []
        for h in self.hyps:
            if h.branch not in branches or not (h.metrics and h.metrics.latency_ms):
                continue
            bound = self.bounds(self.branches[h.branch].lens)
            if h.metrics.max_error < min(bound["standard"], BOUND) and (bound["adversarial"] is None or h.metrics.adv_error < bound["adversarial"]):
                out.append(h)
        return sorted(out, key=lambda h: h.metrics.latency_ms)

    def seed(self, b, tried_all):
        """The first hypothesis on the Speed lens is the harness-seeded FP16 accumulation variant of the best tiles so far (H0's
        when it is the best), judged against the Speed bound. Its claim is built from the bound and params, never by the model."""
        speed = [h for h in self.hyps if self.branches[h.branch].lens == "speed"]
        if b.lens != "speed" or speed:
            return None
        best = self.best_run(b.id)
        ok = self.correct_runs(self.lineage(b.id))
        base = best.params if best else ok[0].params if ok else None
        if base is None or base.acc != "fp32":
            return None
        p = replace(base, acc="fp16")
        if json.dumps(p.dict(), sort_keys=True) in tried_all:
            return None
        bound = self.bounds(b.lens)
        claim = (f"FP16 accumulation on {tiles_text(p)} with {p.gelu} GELU cuts latency by at least 1.20x against the same config with FP32 accumulation, "
                 f"with standard max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")
        h = self.new_hyp(b, claim, {"min_speedup": 1.2, "vs": "parent_best", "max_error": bound["standard"]}, p,
                         "Harness-seeded variant, not proposed by the model: low-precision accumulation tested under the active Speed bound.")
        h.base = base
        return h

    # ---- hybrid ----

    def knobs(self, h):
        """The parameters h changed from what it was timed against."""
        base = self.baseline(h)[0]
        return [k for k in h.params.dict() if getattr(h.params, k) != getattr(base, k)]

    def gain(self, h):
        """h's speedup over what it was timed against: the same-window parent ratio, or its baseline's latency over its own."""
        p, m = self.baseline(h)
        return h.speedup if h.speedup is not None and (h.ref == "parent" or p == NAIVE) else m.latency_ms / h.metrics.latency_ms

    def failed_bound(self, h):
        """The first active bound h's measured errors break, as text, or None. Every lens's bound counts, so a hybrid never inherits
        a change that only a looser lens accepted."""
        m = h.metrics
        for lens in ("speed", "precision", "both"):
            b = self.bounds(lens)
            if m.max_error >= b["standard"]:
                return f"standard error {sci(m.max_error)} is over the {bound_owner(b)} bound {sci(b['standard'])}"
            if b["adversarial"] is not None and m.adv_error >= b["adversarial"]:
                return f"adversarial error {sci(m.adv_error)} is over the {bound_owner(b)} bound {sci(b['adversarial'])}"
        return None

    def merge_why_not(self, h):
        """Why h's change cannot go into a hybrid, or None: it must have measured at least 1.00x over its parent and pass every active bound."""
        if not (h.metrics and h.metrics.latency_ms):
            return "it did not run"
        return (f"{self.gain(h):.2f}x vs its parent, under 1.00x" if self.gain(h) < 1.0 else self.failed_bound(h))

    def skip_hybrid(self, text):
        if text != self.skipped:  # logged once per distinct reason, since this runs after every iteration
            self.skipped = text
            self.log.emit("agent_log", text=text)

    def maybe_hybrid(self):
        """Proposes, once, the best confirmed config combined with the change one run on another branch made from its parent (the largest
        measured gain over its parent). Only a run at 1.00x or better over its parent that passes every active bound can give its change,
        and the best config has to pass them too. It does not need the lenses to disagree: once every branch shares a bound, the question
        is whether two improvements add up. Skipped, with the reason logged once, when nothing qualifies, the accumulator guard fails, or
        the combination already ran. Approval is the presenter's, and neither the proposal nor its run counts toward the cap."""
        best = self.best_run()
        if self.hybrid or not best:
            return
        own = self.lineage(best.branch)
        others = [h for h in self.hyps if h.judgment and h.branch not in own and h.branch != "merge" and self.knobs(h)
                  and any(getattr(best.params, k) != getattr(h.params, k) for k in self.knobs(h))]
        if not others:
            return
        if why := self.merge_why_not(best):
            self.skip_hybrid(f"No hybrid: the best config, {best.id} on {best.branch}, does not qualify ({why}).")
            return
        donors = [h for h in others if not self.merge_why_not(h)]
        if not donors:
            # a later run may still qualify, so the reason is logged when the session ends without a hybrid
            self.unmerged = ("No hybrid: no run on another branch measured at least 1.00x over its parent while passing every active bound ("
                             + "; ".join(f"{h.id} on {h.branch}, {change(self.baseline(h)[0], h.params)}: {self.merge_why_not(h)}" for h in others) + ").")
            return
        self.unmerged = ""
        donor = max(donors, key=self.gain)
        ks = self.knobs(donor)
        p = fit_stages(replace(best.params, **{k: getattr(donor.params, k) for k in ks}))
        per, limit = acc_per_thread(p), ACC_LIMIT[p.acc]
        ran = next((x for x in self.hyps if x.params == p), None)
        pair = (f"{best.id} on {best.branch} ({params_text(best.params)}) with {', '.join(setting(donor.params, k) for k in ks)} "
                f"from {donor.id} on {donor.branch}")
        why = (f"{params_text(p)} puts {per} {p.acc.upper()} accumulator elements on each of {p.warps * 32} threads, over {limit}" if per > limit
               else f"{params_text(p)} already ran as {ran.id} on {ran.branch}" if ran else None)
        if why:
            self.skip_hybrid(f"No hybrid of {pair}: {why}.")
            return
        text = (f"Combine the best confirmed config, {pair}, the largest gain on another branch that passes every active bound "
                f"({self.gain(donor):.2f}x over its parent). Candidate: {params_text(p)}, {per} accumulator elements per thread. "
                "It runs outside the hypothesis cap, and the result shows whether the two changes add up.")
        self.hybrid = {"branches": [best.branch, donor.branch], "params": p, "base": best.params, "faster": best, "text": text,
                       "t": self.log.clock(), "state": "proposed"}
        self.log.emit("hybrid_proposed", branches=self.hybrid["branches"], text=text)

    def approve_hybrid(self, branches):
        x = self.hybrid
        if not x or x["state"] not in ("proposed", "expired") or set(branches) != set(x["branches"]):
            self.note(f"ignored approve_hybrid {branches}: no matching open proposal")
            return
        x["state"] = "approved"
        self.reserved.add("hybrid")
        b = Branch("merge", list(x["branches"]), "Hybrid", "both", budget=1)
        self.branches[b.id] = b
        self.log.emit("branch_created", branch=b.dict(), reason=f"Hybrid approved: {change(x['base'], x['params'])}.")

    def hybrid_hyp(self, b):
        x = self.hybrid
        x["state"] = "done"
        bound = self.bounds(b.lens)
        claim = (f"The hybrid {params_text(x['params'])} is at least 1.02x faster than the faster parent, {x['faster'].id} at "
                 f"{x['faster'].metrics.latency_ms:.3f} ms, with standard max error under {sci(bound['standard'])}"
                 + (f" and adversarial under {sci(bound['adversarial'])}." if bound["adversarial"] else "."))
        h = self.new_hyp(b, claim, {"min_speedup": 1.02, "vs": "parent_best", "max_error": bound["standard"]}, x["params"],
                         f"Hybrid of {x['branches'][0]} and {x['branches'][1]}, approved by the presenter.")
        h.base = x["base"]
        h.bonus = True
        self.reserved.discard("hybrid")
        return h

    def papiers(self, query):
        """Reads Papiers once per query and logs each read: one event per returned item, or one without a note when nothing matched.
        The lock is released while the CLI runs, so commands still apply."""
        if not self.context or not query:
            return []
        if query not in self.read:
            try:
                with self.released():
                    got = self.context(query)
            except Exception as e:  # noqa: BLE001, Papiers is optional context, the loop runs without it
                self.read[query] = []
                self.log.emit("agent_log", text=f"Papiers search unavailable ({type(e).__name__}), continuing without reading context.")
                return []
            self.read[query] = [n | {"excerpt": re.sub(r"\s*\[\[src:[^\]]*\]\]", "", n["excerpt"])[:400]} for n in got[:2]]
            if not self.read[query]:
                self.log.emit("papiers_context", query=query)
            for n in self.read[query]:
                self.log.emit("papiers_context", query=query, note={k: v for k, v in n.items() if v is not None})
        return self.read[query]

    def scrub(self, text):
        """Replaces raw Papiers ids (cnv_..., hl_...) a model echoed into prose with the item's document title."""
        titles = {n["id"]: n.get("document_title") or n["title"] for got in self.read.values() for n in got}
        return PAPIERS_ID.sub(lambda m: f"\"{titles[m.group(0)]}\"" if m.group(0) in titles else "a Papiers item", text)

    def cite(self, h: Hyp, start: Params):
        """Before the run: reads Papiers for the change this hypothesis makes and cites the top item in its note. Nothing is cited when
        the change query finds nothing, even if the prompt had other context, so a citation is always about the change under test."""
        found = self.papiers(facet(h.base or start, h.params))
        if found:
            h.note = f"{h.note.rstrip('.')}. {cite_text(found[0])}" if h.note else cite_text(found[0])
            h.papiers = found[0]["id"]

    def ask_or_wait(self, messages, tag, branch):
        """Calls the fast model. On a timeout or API error the session shows it and waits for a retry command, or retries by
        itself after retry_s, so it never looks frozen. Returns None if the presenter stops the session."""
        while not self.stopped:
            try:
                return self.asking(messages, self.fast, tag)
            except Stopped:
                return None
            except Exception as e:  # noqa: BLE001, any model failure becomes a visible, retryable state
                kind = "llm_timeout" if "timeout" in type(e).__name__.lower() else "llm_error"
                self.log.emit("agent_error", kind=kind, branch=branch, retry_in_s=int(self.retry_s),
                              text=f"The hypothesis model failed ({type(e).__name__}: {str(e)[:120]}). Retrying in {self.retry_s:.0f} s or on retry.")
                self.retry.clear()
                with self.released():
                    self.retry.wait(self.retry_s)
        return None

    def new_hyp(self, b, claim, prediction, params, note):
        # the error bound in the prediction is the lens bound, the agent cannot loosen it
        bound = self.bounds(b.lens)
        return Hyp(f"H{len(self.hyps)}", b.id, claim, prediction | {"max_error": bound["standard"], "bound": bound}, params, note)

    def measure(self, h: Hyp, parent: Params | None):
        """Times the candidate interleaved with naive, eager and, when it has one, its parent best, all in one child process."""
        self.runs += 1
        h.run = f"r{self.runs}"
        self.log.emit("run_started", run=h.run, branch=h.branch, hypothesis=h.id)
        refs = {"naive": {"target": "triton", "params": NAIVE.dict()}, "eager": {"target": "eager"}}
        if parent:
            refs["parent"] = {"target": "triton", "params": parent.dict()}
        h.metrics = self.gpu_run({"target": "triton", "params": h.params.dict(), "refs": refs}, f"{self.session}/{h.id}-{h.run}")
        if h.metrics.timed_out:
            self.log.emit("agent_error", kind="harness_timeout", branch=h.branch,
                          text=f"{h.run} hit the {TIMEOUT_S} s harness timeout and was killed. It counts as a failed run.")
        elif h.metrics.error:
            self.log.emit("agent_log", text=f"{h.run} failed: {h.metrics.error[:300]}")
        self.log.emit("run_finished", run={"id": h.run, "branch": h.branch, "hypothesis": h.id, "params": h.params.dict(), "metrics": h.metrics.contract()},
                      **({"error": "timed out" if h.metrics.timed_out else h.metrics.error[:300]} if h.metrics.error else {}))

    def set_verdict(self, h: Hyp, j: Judgment, bound: dict):
        h.judgment = j
        h.history.append(j.verdict)
        h.bounds.append(bound)
        self.log.emit("verdict", hypothesis=h.id, verdict=j.verdict, provisional=j.provisional, reason=j.reason,
                      checks=[c.dict() for c in j.checks], bound=bound)

    # ---- explanation ----

    def asking(self, messages, model, tag):
        """self.ask in a worker thread, abandoned when the presenter stops the session. Raises Stopped then."""
        out = {}

        def work():
            try:
                out["text"] = self.ask(messages, model, tag)
            except Exception as e:  # noqa: BLE001, re-raised in the caller's thread
                out["error"] = e

        t = threading.Thread(target=work, daemon=True)
        t.start()
        with self.released():
            while t.is_alive():
                if self.stopped:
                    raise Stopped
                t.join(0.1)
        if "error" in out:
            raise out["error"]
        return out["text"]

    def ask_json(self, messages, model, tag, keys):
        """LLM prose for the explanation layer. Falls back to None on any API or format failure, the caller has a factual default."""
        try:
            text = self.asking(messages, model, tag)
            found = re.search(r"\{.*\}", text, re.DOTALL)
            d = json.loads(found.group(0)) if found else {}
        except Stopped:
            return None
        except Exception as e:  # noqa: BLE001, a slow or failing model must not stop the research loop
            self.note(f"explanation model unavailable for {tag} ({type(e).__name__}), using the factual template")
            return None
        return d if all(isinstance(d.get(k), (str, list)) for k in keys) else None

    def post_mortem(self, h: Hyp):
        m = h.metrics
        p = h.prediction
        expected = f"Speedup of at least {p['min_speedup']:.2f}x vs {'naive' if p['vs'] == 'naive' else 'the best so far'} with max error under {sci(p['max_error'])}."
        base = self.start_point(h.branch)[0]
        if m.latency_ms is None:
            found = self.papiers("tile size shared memory" if "shared memory" in (m.error or "") else facet(base, h.params))
            pm = {"hypothesis": h.id, "expected": expected, "observed": f"The run failed: {(m.error or 'no measurement').strip().splitlines()[-1][:200]}",
                  "sweep": []} | failure_cause(h.params, m)
        elif h.ref == "parent" and h.speedup < SLOW and not errs(h):
            pp = h.base or self.best_run(h.branch).params
            parent = next((x.metrics for x in self.hyps if x.params == pp and x.metrics and x.metrics.latency_ms), None)
            found = self.papiers("register pressure spills")
            pm = {"hypothesis": h.id, "expected": expected, "sweep": [],
                  "observed": f"{m.latency_ms:.3f} ms against {m.ref_ms['parent']:.3f} ms for the parent in the same window, {h.speedup:.2f}x (+/- {h.speedup * h.noise:.2f}x)."} \
                | register_cause(h.params, m, h.speedup, parent)
        elif not errs(h):
            bp, bm = self.baseline(h)
            found = self.papiers("register pressure spills" if m.n_spills else facet(bp, h.params))
            vs = "naive" if h.ref == "naive" else "the parent"
            pm = {"hypothesis": h.id, "expected": expected, "sweep": [],
                  "observed": f"{h.speedup:.2f}x vs {vs} (+/- {h.speedup * h.noise:.2f}x) at {m.latency_ms:.3f} ms, standard error {sci(m.max_error)}."} \
                | miss_cause(h.params, bp, m, bm, h.speedup, h.prediction, h.ref, h.noise if h.judgment.verdict == "inconclusive" else None)
        else:
            observed = f"Speedup of {h.speedup:.2f}x (+/- {h.speedup * h.noise:.2f}x), standard error {sci(m.max_error)}, adversarial error {sci(m.adv_error)}."
            s = Metrics() if self.stopped else self.gpu_run({"target": "sweep", "params": h.params.dict(), "ks": SWEEP_KS}, f"{self.session}/{h.id}-sweep")
            sweep = [{"k": k, "adv_error": s.errors[f"k{k}"]} for k in SWEEP_KS if f"k{k}" in s.errors]
            trend = (f"Adversarial error goes from {sci(sweep[0]['adv_error'])} at K={sweep[0]['k']} to {sci(sweep[-1]['adv_error'])} at K={sweep[-1]['k']}"
                     f"{f', above the strict {sci(BOUND)} bound at every K' if all(x['adv_error'] >= BOUND for x in sweep) else ''}." if sweep else "")
            # the query names the measured cause: an error that grows with K is an accumulation length effect
            rising = len(sweep) > 1 and sweep[-1]["adv_error"] > sweep[0]["adv_error"]
            found = self.papiers("accumulation error grows with length" if rising else facet(base, h.params))
            context = "".join(f"\nThe user's Papiers {n.get('kind') or 'note'} from \"{n.get('document_title') or n['title']}\": {n['excerpt'][:300]}"
                              + (f" (their note: {n['annotation']})" if n.get("annotation") else "") for n in found)
            fallback = {"cause": f"{h.judgment.reason} {trend}".strip(),
                        "pivot": f"Drop {ruled_out(h.params, base)}, keep the other parameters, and test whether the gain survives."}
            prose = self.ask_json([{"role": "system", "content": "You write terse, factual post-mortems for GPU kernel experiments. Use only the numbers given. Reply with only JSON: {\"cause\": \"<two sentences on the mechanism>\", \"pivot\": \"<one sentence on what to try next>\"}"},
                                   {"role": "user", "content": f"Error metric: max |y - ref| / max(|ref|, 1) against an FP32 reference. The strict bound is {sci(BOUND)}, the Speed lens tolerates {sci(hyp.SPEED_TOLERANCE)} on the standard batch only.\nHypothesis {h.id}: {h.claim}\nChange: {change(base, h.params)}\nExpected: {expected}\nObserved: {observed}\n"
                                                               f"Verdict reason: {h.judgment.reason}\nError vs K sweep on adversarial magnitudes: {json.dumps(sweep)}{context}"}],
                                  self.strong, f"pm-{h.id}", ("cause", "pivot")) or fallback
            pm = {"hypothesis": h.id, "expected": expected, "observed": observed, "cause": prose["cause"], "pivot": prose["pivot"], "sweep": sweep}
        note = self.best_note(found + self.papiers(facet(self.baseline(h)[0], h.params)), f"{pm['cause']} {pm['pivot']}")
        if note:
            pm["note"] = note["id"]
        pm["cause"], pm["pivot"] = self.scrub(pm["cause"]), self.scrub(pm["pivot"])
        if h.judgment.verdict == "rejected":
            self.avoid.append(f"{ruled_out(h.params, base)} ({h.id} rejected: {h.judgment.reason})")
        self.pms[h.id] = pm
        self.log.emit("post_mortem", post_mortem=pm)

    @staticmethod
    def best_note(items, text):
        """What a post-mortem cites: an annotated highlight whenever one was retrieved, the one whose words overlap the cause and pivot
        most, then the highest ranked in retrieval order. Without one, the item with the most overlap, a highlight first on a tie."""
        w = words(text)
        score = lambda n: len(w & words(" ".join(filter(None, (n.get("highlight_text"), n.get("annotation"), n["excerpt"])))))
        ranked = [(n.get("kind") == "highlight" and bool(n.get("annotation")), score(n), n.get("kind") == "highlight", -i, n) for i, n in enumerate(items)]
        return max(ranked, default=(None,) * 5, key=lambda x: x[:4])[4]

    def summarize(self, h: Hyp, b: Branch, next_step: str):
        """Every field is computed from the verdict, its checks and the run's measured spread, so it cannot contradict them."""
        m = h.metrics
        before_p, before = self.baseline(h)
        j = h.judgment
        moved = [] if m.latency_ms is None else [
            {"metric": "latency", "before": before.latency_ms, "after": m.latency_ms, "unit": "ms"},
            {"metric": "throughput", "before": before.tflops, "after": m.tflops, "unit": "TFLOPS"},
            {"metric": "max error", "before": before.max_error, "after": m.max_error, "unit": ""},
            {"metric": "adversarial error", "before": before.adv_error, "after": m.adv_error, "unit": ""},
        ]
        word = ("Provisionally confirmed" if j.provisional else j.verdict.capitalize())
        speed = next((c for c in j.checks if c.name.startswith("speedup")), None)
        verdict = f"{word}: " + (f"{speed.observed} against a predicted {speed.predicted}. " if speed else "") + j.reason
        confidence, reason, unsure, mind = assess(h)
        summary = {
            "branch": b.id,
            "hypothesis": h.id,
            "tried": f"{h.claim} ({change(before_p, h.params)})",
            "moved": moved,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_reason": reason,
            "uncertainties": unsure,
            "would_change_mind": mind,
            "next_step": next_step,
        }
        self.summaries[b.id] = summary
        self.summarized.add(h.id)
        self.log.emit("summary", summary=summary)

    def baseline(self, h: Hyp):
        """What h was timed against: its harness base when it has one (the seed, a retest, the hybrid), otherwise the branch best before it."""
        x = h.base and next((x for x in self.hyps if x is not h and x.params == h.base and x.metrics and x.metrics.latency_ms), None)
        return (x.params, x.metrics) if x else self.prior(h)

    def prior(self, h: Hyp):
        """The configuration this hypothesis started from, excluding itself."""
        ok = [x for x in self.hyps if x is not h and x.judgment and x.judgment.verdict == "confirmed" and not x.judgment.provisional
              and x.branch in self.lineage(h.branch) and self.hyps.index(x) < self.hyps.index(h)]
        best = min(ok, key=lambda x: x.metrics.latency_ms) if ok else None
        return (best.params, best.metrics) if best else (NAIVE, self.naive)
