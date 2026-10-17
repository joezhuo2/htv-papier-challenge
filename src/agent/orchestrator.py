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

    def dict(self):
        d = {"id": self.id, "branch": self.branch, "claim": self.claim, "prediction": self.prediction, "params": self.params.dict()}
        return d | ({"note": self.note} if self.note else {})


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
    return f"Context from Papiers {n.get('kind') or 'note'} {n['id']}"


def claim_text(start: Params, p: Params, pred, bound):
    """A claim built from the final parameters, used whenever the harness adjusted the proposed ones."""
    vs = "naive" if pred["vs"] == "naive" else "the best configuration on this branch"
    return (f"{change(start, p).capitalize()}, giving {params_text(p)}, is at least {pred['min_speedup']:.2f}x faster than {vs}, "
            f"with standard max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")


def change(a: Params, b: Params):
    diff = [f"{k} {getattr(a, k)} to {getattr(b, k)}" for k in a.dict() if getattr(a, k) != getattr(b, k)]
    return ", ".join(diff) or "no parameter change"


def setting(p: Params, k):
    v = getattr(p, k)
    return f"{v.upper()} accumulation" if k == "acc" else f"{v} GELU" if k == "gelu" else f"{k} {v}"


def ruled_out(p: Params, base: Params):
    keys = [k for k in p.dict() if getattr(p, k) != getattr(base, k) and k in ("acc", "gelu")] or [k for k in p.dict() if getattr(p, k) != getattr(base, k)]
    return ", ".join(setting(p, k) for k in keys) or f"the configuration {params_text(p)}"


def hybrid_guard(p: Params, area):
    """Why a hybrid candidate would overflow the registers or shared memory, or None when it is within bounds."""
    per, limit = acc_per_thread(p), ACC_LIMIT[p.acc]
    if p.block_m * p.block_n > area:
        return f"has a {p.block_m}x{p.block_n} tile ({p.block_m * p.block_n} elements) over the larger parent's {area}"
    if per > limit:
        return f"puts {per} {p.acc.upper()} accumulator elements on each of {p.warps * 32} threads, over {limit}"
    return None


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
        self.hints: dict[str, tuple[str, str, str]] = {}  # branch -> (suggestion, note for the hypothesis, help or directive id)
        self.extra: dict[str, int] = {}  # branch -> iterations added by a help answer
        self.summaries: dict[str, dict] = {}
        self.summarized: set[str] = set()  # hypotheses whose latest verdict has a summary
        self.pms: dict[str, dict] = {}
        self.promises: list[dict] = []  # extra iterations a plan_change promised: {ref, branch, n, start}
        self.conflicts = 0
        self.hybrid: dict | None = None  # {branches, params, base, text, t, state: proposed, approved, expired, done}
        self.skipped = ""  # the last logged reason no hybrid could be proposed
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

    def base_capped(self):
        return self.max_hyps is not None and sum(not h.bonus for h in self.hyps) >= self.max_hyps

    def owed(self):
        """Branches holding a reserved slot: a presenter hunch still to test, or the approved hybrid."""
        out = {bid for bid, hint in self.hints.items() if hint[2] in self.reserved}
        return out | ({"merge"} if "hybrid" in self.reserved else set())

    def capped(self):
        return self.base_capped() and not any(self.branches[b].status in ("running", "waiting_on_human") for b in self.owed() if b in self.branches)

    def spend(self, h, hint):
        """Marks h as the hypothesis that used its hint's reserved slot, if the hint had one."""
        if h and hint and hint[2] in self.reserved:
            h.bonus = True
            self.reserved.discard(hint[2])

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
                if self.capped() and not self.pending():
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
        best = self.best_run()
        out = {"run": best.run, "vs_naive": round(best.metrics.speedups["naive"], 2), "vs_eager": round(best.metrics.speedups["eager"], 2)} if best else {"run": "naive", "vs_naive": 1.0, "vs_eager": round(self.eager.latency_ms / self.naive.latency_ms, 2)}
        if ledger:
            self.export([("ledger", f"Hypothesis ledger {self.hyps[0].id} to {self.hyps[-1].id}", "ledger.md",
                          ex.ledger_md(self.hyps, out if best else None, self.naive.latency_ms, self.eager.latency_ms))]
                        + [("summary", f"Branch summary: {bid}", f"summary-{bid}.md", ex.summary_md(sm)) for bid, sm in self.summaries.items()])
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
        replay = self.log.path.with_name("replay.jsonl")
        self.note(f"replay: ?mode=replay&file=/runs/{self.session}/replay.jsonl (written after session_finished)")
        self.log.emit("session_finished", ledger=ledger, best=out)
        # a replay file only ever holds a complete session
        tmp = replay.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for e in self.log.events))
        tmp.replace(replay)

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
            return
        lens = d["lens"].capitalize()
        self.hints.setdefault(b.id, (d["text"], f"Tested under the {lens} lens bound set by {d['id']}", d["id"], None))
        bound = self.bounds(b.lens)
        self.log.emit("plan_change", help=d["id"], text=f"{d['id']} sets a bound, not a specific change, so the next hypotheses on {b.id} are chosen by the agent "
                                                       f"and judged against max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")

    def tested(self, delta):
        """The latest measured hypothesis whose own change was this hunch's, or None."""
        hits = [x for x in self.hyps if x.judgment and x.metrics and x.metrics.latency_ms and all(getattr(x.params, k) == v for k, v in delta.items())
                and any(getattr(self.prior(x)[0], k) != v for k, v in delta.items())]
        return hits[-1] if hits else None

    def earlier_text(self, delta):
        e = delta and self.tested(delta)
        if not e:
            return ""
        sp = f", {e.speedup:.2f}x vs {'naive' if e.ref == 'naive' else 'its parent'}" if e.speedup else ""
        return (f" It was already tested in {e.id} ({e.judgment.verdict}{sp}) on {e.params.block_m}x{e.params.block_n}x{e.params.block_k} tiles, "
                "so this is a retest on the current best tiles.")

    def hunch(self, b, text, cite, ref, lens, delta):
        """A typed hunch is the next hypothesis on its branch, in a reserved slot that the cap does not count, or gets a plan_change
        saying why not. Never dropped silently."""
        if self.stopped:
            self.log.emit("plan_change", help=ref, text=f"Not testing \"{text[:120]}\" because the session was stopped.")
            return
        self.hints[b.id] = (text, cite, ref, delta)
        self.reserved.add(ref)
        when = "next" if b.status == "running" else f"as soon as {b.id} resumes"
        bound = self.bounds(b.lens)
        self.log.emit("plan_change", help=ref, text=f"Because the {lens.capitalize()} lens sent \"{text[:120]}\", I am testing it {when} on {b.id}, "
                                                    f"judged against max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound."
                                                    + self.earlier_text(delta))

    def rehome_hints(self):
        for bid, hint in list(self.hints.items()):
            text, ref = hint[0], hint[2]
            b = self.branches[bid]
            if b.status in ("running", "waiting_on_human"):
                continue
            del self.hints[bid]
            if hint[3] is None:
                continue  # a bound applies to every branch of its lens anyway, nothing was left untested
            others = [x for x in self.branches.values() if x.lens == b.lens and x.status in ("running", "waiting_on_human")]
            if others:
                self.hints[others[-1].id] = hint
                self.log.emit("plan_change", help=ref, text=f"{bid} closed ({b.status}) before testing \"{text[:120]}\", so I am testing it next on {others[-1].id}.")
            else:
                self.reserved.discard(ref)
                self.log.emit("plan_change", help=ref, text=f"{bid} closed ({b.status}) before testing \"{text[:120]}\" and no other {b.lens} branch is open. "
                                                            "Send it again as a directive to open one.")

    def reevaluate(self, d):
        """Re-judges every measured run under the bound now active on its branch, so each verdict shows the bound in force. A run that
        flips from confirmed to rejected rejects its branch, with a post-mortem and a pivot fork."""
        speed = [x["id"] for x in self.directives if x["lens"] == "speed"]
        for h in self.hyps:
            j = h.judgment
            bound = self.bounds(self.branches[h.branch].lens)
            if not (j and h.metrics.latency_ms) or h.bounds[-1] == bound:
                continue
            new = judge(h.prediction, h.metrics, h.speedup, h.noise, bound["standard"], bound["adversarial"])
            self.set_verdict(h, new, bound)
            if j.verdict == "confirmed" and new.verdict == "rejected":
                self.branches[h.branch].status = "rejected"
                self.log.emit("branch_status", branch=h.branch, status="rejected")
                if speed:
                    self.conflicts += 1
                    batches = [c.name.split()[0] for c in new.checks if c.passed is False and "error" in c.name]
                    self.log.emit("conflict_detected", directives=speed + [d["id"]],
                                  reason=f"The best candidate for the Speed directive ({h.id}) violates the Precision bound on the {' and '.join(batches)} "
                                         f"batch{'es' if len(batches) > 1 else ''}. Forking instead of overwriting either directive.")
                self.post_mortem(h)
                self.summarize(h, self.branches[h.branch], f"Fork: keep what worked in {h.id} and drop {ruled_out(h.params, self.start_point(h.branch)[0])}.")
                self.export_failure(h, self.branches[h.branch])
                self.pivot(h)
                self.refresh_helps(h)
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
        self.fork(b.id, b.lens, f"{'Speed' if b.lens == 'speed' else 'Precision'} lens, pivot from {h.id}",
                  f"Pivot from the {h.id} post-mortem: drop {ruled_out(h.params, self.start_point(b.id)[0])} and keep the rest.")

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
        ok = [h for h in self.hyps if h.judgment and h.judgment.verdict == "confirmed" and not h.judgment.provisional
              and (branch is None or h.branch in self.lineage(branch))]
        return min(ok, key=lambda h: h.metrics.latency_ms) if ok else None

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
        if b.fails >= 2 and not any(x["branch"] == b.id and open_help(x) for x in self.helps.values()):
            target = b.lens if b.lens != "both" else "speed"
            self.summarize(h, b, f"Ask the {target.capitalize()} lens how to continue after two failed runs in a row.")
            self.request_help(h, b, target, failed=True)
            return

        # Only correctness failures end a branch with a post-mortem and a pivot. A missed speedup is a normal miss.
        unsafe = v == "rejected" and any(c.passed is False and "error" in c.name for c in h.judgment.checks)
        if unsafe and b.lens != "both":
            b.status = "rejected"
            self.log.emit("branch_status", branch=b.id, status="rejected")
            self.post_mortem(h)
            self.summarize(h, b, f"Fork: drop {ruled_out(h.params, self.start_point(b.id)[0])} and keep the rest.")
            self.export_failure(h, b)
            self.pivot(h)
            return
        slow = h.ref == "parent" and h.speedup is not None and h.speedup < SLOW
        if unsafe or slow:
            self.post_mortem(h)

        b.misses = 0 if v == "confirmed" else b.misses + 1
        if v == "inconclusive" and not any(x["branch"] == b.id for x in self.helps.values()):
            target = b.lens if b.lens != "both" else "speed"
            self.summarize(h, b, f"Ask the {target.capitalize()} lens whether to keep tuning or change approach.")
            self.request_help(h, b, target)
            return
        budget = (b.budget or (self.main_iters if b.id == "main" else self.branch_iters)) + self.extra.get(b.id, 0)
        # two misses in a row end a branch, unless a lens owner was promised more iterations on it
        done = b.iters >= budget or (b.misses >= 2 and not self.promised(b.id))
        if v == "rejected":
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
        if unsafe or slow:
            self.export_failure(h, b)
        if b.id == "merge" and v == "confirmed":
            for pid in b.parents:
                self.close(self.branches[pid], "merged")
        if done:
            self.close(b, "done")
        self.maybe_hybrid()

    # ---- help requests ----

    def pending(self):
        return any(open_help(x) for x in self.helps.values()) or bool(self.hybrid and self.hybrid["state"] == "proposed")

    def successor(self, b):
        """The open branch that continues a rejected one: its pivot, the newest open child on the same lens."""
        kids = [x for x in self.branches.values() if b.id in x.parents and x.lens == b.lens and x.status in ("running", "waiting_on_human")]
        return kids[-1] if kids else None

    def live(self, bid):
        """The branch a help answer applies to: the branch itself, or its successor once it was rejected. None when nothing continues it."""
        b = self.branches[bid]
        return (self.successor(b) if b.status == "rejected" else b)

    def evidence(self, lineage):
        return [f"{x.run}: {x.metrics.latency_ms:.3f} ms, " + ("provisionally confirmed" if x.judgment.provisional else x.judgment.verdict)
                + (f", {c.observed}" if (c := next((c for c in x.judgment.checks if c.name.startswith("speedup")), None)) else "")
                for x in reversed(self.hyps) if x.branch in lineage and x.metrics and x.metrics.latency_ms and x.judgment][:3]

    def refresh_helps(self, h):
        """After h's verdict flipped, a pending request on a rejected branch moves to its successor (or is withdrawn when none is open),
        and any pending request whose evidence lists h's run is reissued with the current verdicts. Same id, so an answer still lands."""
        for hid, x in self.helps.items():
            if not open_help(x):
                continue
            b = self.branches[x["branch"]]
            moved = b.status == "rejected"
            if not moved and not any(e.startswith(f"{h.run}:") for e in x["help"]["evidence"]):
                continue
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
            x["help"], x["t"] = req, self.log.clock()
            self.log.emit("help_requested", help=req)

    def request_help(self, h: Hyp, b: Branch, target, failed=False):
        lineage = self.lineage(b.id)
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
        req = {
            "id": hid, "target": target,
            "question": question,
            "evidence": evidence,
            "options": options,
            "default": "tile_search", "blocks": b.id, "deadline_s": int(self.help_deadline_s),
            "what_changes": "Keep tuning adds two more iterations on this branch, accept closes it, and a suggestion becomes the next hypothesis to test.",
        }
        self.helps[hid] = {"help": req, "branch": b.id, "t": self.log.clock(), "answered": False, "expired": False, "cancelled": False,
                           "lineage": lineage, "failed": failed}
        self.log.emit("help_requested", help=req)
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
            if not open_help(x) or self.log.clock() - x["t"] < x["help"]["deadline_s"]:
                continue
            x["expired"] = True
            b = self.live(x["branch"])
            if b is None:
                continue
            self.promise(hid, b, 2)
            self.log.emit("plan_change", help=hid, text=f"No answer to {hid} by the deadline, so I am proceeding on the default assumption: keep tuning tiles and warps on {b.id}. "
                                                        "Results on this branch are flagged assumption pending until an answer arrives.")
            self.reopen(b)

    def promised(self, bid):
        """Iterations a plan_change promised on this branch that have not run yet."""
        return sum(max(0, pr["n"] - (sum(h.branch == bid for h in self.hyps) - pr["start"])) for pr in self.promises if pr["branch"] == bid)

    def promise(self, ref, b, n):
        self.extra[b.id] = self.extra.get(b.id, 0) + n
        self.promises.append({"ref": ref, "branch": b.id, "n": n, "start": sum(h.branch == b.id for h in self.hyps)})

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
            if cmd["option"] == "other" and note:
                delta = hyp.hunch_change(note) or {}
                self.hints[b.id] = (note, f"Suggested by the {lens} lens in answer to {cmd['help']}", cmd["help"], delta)
                text = (f"{late}Because the {lens} lens suggests \"{note}\", I am testing that next on {b.id} instead of more tile search, still under the active error bound."
                        + self.earlier_text(delta))
            else:
                text = f"{late}Because the {lens} lens wants more tile search, I am giving {b.id} two more iterations."
            if not x["expired"]:
                self.promise(cmd["help"], b, 2)
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
        if b.status != "running":
            b.status = "running"
            self.log.emit("branch_status", branch=b.id, status="running")

    def close(self, b, status):
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
        if b.id == "merge" and self.hybrid and self.hybrid["state"] == "approved":
            return self.hybrid_hyp(b)
        # the seeded FP16 variant is the first Speed lens hypothesis, ahead of a pending hunch
        seed = None if self.base_capped() else self.seed(b, tried_all)
        if seed:
            if b.id in self.hints and self.hints[b.id][3] is not None:
                self.log.emit("plan_change", help=self.hints[b.id][2], text=f"The scheduled FP16 variant runs first on {b.id}, "
                                                                            f"then \"{self.hints[b.id][0][:120]}\" is the next hypothesis there.")
            return seed
        hint = self.hints.pop(b.id, None)
        if hint and hint[3] and self.tested(hint[3]):
            retest = self.retest(b, start, hint, tried_all)
            if retest:
                self.spend(retest, hint)
                return retest
            self.reserved.discard(hint[2])
            hint = None

        tried = [{"params": h.params.dict(), "verdict": h.judgment.verdict, "metrics": h.metrics.contract()}
                 for h in self.hyps if h.branch == b.id and h.judgment and h.metrics.latency_ms]
        directives = [d for d in self.directives if b.lens in ("both", d["lens"])]
        planned = facet(start, replace(start, **{k: v for k, v in hint[3].items() if k in start.dict()})) if hint and hint[3] else None
        notes = self.papiers(planned or QUERIES[b.lens])
        messages = hyp.prompt(b.dict(), directives, {"params": start.dict(), "metrics": metrics.contract()}, tried, self.avoid,
                              ([f"Suggestion from a human lens owner, test this next: {hint[0]}"] if hint and hint[3] is not None else [])
                              + [f"Papiers {n['id']} ({n['title']}): {n['excerpt']}" for n in notes], self.bounds(b.lens)["standard"], self.headroom)
        why, first = "", ""
        for attempt in range(3):
            text = self.ask_or_wait(messages, f"{b.id}-{b.iters}-{attempt}", b.id)
            if text is None:
                if hint:
                    self.hints[b.id] = hint
                return None
            h, why = hyp.parse(text)
            if h and json.dumps(h["params"].dict(), sort_keys=True) in tried_all:
                h, why = None, "that configuration was already tried"
            if h:
                cite = hint[1] if hint else None
                claim = h["claim"]
                if h["adjusted"]:
                    claim = claim_text(start, h["params"], h["prediction"], self.bounds(b.lens))
                    cite = f"{cite}. {h['adjusted']}" if cite else h["adjusted"]
                new = self.new_hyp(b, claim, h["prediction"], h["params"], cite)
                self.spend(new, hint)
                if hint and hint[3] is not None and first:
                    self.log.emit("plan_change", help=hint[2], text=f"My first reading of \"{hint[0][:120]}\" was not testable ({first}), so {new.id} tests "
                                                                    f"the closest valid version: {change(start, new.params)}.")
                return new
            first = first or why
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Invalid: {why}. Reply again with only the JSON object."}]
        if hint:
            self.reserved.discard(hint[2])
        if hint and hint[3] is not None:
            self.log.emit("plan_change", help=hint[2], text=f"I could not turn \"{hint[0][:120]}\" into a valid configuration on {b.id} ({why}), so it is not tested.")
        self.log.emit("agent_log", text=f"No valid hypothesis for branch {b.id} after 3 attempts ({why}). Closing the branch.")
        return None

    def retest(self, b, start, hint, tried_all):
        """A hunch tested earlier is rerun as that change on this branch's current best params, built in code like the seed."""
        e = self.tested(hint[3])
        try:
            p = fit_stages(replace(start, **hint[3]))
        except ValueError as err:
            self.log.emit("plan_change", help=hint[2], text=f"\"{hint[0][:120]}\" names an invalid value ({err}), so it is not retested.")
            return None
        if acc_per_thread(p) > ACC_LIMIT[p.acc]:
            self.log.emit("plan_change", help=hint[2], text=f"\"{hint[0][:120]}\" on {b.id}'s current best gives {params_text(p)}, which puts {acc_per_thread(p)} "
                                                           f"accumulator elements on each thread, over the {ACC_LIMIT[p.acc]} register limit, so it is not retested.")
            return None
        if p == start or json.dumps(p.dict(), sort_keys=True) in tried_all:
            ran = next((x for x in self.hyps if x.params == p), None)
            why = (f"that exact config, {params_text(p)}, already ran as {ran.id} on {ran.branch} ({ran.judgment.verdict})" if ran
                   else f"{b.id}'s current best already has {ruled_out(p, self.prior(e)[0])}")
            self.log.emit("plan_change", help=hint[2], text=f"\"{hint[0][:120]}\" is not retested because {why}."
                                                           + (f" The hunch was last tested in {e.id} ({e.judgment.verdict})." if e is not ran else ""))
            return None
        bound = self.bounds(b.lens)
        claim = (f"Retest of {e.id}: {change(start, p)} on the current best {tiles_text(start)} is at least 1.02x faster than the same config without it, "
                 f"with standard max error under {sci(bound['standard'])}, the {bound_owner(bound)} bound.")
        h = self.new_hyp(b, claim, {"min_speedup": 1.02, "vs": "parent_best", "max_error": bound["standard"]}, p,
                         f"{hint[1]}, a retest of {e.id} on the current best tiles")
        h.base = start
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

    def maybe_hybrid(self):
        """After a lens conflict, proposes merging the fastest correct run of each lens: the parameters each lens changed from the
        common start, with the Precision lens winning where both changed the same one. The union of two tile changes can overflow
        the registers, so a candidate must keep the tile area within the larger parent's and the accumulator per thread within
        ACC_LIMIT. When the full union does not, the hybrid keeps one parent's tile shape and takes the merged dtype and GELU.
        Approval is the presenter's."""
        if self.hybrid or not self.conflicts or not any(d["lens"] == "precision" for d in self.directives):
            return
        lens_best = {}
        for lens in ("precision", "speed"):
            runs = self.correct_runs([b.id for b in self.branches.values() if b.lens == lens])
            if not runs:
                return
            lens_best[lens] = runs[0]
        pb, sb = lens_best["precision"], lens_best["speed"]
        base = self.start_point("main")[0]
        union = Params(**{k: getattr(pb.params, k) if getattr(pb.params, k) != getattr(base, k) else getattr(sb.params, k) for k in base.dict()})
        faster, slower = sorted((pb, sb), key=lambda h: h.metrics.latency_ms)
        area = max(x.params.block_m * x.params.block_n for x in (pb, sb))
        rejects = []
        for src in (None, faster, slower):
            p = fit_stages(union if src is None else replace(union, **{k: getattr(src.params, k) for k in TILE}))
            same = next((x for x in (pb, sb) if x.params == p), None)
            ran = next((x for x in self.hyps if x.params == p), None)
            why = hybrid_guard(p, area) or (f"is {same.id}'s own configuration" if same else f"was already run as {ran.id}" if ran else None)
            if not why:
                break
            rejects.append(f"The full union, {params_text(p)}, {why}" if src is None else f"The tile shape of {src.branch} ({src.id}) gives {params_text(p)}, which {why}")
        else:
            # logged once per distinct reason, since this runs after every iteration
            text = f"No hybrid of {pb.branch} ({pb.id}) and {sb.branch} ({sb.id}) to propose: {'. '.join(rejects)}."
            if text != self.skipped:
                self.skipped = text
                self.log.emit("agent_log", text=text)
            return
        keep = f" Keeping the tile shape of {src.branch} ({src.id}) instead." if src else ""
        text = (f"Combine {change(base, sb.params)} from {sb.branch} ({sb.id}) with {change(base, pb.params)} from {pb.branch} ({pb.id}). "
                + (f"{'. '.join(rejects)}.{keep} " if rejects else "")
                + f"Candidate: {params_text(p)}, {acc_per_thread(p)} accumulator elements per thread. "
                "The expected gain is small, so the result shows whether the two lenses' changes add up.")
        self.hybrid = {"branches": [pb.branch, sb.branch], "params": p, "base": faster.params, "faster": faster, "text": text,
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

    def cite(self, h: Hyp, start: Params):
        """Before the run: reads Papiers for the change this hypothesis makes and cites the top item in its note. Nothing is cited when
        the change query finds nothing, even if the prompt had other context, so a citation is always about the change under test."""
        found = self.papiers(facet(h.base or start, h.params))
        if found:
            h.note = f"{h.note.rstrip('.')}. {cite_text(found[0])}" if h.note else cite_text(found[0])

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
        elif h.ref == "parent" and h.speedup < SLOW and m.max_error < h.prediction["max_error"]:
            pp = h.base or self.best_run(h.branch).params
            parent = next((x.metrics for x in self.hyps if x.params == pp and x.metrics and x.metrics.latency_ms), None)
            found = self.papiers("register pressure spills")
            pm = {"hypothesis": h.id, "expected": expected, "sweep": [],
                  "observed": f"{m.latency_ms:.3f} ms against {m.ref_ms['parent']:.3f} ms for the parent in the same window, {h.speedup:.2f}x (+/- {h.speedup * h.noise:.2f}x)."} \
                | register_cause(h.params, m, h.speedup, parent)
        else:
            observed = f"Speedup of {h.speedup:.2f}x (+/- {h.speedup * h.noise:.2f}x), standard error {sci(m.max_error)}, adversarial error {sci(m.adv_error)}."
            s = Metrics() if self.stopped else self.gpu_run({"target": "sweep", "params": h.params.dict(), "ks": SWEEP_KS}, f"{self.session}/{h.id}-sweep")
            sweep = [{"k": k, "adv_error": s.errors[f"k{k}"]} for k in SWEEP_KS if f"k{k}" in s.errors]
            trend = (f"Adversarial error goes from {sci(sweep[0]['adv_error'])} at K={sweep[0]['k']} to {sci(sweep[-1]['adv_error'])} at K={sweep[-1]['k']}"
                     f"{f', above the strict {sci(BOUND)} bound at every K' if all(x['adv_error'] >= BOUND for x in sweep) else ''}." if sweep else "")
            # the query names the measured cause: an error that grows with K is an accumulation length effect
            rising = len(sweep) > 1 and sweep[-1]["adv_error"] > sweep[0]["adv_error"]
            found = self.papiers("accumulation error grows with length" if rising else facet(base, h.params))
            context = "".join(f"\nThe user's Papiers {n.get('kind') or 'note'} {n['id']} from \"{n.get('document_title') or n['title']}\": {n['excerpt'][:300]}"
                              + (f" (their note: {n['annotation']})" if n.get("annotation") else "") for n in found)
            fallback = {"cause": f"{h.judgment.reason} {trend}".strip(),
                        "pivot": f"Drop {ruled_out(h.params, base)}, keep the other parameters, and test whether the gain survives."}
            prose = self.ask_json([{"role": "system", "content": "You write terse, factual post-mortems for GPU kernel experiments. Use only the numbers given. Reply with only JSON: {\"cause\": \"<two sentences on the mechanism>\", \"pivot\": \"<one sentence on what to try next>\"}"},
                                   {"role": "user", "content": f"Error metric: max |y - ref| / max(|ref|, 1) against an FP32 reference. The strict bound is {sci(BOUND)}, the Speed lens tolerates {sci(hyp.SPEED_TOLERANCE)} on the standard batch only.\nHypothesis {h.id}: {h.claim}\nChange: {change(base, h.params)}\nExpected: {expected}\nObserved: {observed}\n"
                                                               f"Verdict reason: {h.judgment.reason}\nError vs K sweep on adversarial magnitudes: {json.dumps(sweep)}{context}"}],
                                  self.strong, f"pm-{h.id}", ("cause", "pivot")) or fallback
            pm = {"hypothesis": h.id, "expected": expected, "observed": observed, "cause": prose["cause"], "pivot": prose["pivot"], "sweep": sweep}
        if found:
            pm["note"] = found[0]["id"]
        self.avoid.append(f"{ruled_out(h.params, base)} ({h.id} rejected: {h.judgment.reason})")
        self.pms[h.id] = pm
        self.log.emit("post_mortem", post_mortem=pm)

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
