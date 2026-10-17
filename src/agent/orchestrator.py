"""The research loop: hypotheses with stated predictions, constrained runs, code computed verdicts, branches per lens, post-mortems.

Branches run one at a time on the single GPU. Commands from the UI (directives) are queued and applied between iterations.
"""

import json
import queue
import re
import threading
import time
from dataclasses import dataclass, field, replace

from agent import export as ex
from agent import hypothesis as hyp
from agent.events import Log
from agent.verdict import Judgment, judge, sci
from harness.data import BOUND
from harness.kernel import NAIVE, Params
from harness.metrics import Metrics
from harness.runner import TIMEOUT_S

SWEEP_KS = [256, 512, 1024, 2048, 4096]
LENS_IDS = {"speed": "speed", "precision": "prec"}
QUERIES = {"both": "Triton matrix multiplication tiling warps pipeline stages", "speed": "FP16 accumulation throughput tensor cores",
           "precision": "accumulation numerical error precision bound"}
MAX_PER_LENS = 3
DIRECTIVE_GAP_S = 10.0
MAX_DIRECTIVES = 6


@dataclass
class Branch:
    id: str
    parents: list[str]
    label: str
    lens: str
    status: str = "running"
    iters: int = 0
    misses: int = 0

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

    def dict(self):
        d = {"id": self.id, "branch": self.branch, "claim": self.claim, "prediction": self.prediction, "params": self.params.dict()}
        return d | ({"note": self.note} if self.note else {})


def change(a: Params, b: Params):
    diff = [f"{k} {getattr(a, k)} to {getattr(b, k)}" for k in a.dict() if getattr(a, k) != getattr(b, k)]
    return ", ".join(diff) or "no parameter change"


def ruled_out(p: Params, base: Params):
    keys = [k for k in p.dict() if getattr(p, k) != getattr(base, k) and k in ("acc", "gelu")] or [k for k in p.dict() if getattr(p, k) != getattr(base, k)]
    return ", ".join(f"{k}={getattr(p, k)}" for k in keys) or f"the configuration {p.dict()}"


class Stopped(Exception):
    pass


class Orchestrator:
    """engine(spec, run_id) -> Metrics runs one experiment (harness.runner.run in production).
    ask(messages, model, tag) -> str is the LLM. fast and strong are model ids.
    context(query) -> [{id, title, excerpt}] reads Papiers, optional. headroom is harness.noise.headroom(), quoted in the prompt.
    record marks a recording session: uncapped by the live demo cap, it saves its log as a replay."""

    def __init__(self, engine, ask, log: Log, session: str, fast: str, strong: str, band: float,
                 main_iters=2, branch_iters=3, idle_s=120.0, context=None, help_deadline_s=45.0, max_hyps=None, retry_s=30.0, headroom=None):
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
        self.retry = threading.Event()
        self.helps: dict[str, dict] = {}  # id -> {help, branch, t, answered, expired}
        self.hints: dict[str, tuple[str, str]] = {}  # branch -> (suggestion, note for the hypothesis)
        self.extra: dict[str, int] = {}  # branch -> iterations added by a help answer
        self.summaries: dict[str, dict] = {}
        self.pms: dict[str, dict] = {}

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

    def stop(self):
        self.stopped = True
        self.retry.set()

    def capped(self):
        return self.max_hyps is not None and len(self.hyps) >= self.max_hyps

    def run(self):
        if not self.start():
            return
        idle = time.monotonic()
        while not self.stopped:
            if self.drain():
                idle = time.monotonic()
            self.check_deadlines()
            b = self.next_branch()
            if b:
                self.iterate(b)
                idle = time.monotonic()
            elif (self.capped() or time.monotonic() - idle > self.idle_s) and not self.pending():
                break
            else:
                time.sleep(0.1)
        self.finish()

    # ---- session ----

    def start(self):
        """Measures the baselines. Returns False, with the reason in the agent log, when the session cannot start."""
        self.log.t0 = self.log.clock()  # session time starts at Start, not when the server built the session
        self.log.emit("agent_log", text="Measuring baselines: PyTorch eager and the naive Triton kernel.")
        self.eager = self.engine({"target": "eager"}, f"{self.session}/eager")
        self.naive = self.engine({"target": "triton", "params": NAIVE.dict()}, f"{self.session}/naive")
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
        ledger = [{"hypothesis": h.id, "verdict": h.judgment.verdict, "history": h.history} for h in self.hyps if h.judgment]
        best = self.best_run()
        out = {"run": best.run, "vs_naive": round(best.metrics.speedups["naive"], 2), "vs_eager": round(best.metrics.speedups["eager"], 2)} if best else {"run": "naive", "vs_naive": 1.0, "vs_eager": round(self.eager.latency_ms / self.naive.latency_ms, 2)}
        if ledger:
            self.export([("ledger", f"Hypothesis ledger {self.hyps[0].id} to {self.hyps[-1].id}", "ledger.md",
                          ex.ledger_md(self.hyps, out if best else None, self.naive.latency_ms, self.eager.latency_ms))]
                        + [("summary", f"Branch summary: {bid}", f"summary-{bid}.md", ex.summary_md(sm)) for bid, sm in self.summaries.items()])
        if self.record:
            self.log.emit("agent_log", text=f"Recording saved as a replay: open ?mode=replay&file=/runs/{self.session}/events.jsonl")
        self.log.emit("session_finished", ledger=ledger, best=out)

    # ---- commands ----

    def drain(self):
        handled = False
        while not self.commands.empty():
            cmd = self.commands.get()
            handled = True
            if cmd["type"] == "add_directive":
                self.add_directive(cmd["lens"], cmd["text"])
            elif cmd["type"] == "answer_help":
                self.answer(cmd)
            elif cmd["type"] in ("retry", "start"):
                self.retry.set()
            else:
                self.log.emit("agent_log", text=f"Received {cmd['type']}, which this build does not act on yet.")
        return handled

    def add_directive(self, lens, text):
        now = self.log.clock() - self.log.t0
        recent = [x for x in self.directives if x["lens"] == lens and now - x["at"] < DIRECTIVE_GAP_S]
        if recent or len(self.directives) >= MAX_DIRECTIVES:
            why = f"another {lens} directive arrived less than {DIRECTIVE_GAP_S:.0f} s ago" if recent else f"the session already has {MAX_DIRECTIVES} directives"
            self.log.emit("agent_log", text=f"Ignored a {lens} directive ({why}): {text[:80]}")
            return
        d = {"id": f"d{len(self.directives) + 1}", "lens": lens, "text": text, **hyp.directive(lens, text)}
        self.directives.append(d | {"at": now})
        self.log.emit("directive_added", directive=d)
        if lens == "speed":
            self.log.emit("agent_log", text=f"Speed lens bound assigned: max error under {sci(d['threshold'])} on the standard batch for Speed branches "
                                            f"(default {sci(hyp.SPEED_TOLERANCE)}, ceiling {sci(hyp.SPEED_CEILING)}).")
        if lens == "precision":
            self.log.emit("agent_log", text=f"Precision bound assigned: {sci(d['threshold'])} on the standard and adversarial batches for every branch. "
                                            "Re-evaluating logged runs, no new runs needed.")
            self.reevaluate(d)
        if not any(b.lens == lens and b.status in ("running", "waiting_on_human") for b in self.branches.values()):
            parent = self.best_run()
            pid = parent.branch if parent else "main"
            reason = f"{'Speed' if lens == 'speed' else 'Precision'} lens directive {d['id']}: {text}"
            self.fork(pid, lens, f"{'Speed' if lens == 'speed' else 'Precision'}: {text[:32]}", reason)

    def reevaluate(self, d):
        speed = [x["id"] for x in self.directives if x["lens"] == "speed"]
        for h in self.hyps:
            j = h.judgment
            if not (j and j.verdict == "confirmed" and j.provisional):
                continue
            bound = {"standard": d["threshold"], "adversarial": d["threshold"], "set_by": "precision"}
            new = judge(h.prediction, h.metrics, h.speedup, h.noise, d["threshold"], d["threshold"])
            if new.verdict == j.verdict and new.provisional == j.provisional:
                continue
            self.set_verdict(h, new, bound)
            if new.verdict == "rejected":
                self.branches[h.branch].status = "rejected"
                self.log.emit("branch_status", branch=h.branch, status="rejected")
                if speed:
                    self.log.emit("conflict_detected", directives=speed + [d["id"]],
                                  reason=f"The best candidate for the Speed directive ({h.id}) violates the Precision bound on the adversarial batch. Forking instead of overwriting either directive.")
                self.post_mortem(h)
                self.summarize(h, self.branches[h.branch], f"Fork: keep what worked in {h.id} and drop {ruled_out(h.params, self.start_point(h.branch)[0])}.")
                self.export_failure(h, self.branches[h.branch])
                self.pivot(h)

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
        self.fork(b.id, b.lens, f"{'Speed' if b.lens == 'speed' else 'Precision'}: pivot from {h.id}",
                  f"Pivot from the {h.id} post-mortem: drop {ruled_out(h.params, self.start_point(b.id)[0])} and keep the rest.")

    def next_branch(self):
        if self.capped() or self.stopped:
            return None
        live = [b for b in self.branches.values() if b.status == "running"]
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
        self.hyps.append(h)
        self.log.emit("hypothesis_proposed", hypothesis=h.dict())
        best = self.best_run(b.id)
        ref = "parent" if h.prediction["vs"] == "parent_best" and best else "naive"
        self.measure(h, best.params if ref == "parent" else None)
        h.speedup = h.metrics.speedups.get(ref)
        h.noise = max(self.band, h.metrics.spreads.get(ref, 0.0))
        bound = self.bounds(b.lens)
        self.set_verdict(h, judge(h.prediction, h.metrics, h.speedup, h.noise, bound["standard"], bound["adversarial"]), bound)
        v = h.judgment.verdict

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
        if unsafe:
            self.post_mortem(h)

        b.misses = 0 if v == "confirmed" else b.misses + 1
        if v == "inconclusive" and not any(x["branch"] == b.id for x in self.helps.values()):
            target = b.lens if b.lens != "both" else "speed"
            self.summarize(h, b, f"Ask the {target.capitalize()} lens whether to keep tuning or change approach.")
            self.request_help(h, b, target)
            return
        budget = (self.main_iters if b.id == "main" else self.branch_iters) + self.extra.get(b.id, 0)
        done = b.iters >= budget or b.misses >= 2
        if v == "rejected":
            nxt = "Stop this branch and wait for a lens directive." if done else "Try a different change on this branch, avoiding what the post-mortem ruled out."
        elif v == "confirmed" and h.judgment.provisional:
            nxt = "Hold and flag the adversarial error until a precision lens is attached."
        elif done:
            nxt = "Stop this branch and wait for a lens directive." if b.id == "main" else "Stop this branch, the iteration budget is spent."
        else:
            nxt = "Refine from the best configuration on this branch." if v == "confirmed" else "Try a different change, this one did not clearly beat the noise band."
        self.summarize(h, b, nxt)
        if unsafe:
            self.export_failure(h, b)
        if done:
            self.close(b, "done")

    # ---- help requests ----

    def pending(self):
        return any(not x["answered"] and not x["expired"] for x in self.helps.values())

    def request_help(self, h: Hyp, b: Branch, target):
        speed = next(c for c in h.judgment.checks if c.name.startswith("speedup"))
        lineage = self.lineage(b.id)
        evidence = [f"{x.run}: {x.metrics.latency_ms:.3f} ms, {x.judgment.verdict}" + (f", {c.observed}" if (c := next((c for c in x.judgment.checks if c.name.startswith("speedup")), None)) else "")
                    for x in reversed(self.hyps) if x.branch in lineage and x.metrics and x.metrics.latency_ms and x.judgment][:3]
        hid = f"hq{len(self.helps) + 1}"
        req = {
            "id": hid, "target": target,
            "question": f"Branch {b.id} reached {speed.observed.split(' ')[0]} against my {speed.predicted} prediction, inside the noise band. Should I keep tuning or change approach?",
            "evidence": evidence,
            "options": [{"id": "tile_search", "label": "Keep tuning tiles and warps"},
                        {"id": "accept", "label": f"Accept {speed.observed.split(' ')[0]} and stop this branch"},
                        {"id": "other", "label": "Suggest something else"}],
            "default": "tile_search", "blocks": b.id, "deadline_s": int(self.help_deadline_s),
            "what_changes": "Keep tuning adds two more iterations on this branch, accept closes it, and a suggestion becomes the next hypothesis to test.",
        }
        self.helps[hid] = {"help": req, "branch": b.id, "t": self.log.clock(), "answered": False, "expired": False}
        self.log.emit("help_requested", help=req)
        b.status = "waiting_on_human"
        self.log.emit("branch_status", branch=b.id, status="waiting_on_human")
        others = [x.id for x in self.branches.values() if x.status == "running"]
        self.log.emit("agent_log", text=f"Branch {b.id} is blocked on the {target.capitalize()} lens ({hid}). "
                                        + (f"Continuing on {', '.join(others)}." if others else "Nothing else is runnable, waiting for the answer or the deadline."))

    def check_deadlines(self):
        for hid, x in self.helps.items():
            if x["answered"] or x["expired"] or self.log.clock() - x["t"] < x["help"]["deadline_s"]:
                continue
            x["expired"] = True
            b = self.branches[x["branch"]]
            self.extra[b.id] = self.extra.get(b.id, 0) + 2
            self.log.emit("plan_change", help=hid, text=f"No answer to {hid} by the deadline, so I am proceeding on the default assumption: keep tuning tiles and warps on {b.id}. "
                                                        "Results on this branch are flagged assumption pending until an answer arrives.")
            self.reopen(b)

    def answer(self, cmd):
        x = self.helps.get(cmd["help"])
        if not x or x["answered"] or cmd["option"] not in {o["id"] for o in x["help"]["options"]}:
            self.log.emit("agent_log", text=f"Ignored an answer to {cmd['help']}: unknown request, unknown option, or already answered.")
            return
        x["answered"] = True
        note = (cmd.get("note") or "").strip()
        self.log.emit("help_answered", help=cmd["help"], option=cmd["option"], by=cmd["by"], **({"note": note} if note else {}))
        b = self.branches[x["branch"]]
        late = "The answer arrived after the deadline, so I am revising the default. " if x["expired"] else ""
        lens = cmd["by"].capitalize()
        b.misses = 0
        if cmd["option"] == "accept":
            text = f"{late}Because the {lens} lens accepts the current result, I am closing {b.id} and keeping its best configuration."
            self.log.emit("plan_change", help=cmd["help"], text=text)
            if b.status in ("running", "waiting_on_human"):
                self.close(b, "done")
        else:
            if cmd["option"] == "other" and note:
                self.hints[b.id] = (note, f"Suggested by the {lens} lens in answer to {cmd['help']}")
                text = f"{late}Because the {lens} lens suggests \"{note}\", I am testing that next on {b.id} instead of more tile search, still under the active error bound."
            else:
                text = f"{late}Because the {lens} lens wants more tile search, I am giving {b.id} two more iterations."
            if not x["expired"]:
                self.extra[b.id] = self.extra.get(b.id, 0) + 2
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
        hint = self.hints.pop(b.id, None)
        # a human suggestion goes first, the seeded FP16 variant waits for a later iteration and for a confirmed configuration,
        # since FP16 on the naive tiles measures nothing about accumulation
        tested = any(h.params.acc == "fp16" and self.branches[h.branch].lens == "speed" for h in self.hyps)
        if not hint and b.lens == "speed" and b.iters >= 2 and not tested and start.acc == "fp32" and self.best_run(b.id):
            p = replace(start, acc="fp16")
            if json.dumps(p.dict(), sort_keys=True) not in tried_all:
                return self.new_hyp(b, "FP16 accumulation on the best configuration so far will cut latency by at least 1.2x and stay under the Speed tolerance.",
                                    {"min_speedup": 1.2, "vs": "parent_best", "max_error": BOUND}, p,
                                    "Harness variant from the precision sweep (plan Section 9): low-precision accumulation had not been tested under the Speed tolerance.")

        tried = [{"params": h.params.dict(), "verdict": h.judgment.verdict, "metrics": h.metrics.contract()}
                 for h in self.hyps if h.branch == b.id and h.judgment and h.metrics.latency_ms]
        directives = [d for d in self.directives if b.lens in ("both", d["lens"])]
        notes = self.papiers(b, directives)
        messages = hyp.prompt(b.dict(), directives, {"params": start.dict(), "metrics": metrics.contract()}, tried, self.avoid,
                              ([f"Suggestion from a human lens owner, test this next: {hint[0]}"] if hint else [])
                              + [f"Papiers {n['id']} ({n['title']}): {n['excerpt']}" for n in notes], self.bounds(b.lens)["standard"], self.headroom)
        why = ""
        for attempt in range(3):
            text = self.ask_or_wait(messages, f"{b.id}-{b.iters}-{attempt}", b.id)
            if text is None:
                return None
            h, why = hyp.parse(text)
            if h and json.dumps(h["params"].dict(), sort_keys=True) in tried_all:
                h, why = None, "that configuration was already tried"
            if h:
                cite = hint[1] if hint else f"Context from Papiers note {notes[0]['id']}" if notes else None
                return self.new_hyp(b, h["claim"], h["prediction"], h["params"], cite)
            messages = messages + [{"role": "assistant", "content": text}, {"role": "user", "content": f"Invalid: {why}. Reply again with only the JSON object."}]
        self.log.emit("agent_log", text=f"No valid hypothesis for branch {b.id} after 3 attempts ({why}). Closing the branch.")
        return None

    def papiers(self, b, directives):
        """Reads Papiers for this branch's topic once per query and logs each read: one event per returned item, or one without a note when the search came back empty."""
        if not self.context:
            return []
        query = next((d["text"] for d in reversed(directives) if d["lens"] == b.lens), None) or QUERIES[b.lens]
        if query not in self.read:
            try:
                self.read[query] = [n | {"excerpt": re.sub(r"\s*\[\[src:[^\]]*\]\]", "", n["excerpt"])[:400]} for n in self.context(query)[:2]]
            except Exception as e:  # noqa: BLE001, Papiers is optional context, the loop runs without it
                self.read[query] = []
                self.log.emit("agent_log", text=f"Papiers search unavailable ({type(e).__name__}), continuing without reading context.")
                return []
            if not self.read[query]:
                self.log.emit("papiers_context", query=query)
            for n in self.read[query]:
                self.log.emit("papiers_context", query=query, note={"id": n["id"], "title": n["title"], "excerpt": n["excerpt"]})
        return self.read[query]

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
                self.retry.wait(self.retry_s)
                self.drain()
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
        h.metrics = self.engine({"target": "triton", "params": h.params.dict(), "refs": refs}, f"{self.session}/{h.id}-{h.run}")
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
            self.log.emit("agent_log", text=f"Explanation model unavailable ({type(e).__name__}), using the factual summary.")
            return None
        return d if all(isinstance(d.get(k), (str, list)) for k in keys) else None

    def post_mortem(self, h: Hyp):
        m = h.metrics
        p = h.prediction
        expected = f"Speedup of at least {p['min_speedup']:.2f}x vs {'naive' if p['vs'] == 'naive' else 'the best so far'} with max error under {sci(p['max_error'])}."
        if m.latency_ms is None:
            observed = f"The run failed: {(m.error or '')[:200]}"
            sweep = []
        else:
            observed = f"Speedup of {h.speedup:.2f}x (+/- {h.speedup * h.noise:.2f}x), standard error {sci(m.max_error)}, adversarial error {sci(m.adv_error)}."
            s = Metrics() if self.stopped else self.engine({"target": "sweep", "params": h.params.dict(), "ks": SWEEP_KS}, f"{self.session}/{h.id}-sweep")
            sweep = [{"k": k, "adv_error": s.errors[f"k{k}"]} for k in SWEEP_KS if f"k{k}" in s.errors]
        trend = (f"Adversarial error goes from {sci(sweep[0]['adv_error'])} at K={sweep[0]['k']} to {sci(sweep[-1]['adv_error'])} at K={sweep[-1]['k']}"
                 f"{f', above the strict {sci(BOUND)} bound at every K' if all(x['adv_error'] >= BOUND for x in sweep) else ''}." if sweep else "")
        base = self.start_point(h.branch)[0]
        fallback = {"cause": f"{h.judgment.reason} {trend}".strip(),
                    "pivot": f"Drop {ruled_out(h.params, base)}, keep the other parameters, and test whether the gain survives."}
        prose = self.ask_json([{"role": "system", "content": "You write terse, factual post-mortems for GPU kernel experiments. Use only the numbers given. Reply with only JSON: {\"cause\": \"<two sentences on the mechanism>\", \"pivot\": \"<one sentence on what to try next>\"}"},
                               {"role": "user", "content": f"Error metric: max |y - ref| / max(|ref|, 1) against an FP32 reference. The strict bound is {sci(BOUND)}, the Speed lens tolerates {sci(hyp.SPEED_TOLERANCE)} on the standard batch only.\nHypothesis {h.id}: {h.claim}\nChange: {change(base, h.params)}\nExpected: {expected}\nObserved: {observed}\n"
                                                           f"Verdict reason: {h.judgment.reason}\nError vs K sweep on adversarial magnitudes: {json.dumps(sweep)}"}],
                              self.strong, f"pm-{h.id}", ("cause", "pivot")) or fallback
        self.avoid.append(f"{ruled_out(h.params, base)} ({h.id} rejected: {h.judgment.reason})")
        pm = {"hypothesis": h.id, "expected": expected, "observed": observed, "cause": prose["cause"], "pivot": prose["pivot"], "sweep": sweep}
        self.pms[h.id] = pm
        self.log.emit("post_mortem", post_mortem=pm)

    def summarize(self, h: Hyp, b: Branch, next_step: str):
        m = h.metrics
        before_p, before = self.prior(h)
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
        margins = [c for c in j.checks if c.passed is not None]
        confidence = "low" if j.provisional or j.verdict == "inconclusive" else "high" if all(c.passed is not None for c in j.checks) else "medium"
        reason = (f"Run to run latency noise is about {self.band * 100:.1f}% and every check cleared it." if confidence == "high"
                  else "A check sits inside the noise band." if j.verdict == "inconclusive"
                  else "The adversarial error is measured but not gated by any lens yet." if j.provisional
                  else f"{len(margins)} of {len(j.checks)} checks are outside the noise band.")
        prose = self.ask_json([{"role": "system", "content": "You write terse, factual research summaries for GPU kernel experiments. Use only the numbers given. Reply with only JSON: {\"uncertainties\": [\"<short item>\", \"<short item>\"], \"would_change_mind\": \"<one sentence naming a specific result>\"}"},
                               {"role": "user", "content": f"Hypothesis {h.id}: {h.claim}\nChange: {change(before_p, h.params)}\nVerdict: {verdict}\nMeasured: {json.dumps(m.contract())}"}],
                              self.strong, f"sum-{h.id}-{len(h.history)}", ("uncertainties", "would_change_mind"))
        summary = {
            "branch": b.id,
            "tried": f"{h.claim} ({change(before_p, h.params)})" + (f" {h.note}" if h.note else ""),
            "moved": moved,
            "verdict": verdict,
            "confidence": confidence,
            "confidence_reason": reason,
            "uncertainties": (prose or {}).get("uncertainties") or ["Measured on one input shape (2048 cubed) only."],
            "would_change_mind": (prose or {}).get("would_change_mind") or "A rerun outside the noise band in the other direction.",
            "next_step": next_step,
        }
        self.summaries[b.id] = summary
        self.log.emit("summary", summary=summary)

    def prior(self, h: Hyp):
        """The configuration this hypothesis started from, excluding itself."""
        ok = [x for x in self.hyps if x is not h and x.judgment and x.judgment.verdict == "confirmed" and not x.judgment.provisional
              and x.branch in self.lineage(h.branch) and self.hyps.index(x) < self.hyps.index(h)]
        best = min(ok, key=lambda x: x.metrics.latency_ms) if ok else None
        return (best.params, best.metrics) if best else (NAIVE, self.naive)
