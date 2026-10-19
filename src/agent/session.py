"""Builds a production Orchestrator: real harness runs, NIM models, events under runs/sessions/<id>/."""

import os
import time

from agent import papiers
from agent.events import Log
from agent.llm import (
    FAST_MODEL,
    GEMINI_FAST_MODEL,
    GEMINI_STRONG_MODEL,
    OMNIROUTE_MODEL,
    STRONG_MODEL,
    Client,
    available,
)
from agent.orchestrator import Orchestrator
from harness import noise, runner

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "180"))
# post-mortem prose is optional (a factual template stands in), so it gets one fast try instead of a chain of slow fallbacks
STRONG_TIMEOUT_S = float(os.environ.get("STRONG_TIMEOUT_S", "15"))
TIMEOUTS = {"fast": LLM_TIMEOUT_S, "strong": STRONG_TIMEOUT_S}
DOWN_S = 300.0
# errors that will not fix themselves within a session: the route is dropped for good
FATAL = ("PermissionDeniedError", "BadRequestError", "AuthenticationError", "NotFoundError")


def chains():
    """Role to [(provider, model)] in order of preference. Hypotheses: OmniRoute (unless OMNIROUTE=0), then Gemini when its key is
    set, then NIM. Post-mortems: pinned to Gemini's strong model when its key is set, NIM otherwise."""
    omni = [("omniroute", OMNIROUTE_MODEL)] if available("omniroute") else []
    gem = available("gemini")
    return {
        "fast": omni + ([("gemini", GEMINI_FAST_MODEL)] if gem else []) + [("nim", FAST_MODEL)],
        "strong": [("gemini", GEMINI_STRONG_MODEL)] if gem else [("nim", STRONG_MODEL)],
    }


def asker(debug, roles=None):
    """ask(messages, role, tag): tries each live (provider, model) route for the role in turn. A route that fails is skipped for DOWN_S, or for the
    rest of the session on a permission or bad request error, so a broken gateway costs one failed call, not one per call.
    Failovers are written to the debug file, not the replay."""
    roles = roles or chains()
    clients, down = {}, {}

    def note(text):
        with open(debug, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} {text}\n")

    def ask(messages, role, tag):
        chain = roles[role]
        live = [x for x in chain if down.get(x, 0) <= time.monotonic()] or chain[-1:]
        last = None
        for provider, model in live:
            t = time.monotonic()
            try:
                final = (provider, model) == chain[-1]
                client = clients.setdefault((provider, role), Client(timeout=TIMEOUTS[role], max_retries=1 if final else 0, provider=provider))
                out = client.complete(messages, model=model, tag=tag, max_tokens=2048)
                note(f"{role}: {model} on {provider} answered in {time.monotonic() - t:.1f} s" + (" (cache)" if out.get("cached") else "")
                     + (f" after {type(last).__name__} on an earlier route" if last else ""))
                return out["text"]
            except Exception as e:  # noqa: BLE001, try the next provider, the last failure propagates
                last = e
                fatal = type(e).__name__ in FATAL
                down[provider, model] = float("inf") if fatal else time.monotonic() + DOWN_S  # per route: a quota on one model leaves the others usable
                note(f"{role}: {model} on {provider} failed after {time.monotonic() - t:.1f} s ({type(e).__name__}: {str(e)[:160]}), "
                     + ("dropped for the session" if fatal else f"skipped for {DOWN_S:.0f} s"))
        raise last

    return ask


def new(sid=None, **kw) -> Orchestrator:
    if not sid:
        sid = time.strftime("%Y%m%d-%H%M%S")
        while (runner.RUNS / "sessions" / sid).exists():  # a restart within the same second
            sid += "b"
    log = Log(runner.RUNS / "sessions" / sid / "events.jsonl")
    # one library for both, so the [RWR] highlights found at start are never retrieved as context
    lib = papiers.Library() if papiers.binary() else None
    return Orchestrator(lambda spec, run_id: runner.run(spec, run_id=run_id), asker(log.path.with_name("debug.log")), log, f"sessions/{sid}",
                        "fast", "strong", noise.band(), context=lib and lib.retrieve, read_back=lib and lib.prior_verdicts,
                        headroom=noise.headroom(), **kw)
