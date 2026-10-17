"""Builds a production Orchestrator: real harness runs, NIM models, events under runs/sessions/<id>/."""

import os
import time

from agent import papiers
from agent.events import Log
from agent.llm import (
    FAST_MODEL,
    GEMINI_FAST_MODEL,
    GEMINI_STRONG_MODEL,
    STRONG_MODEL,
    Client,
    available,
)
from agent.orchestrator import Orchestrator
from harness import noise, runner

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "180"))


def chains():
    """Role to [(provider, model)] in order of preference: Gemini first when its key is set, NIM as the fallback."""
    gem = available("gemini")
    return {
        "fast": ([("gemini", GEMINI_FAST_MODEL)] if gem else []) + [("nim", FAST_MODEL)],
        "strong": ([("gemini", GEMINI_STRONG_MODEL)] if gem else []) + [("nim", STRONG_MODEL)],
    }


def asker(log, roles=None):
    """ask(messages, role, tag): tries each provider for the role in turn and logs a fallback, so a Gemini outage costs one retry."""
    roles = roles or chains()
    clients = {}

    def ask(messages, role, tag):
        last = None
        for i, (provider, model) in enumerate(roles[role]):
            try:
                client = clients.setdefault(provider, Client(timeout=LLM_TIMEOUT_S, max_retries=1, provider=provider))
                text = client.complete(messages, model=model, tag=tag, max_tokens=2048)["text"]
                if i:
                    log.emit("agent_log", text=f"{roles[role][0][1]} failed ({type(last).__name__}), answered by {model} on {provider}.")
                return text
            except Exception as e:  # noqa: BLE001, try the next provider, the last failure propagates
                last = e
        raise last

    return ask


def new(sid=None, **kw) -> Orchestrator:
    sid = sid or time.strftime("%Y%m%d-%H%M%S")
    log = Log(runner.RUNS / "sessions" / sid / "events.jsonl")
    return Orchestrator(lambda spec, run_id: runner.run(spec, run_id=run_id), asker(log), log, f"sessions/{sid}",
                        "fast", "strong", noise.band(), context=papiers.search if papiers.binary() else None,
                        headroom=noise.headroom(), **kw)
