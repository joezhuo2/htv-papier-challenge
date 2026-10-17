"""Builds a production Orchestrator: real harness runs, NIM models, events under runs/sessions/<id>/."""

import os
import time

from agent import papiers
from agent.events import Log
from agent.llm import FAST_MODEL, STRONG_MODEL, Client
from agent.orchestrator import Orchestrator
from harness import noise, runner

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "180"))


def asker():
    clients = {}

    def ask(messages, model, tag):
        client = clients.setdefault(model, Client(timeout=LLM_TIMEOUT_S, max_retries=1))
        return client.complete(messages, model=model, tag=tag, max_tokens=2048)["text"]

    return ask


def new(sid=None, **kw) -> Orchestrator:
    sid = sid or time.strftime("%Y%m%d-%H%M%S")
    log = Log(runner.RUNS / "sessions" / sid / "events.jsonl")
    return Orchestrator(lambda spec, run_id: runner.run(spec, run_id=run_id), asker(), log, f"sessions/{sid}",
                        FAST_MODEL, STRONG_MODEL, noise.band(), context=papiers.search if papiers.binary() else None, **kw)
