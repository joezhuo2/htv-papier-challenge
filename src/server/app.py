"""uv run uvicorn server.app:app --port 8000

One research session per server process. The session waits for a start command from the UI (or starts at once with AUTOSTART=1)
and then runs the orchestrator in a background thread. The default start keeps the table demo cap (MAX_HYPS, default 1). A start
with record: true is the recording mode: the full loop for up to RECORD_HYPS hypotheses (default 8) or until Stop, with the log
saved as a replay. POST /restart stops the current session and replaces it with a new one waiting for Start. Each WebSocket client first gets the session history as one array, then
every new event, and may send Command objects. GET /preflight checks the GPU, the NIM key and the Papiers token for real.
"""

import asyncio
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

COMMANDS = {"add_directive", "answer_help", "approve_hybrid", "start", "stop", "retry"}


def valid(cmd):
    if not isinstance(cmd, dict) or cmd.get("type") not in COMMANDS:
        return False
    if cmd["type"] == "add_directive":
        return cmd.get("lens") in ("speed", "precision") and isinstance(cmd.get("text"), str) and bool(cmd["text"].strip())
    return True


def check_gpu():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=10, check=False)
    return out.returncode == 0 and bool(out.stdout.strip()), out.stdout.strip() or out.stderr.strip()[:200]


def check_nim():
    from agent.llm import BASE_URL
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return False, "NVIDIA_API_KEY is not set"
    r = httpx.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    return r.status_code == 200, f"{BASE_URL}/models answered {r.status_code}"


def check_gemini():
    from agent.llm import GEMINI_URL, available
    if not available("gemini"):
        return False, "GEMINI_API_KEY is not set, NIM is used"
    r = httpx.get(f"{GEMINI_URL}models", headers={"Authorization": f"Bearer {os.environ['GEMINI_API_KEY']}"}, timeout=10)
    return r.status_code == 200, f"{GEMINI_URL}models answered {r.status_code}"


def check_papiers():
    from agent import papiers
    exe = papiers.binary()
    if not exe:
        return False, "papiers CLI not found"
    papiers.search("Triton", limit=1)
    return True, "search answered"


CHECKS = {"gpu": check_gpu, "gemini": check_gemini, "nim": check_nim, "papiers": check_papiers}


def run_check(fn):
    try:
        ok, detail = fn()
    except Exception as e:  # noqa: BLE001, a failed check is reported, not raised
        ok, detail = False, f"{type(e).__name__}: {str(e)[:200]}"
    return {"ok": ok, "detail": detail}


def build():
    from agent import session
    cap = os.environ.get("MAX_HYPS", "1")
    return session.new(main_iters=int(os.environ.get("MAIN_ITERS", "2")), branch_iters=int(os.environ.get("BRANCH_ITERS", "3")),
                       idle_s=float(os.environ.get("IDLE_S", "600")), max_hyps=int(cap) if cap else None,
                       help_deadline_s=float(os.environ.get("HELP_DEADLINE_S", "45")))


def create(factory=None, checks=CHECKS, autostart=None):
    factory = factory or build
    state = {"socks": set()}
    autostart = os.environ.get("AUTOSTART") == "1" if autostart is None else autostart

    record_hyps = int(os.environ.get("RECORD_HYPS", "8"))

    def start(record=False):
        if state.get("thread"):
            return
        if record:
            state["orch"].record, state["orch"].max_hyps = True, record_hyps
            state["orch"].log.emit("agent_log", text=f"Recording mode: the full loop for up to {record_hyps} hypotheses or until Stop. "
                                                     "Directives and help answers are accepted at any point.")
        state["thread"] = threading.Thread(target=state["orch"].run, daemon=True)
        state["thread"].start()

    @asynccontextmanager
    async def lifespan(app):
        state["orch"] = factory()
        if autostart:
            start()
        yield
        state["orch"].stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        orch = state["orch"]
        return {"session": orch.session, "events": len(orch.log.events), "log": str(orch.log.path), "started": bool(state.get("thread")), "record": orch.record}

    @app.post("/restart")
    async def restart():
        """Stops the current session (an in-flight harness run finishes first, at most about 25 s), disconnects every client and
        starts over with a new session that waits for Start. Clients reload and reconnect to it."""
        old, thread = state["orch"], state.pop("thread", None)
        old.stop()
        if thread:
            await asyncio.to_thread(thread.join, 60)
        if not (thread and thread.is_alive()):
            old.log.close()
        for sock in list(state["socks"]):
            await sock.close()
        state["orch"] = factory()
        return {"session": state["orch"].session}

    @app.get("/preflight")
    def preflight():
        orch = state["orch"]
        return {"started": bool(state.get("thread")), "record": orch.record, "cap": orch.max_hyps, "record_cap": record_hyps,
                "checks": {name: run_check(fn) for name, fn in checks.items()}}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
        state["socks"].add(sock)
        orch = state["orch"]
        loop = asyncio.get_running_loop()
        out: asyncio.Queue = asyncio.Queue()
        push = lambda e: loop.call_soon_threadsafe(out.put_nowait, e)
        history = orch.log.subscribe(push)

        async def send():
            if history:
                await sock.send_text(json.dumps(history))
            while True:
                await sock.send_text(json.dumps(await out.get()))

        sender = asyncio.create_task(send())
        try:
            while True:
                try:
                    cmd = json.loads(await sock.receive_text())
                except json.JSONDecodeError:
                    continue
                if not valid(cmd):
                    continue
                if cmd["type"] == "start":
                    start(cmd.get("record") is True)
                else:
                    orch.command(cmd)
        except (WebSocketDisconnect, RuntimeError):
            pass  # RuntimeError: the socket was closed by a restart
        finally:
            state["socks"].discard(sock)
            sender.cancel()
            orch.log.unsubscribe(push)

    return app


app = create()
