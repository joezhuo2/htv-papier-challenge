"""uv run uvicorn server.app:app --port 8000

One research session per server process. The session waits for a start command from the UI (or starts at once with AUTOSTART=1)
and then runs the orchestrator in a background thread. The default start keeps the table demo cap (MAX_HYPS, default 1). A start
with record: true is the recording mode: the full loop for up to RECORD_HYPS hypotheses (default 8) or until Stop, with the log
saved as a replay. POST /restart stops the current session and replaces it with a new one waiting for Start. Each WebSocket client first gets the session history as one array, then
every new event, and may send Command objects. GET /preflight checks the GPU, the NIM key and the Papiers token for real.
GET /papiers/writes previews the Papiers write-back of a finished run (the live session, or ?file= a replay under demo/ or runs/), and
POST /papiers/writes applies the approved items, for the finished live session only: a replay or mock log can never write.
"""

import asyncio
import json
import os
import subprocess
import threading
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[2]
REPLAY_DIRS = ("demo", "runs")

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


def check_omniroute():
    from agent.llm import OMNIROUTE_MODEL, OMNIROUTE_URL, available
    if not available("omniroute"):
        return False, "disabled with OMNIROUTE=0, Gemini and NIM are used"
    r = httpx.get(f"{OMNIROUTE_URL}/models", headers={"Authorization": f"Bearer {os.environ.get('OMNIROUTE_API_KEY', 'unused')}"}, timeout=5)
    ids = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
    listed = "is listed" if OMNIROUTE_MODEL in ids else f"is not among its {len(ids)} models"
    return r.status_code == 200 and OMNIROUTE_MODEL in ids, f"{OMNIROUTE_URL}/models answered {r.status_code}, {OMNIROUTE_MODEL} {listed}"


def check_papiers():
    from agent import papiers
    exe = papiers.binary()
    if not exe:
        return False, "papiers CLI not found"
    n = papiers.Library().counts()
    detail = f"the CLI sees {n['highlight']} highlights, {n['note']} notes and {n['document']} documents"
    if n["highlight"] == 0:
        return True, f"{detail}. No highlights, so hypotheses fall back to passages and conversations", True
    return True, detail


CHECKS = {"gpu": check_gpu, "omniroute": check_omniroute, "gemini": check_gemini, "nim": check_nim, "papiers": check_papiers}


def run_check(fn):
    try:
        ok, detail, *warn = fn()
    except Exception as e:  # noqa: BLE001, a failed check is reported, not raised
        ok, detail, warn = False, f"{type(e).__name__}: {str(e)[:200]}", []
    return {"ok": ok, "detail": detail} | ({"warn": True} if warn and warn[0] else {})


class Approval(BaseModel):
    session: str
    ids: list[str]


def replay_file(file):
    path = (ROOT / file.lstrip("/")).resolve()
    if not any(path.is_relative_to(ROOT / d) for d in REPLAY_DIRS) or not path.is_file():
        raise HTTPException(404, f"no replay at {file}")
    return path


def build():
    from agent import session
    cap = os.environ.get("MAX_HYPS", "1")
    return session.new(main_iters=int(os.environ.get("MAIN_ITERS", "2")), branch_iters=int(os.environ.get("BRANCH_ITERS", "3")),
                       idle_s=float(os.environ.get("IDLE_S", "600")), max_hyps=int(cap) if cap else None,
                       help_deadline_s=float(os.environ.get("HELP_DEADLINE_S", "45")), directive_wait_s=float(os.environ.get("DIRECTIVE_WAIT_S", "90")),
                       end_wait_s=float(os.environ.get("END_WAIT_S", "15")))


def create(factory=None, checks=CHECKS, autostart=None):
    factory = factory or build
    state = {"socks": set()}
    autostart = os.environ.get("AUTOSTART") == "1" if autostart is None else autostart

    record_hyps = int(os.environ.get("RECORD_HYPS", "8"))

    def start(record=False):
        if state.get("thread"):
            return
        if record:
            state["orch"].set_record(record_hyps)
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

    def live_run(session=None):
        """The live session's (events, session id, ledger path, why apply is refused or None)."""
        from agent import writeback
        orch = state["orch"]
        sid = orch.session.split("/")[-1]
        if session not in (None, sid):
            raise HTTPException(404, f"{session} is not this server's live session ({sid})")
        events = list(orch.log.events)
        why = ("The session has not finished." if not orch.done else "Mock data never writes to Papiers." if any(e.get("mock") for e in events)
               else None)
        return events, sid, orch.log.path.with_name(writeback.LEDGER), why

    @app.get("/papiers/writes")
    def writes(file: str | None = None):
        from agent import writeback
        if file:
            path = replay_file(file)
            ledger = path.with_name(writeback.LEDGER)
            return writeback.preview(writeback.load(path), path.parent.name, ledger if ledger.exists() else None) | {
                "apply": False, "reason": "A replay is read-only: Apply works only on the live session that produced the run."}
        events, sid, ledger, why = live_run()
        if not any(e["type"] == "session_finished" for e in events):
            raise HTTPException(409, "The session has not finished, write-back is available after the final verdicts.")
        return writeback.preview(events, sid, ledger) | {"apply": why is None, "reason": why}

    @app.post("/papiers/writes")
    def write(body: Approval):
        from agent import writeback
        events, sid, ledger, why = live_run(body.session)
        if why:
            raise HTTPException(403, why)
        try:
            results = writeback.apply(events, sid, body.ids, ledger)
        except ValueError as e:
            raise HTTPException(400, str(e)) from None
        except FileNotFoundError as e:
            raise HTTPException(503, str(e)) from None
        state["orch"].log.emit("papiers_written", results=[{k: r[k] for k in ("id", "status", "error") if k in r} for r in results])
        return {"session": sid, "results": results}

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
