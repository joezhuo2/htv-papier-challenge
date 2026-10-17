"""uv run uvicorn server.app:app --port 8000

One research session per server process. The orchestrator runs in a background thread. Each WebSocket client first gets the
session history as one array (it starts with session_started, so the UI resets), then every new event, and may send Command objects.
"""

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

COMMANDS = {"add_directive", "answer_help", "approve_hybrid"}


def valid(cmd):
    if not isinstance(cmd, dict) or cmd.get("type") not in COMMANDS:
        return False
    if cmd["type"] == "add_directive":
        return cmd.get("lens") in ("speed", "precision") and isinstance(cmd.get("text"), str) and bool(cmd["text"].strip())
    return True


def create(factory=None):
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        if factory is None:
            from agent import session
            orch = session.new(main_iters=int(os.environ.get("MAIN_ITERS", "2")), branch_iters=int(os.environ.get("BRANCH_ITERS", "3")),
                               idle_s=float(os.environ.get("IDLE_S", "600")))
        else:
            orch = factory()
        state["orch"] = orch
        thread = threading.Thread(target=orch.run, daemon=True)
        thread.start()
        yield
        orch.stop()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    def health():
        orch = state["orch"]
        return {"session": orch.session, "events": len(orch.log.events), "log": str(orch.log.path)}

    @app.websocket("/ws")
    async def ws(sock: WebSocket):
        await sock.accept()
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
                if valid(cmd):
                    orch.command(cmd)
        except WebSocketDisconnect:
            pass
        finally:
            sender.cancel()
            orch.log.unsubscribe(push)

    return app


app = create()
