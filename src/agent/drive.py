"""uv run python -m agent.drive: plays both lenses against a running server (uvicorn server.app:app --port 8000) in recording mode.

The recording plan, scripted: Start recording, the Speed directive after H0, the Precision directive once a verdict is provisional,
the tanh GELU hunch 25 s later, keep tuning on every help request after 12 s, and approval of the hybrid after 6 s. Prints every
event. The replay is runs/sessions/<id>/replay.jsonl, named in the server's /health.
"""

import asyncio
import json
import time

import websockets

SPEED = "Push throughput harder than tile tuning alone. I can accept max error up to 2e-2."
PREC = "Accuracy matters more than speed here. Keep max error under 1e-3 on every batch, including the adversarial one."
HUNCH = "Try the tanh approximate GELU. I think it is cheap and the error should stay small."


async def main(url="ws://localhost:8000/ws"):
    async with websockets.connect(url, max_size=None) as ws:
        loop = asyncio.get_running_loop()
        later = lambda s, cmd: loop.call_later(s, lambda: asyncio.ensure_future(ws.send(json.dumps(cmd))))
        await ws.send(json.dumps({"type": "start", "record": True}))
        sent, t0 = set(), time.time()
        while True:
            raw = json.loads(await ws.recv())
            for e in raw if isinstance(raw, list) else [raw]:
                t = e["type"]
                print(f"{time.time() - t0:6.1f} {t} {json.dumps({k: v for k, v in e.items() if k not in ('type', 't')})[:160]}", flush=True)
                if t == "verdict" and "speed" not in sent:
                    sent.add("speed")
                    later(3, {"type": "add_directive", "lens": "speed", "text": SPEED})
                if t == "verdict" and e["provisional"] and "prec" not in sent:
                    sent.add("prec")
                    later(4, {"type": "add_directive", "lens": "precision", "text": PREC})
                if t == "directive_added" and e["directive"]["lens"] == "precision" and "hunch" not in sent:
                    sent.add("hunch")
                    later(25, {"type": "add_directive", "lens": "speed", "text": HUNCH})
                if t == "help_requested" and e["help"]["id"] not in sent:
                    sent.add(e["help"]["id"])
                    later(12, {"type": "answer_help", "help": e["help"]["id"], "option": "tile_search", "by": e["help"]["target"]})
                if t == "hybrid_proposed":
                    later(6, {"type": "approve_hybrid", "branches": e["branches"]})
                if t == "session_finished":
                    return


if __name__ == "__main__":
    asyncio.run(main())
