"""uv run python -m agent.run [--directive N:lens:text ...]: a headless session.

Each --directive is sent once N hypotheses have verdicts, for example 1:speed:"Push throughput harder" 3:precision:"Max error under 1e-3 on the adversarial batch".
The event log lands in runs/sessions/<id>/events.jsonl and replays in the UI with ?mode=replay&file=/runs/sessions/<id>/events.jsonl.
"""

import argparse
import threading
import time

from agent import session


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directive", action="append", default=[], help="N:lens:text")
    ap.add_argument("--id")
    ap.add_argument("--main-iters", type=int, default=2)
    ap.add_argument("--branch-iters", type=int, default=3)
    ap.add_argument("--idle", type=float, default=5.0, help="seconds without work before the session finishes")
    a = ap.parse_args()

    script = sorted((int(n), lens, text) for n, lens, text in (d.split(":", 2) for d in a.directive))
    orch = session.new(a.id, main_iters=a.main_iters, branch_iters=a.branch_iters, idle_s=a.idle)
    t = threading.Thread(target=orch.run)
    t.start()
    while t.is_alive():
        judged = sum(h.judgment is not None for h in orch.hyps)
        while script and judged >= script[0][0]:
            _, lens, text = script.pop(0)
            orch.command({"type": "add_directive", "lens": lens, "text": text})
        time.sleep(0.2)
    print(orch.log.path)


if __name__ == "__main__":
    main()
