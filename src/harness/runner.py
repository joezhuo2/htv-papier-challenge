import json
import os
import select
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

from harness.metrics import Metrics

RUNS = Path(__file__).resolve().parents[2] / "runs"
TIMEOUT_S = 15
STARTUP_S = 30
CMD = [sys.executable, "-m", "harness.experiment"]


def run(spec, run_id=None, timeout=TIMEOUT_S, cmd=CMD, runs=RUNS) -> Metrics:
    """Runs one experiment in a fresh process and logs evidence under runs/<run_id>/.

    The hard timeout starts when the child reports ready (torch imported, CUDA initialized), since startup alone takes ~9 s here.
    On timeout the whole process group is killed.
    """
    run_id = run_id or uuid.uuid4().hex[:8]
    t = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, start_new_session=True,
    )
    readable = bool(select.select([proc.stdout], [], [], STARTUP_S)[0])
    ready = readable and proc.stdout.readline().strip() == "ready"
    startup_s = time.monotonic() - t
    timed_out = not readable
    if ready:
        try:
            stdout, stderr = proc.communicate(json.dumps(spec), timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
    if timed_out:
        os.killpg(proc.pid, signal.SIGKILL)
    if timed_out or not ready:
        stdout, stderr = proc.communicate()

    lines = stdout.strip().splitlines()
    if ready and lines and not timed_out and proc.returncode == 0:
        m = Metrics.from_dict(json.loads(lines[-1]))
    else:
        error = ("timeout" if ready else "startup timeout") if timed_out else f"exit {proc.returncode}: {stderr.strip()[-2000:]}"
        m = Metrics(target=spec["target"], params=spec.get("params"), error=error)
    m.run_id, m.startup_s, m.wall_s, m.timed_out = run_id, round(startup_s, 2), round(time.monotonic() - t, 2), timed_out

    d = runs / run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "spec.json").write_text(json.dumps(spec, indent=2))
    (d / "stdout.txt").write_text(stdout)
    (d / "stderr.txt").write_text(stderr)
    (d / "metrics.json").write_text(json.dumps(m.dict(), indent=2))
    return m
