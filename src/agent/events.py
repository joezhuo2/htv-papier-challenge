"""Session event log: every DemoEvent goes to runs/sessions/<id>/events.jsonl (the source of truth) and to live listeners."""

import json
import threading
import time
from pathlib import Path


class Log:
    def __init__(self, path: Path, clock=time.monotonic):
        self.path = path
        self.clock = clock
        self.t0 = clock()
        self.events = []
        self.listeners = []
        self.lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)
        self.file = path.open("a")

    def emit(self, type, **payload):
        with self.lock:
            e = {"t": int((self.clock() - self.t0) * 1000), "type": type, **payload}
            self.events.append(e)
            self.file.write(json.dumps(e) + "\n")
            self.file.flush()
            for fn in list(self.listeners):
                fn(e)
        return e

    def subscribe(self, fn):
        """Returns the events so far and registers fn for new ones, atomically, so a client never misses or repeats an event."""
        with self.lock:
            self.listeners.append(fn)
            return list(self.events)

    def unsubscribe(self, fn):
        with self.lock:
            self.listeners.remove(fn)

    def close(self):
        self.file.close()
