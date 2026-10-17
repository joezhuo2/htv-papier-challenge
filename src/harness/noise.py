import json
from pathlib import Path

from harness.kernel import Params

BAND_FILE = Path(__file__).with_name("noise_band.json")
HEADROOM_FILE = Path(__file__).with_name("headroom.json")
CONFIG = Params(block_m=128, block_n=128, block_k=32, warps=4, stages=3)


def band():
    """Relative latency noise band measured by tests/test_harness_gpu.py::test_noise_band."""
    return json.loads(BAND_FILE.read_text())["band"]


def headroom():
    """Same-window speedups of the best sweep configuration vs naive and vs eager over N processes, written by harness.stability."""
    return json.loads(HEADROOM_FILE.read_text())
