import json
from pathlib import Path

from harness.kernel import Params

BAND_FILE = Path(__file__).with_name("noise_band.json")
CONFIG = Params(block_m=128, block_n=128, block_k=32, warps=4, stages=3)


def band():
    """Relative latency noise band measured by tests/test_harness_gpu.py::test_noise_band."""
    return json.loads(BAND_FILE.read_text())["band"]
