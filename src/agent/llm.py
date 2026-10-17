import hashlib
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "runs" / "llm_cache"
BASE_URL = "https://integrate.api.nvidia.com/v1"

load_dotenv(ROOT / ".env")
FAST_MODEL = os.environ.get("FAST_MODEL", "openai/gpt-oss-20b")
STRONG_MODEL = os.environ.get("STRONG_MODEL", "google/gemma-4-31b-it")

# Per-model request extras. Nemotron reasons by default and spends the whole budget on reasoning_content.
EXTRA = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": {"chat_template_kwargs": {"enable_thinking": False}},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
}


class Client:
    def __init__(self, timeout=120, max_retries=2, cache=True):
        self.api = OpenAI(base_url=BASE_URL, api_key=os.environ["NVIDIA_API_KEY"], timeout=timeout, max_retries=max_retries)
        self.cache = cache

    def complete(self, messages, model=FAST_MODEL, temperature=0.6, max_tokens=1024, tag=""):
        """Returns {"text", "latency_s", "cached"}. tag separates cache entries for repeated trials of the same prompt."""
        key = hashlib.sha256(json.dumps([model, messages, temperature, max_tokens, tag]).encode()).hexdigest()[:24]
        path = CACHE / f"{key}.json"
        if self.cache and path.exists():
            return json.loads(path.read_text()) | {"cached": True}
        t = time.monotonic()
        r = self.api.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                             extra_body=EXTRA.get(model, {}))
        out = {"text": r.choices[0].message.content or "", "latency_s": round(time.monotonic() - t, 2), "model": model}
        if self.cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(out))
        return out | {"cached": False}
