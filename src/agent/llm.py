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
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

load_dotenv(ROOT / ".env")
FAST_MODEL = os.environ.get("FAST_MODEL", "openai/gpt-oss-20b")
STRONG_MODEL = os.environ.get("STRONG_MODEL", "google/gemma-4-31b-it")
# Gemini is the default provider when GEMINI_API_KEY is set, NIM (the models above) is the fallback
GEMINI_FAST_MODEL = os.environ.get("GEMINI_FAST_MODEL", "gemini-3.5-flash-lite")
GEMINI_STRONG_MODEL = os.environ.get("GEMINI_STRONG_MODEL", "gemini-3.8-flash")
# OmniRoute is a local OpenAI compatible gateway that routes an intent alias such as auto/coding to a configured provider.
# It is the first choice for both roles unless OMNIROUTE=0, Gemini and then NIM are the fallbacks.
OMNIROUTE_URL = os.environ.get("OMNIROUTE_URL", "http://localhost:20128/v1")
OMNIROUTE_MODEL = os.environ.get("OMNIROUTE_MODEL", "auto/coding")
PROVIDERS = {"nim": (BASE_URL, "NVIDIA_API_KEY"), "gemini": (GEMINI_URL, "GEMINI_API_KEY"), "omniroute": (OMNIROUTE_URL, "OMNIROUTE_API_KEY")}

# Per-model request extras. Nemotron reasons by default and spends the whole budget on reasoning_content.
EXTRA = {
    "nvidia/nemotron-3.5-lightning-30b-a3b": {"chat_template_kwargs": {"enable_thinking": False}},
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    # thinking cannot be turned off on Gemini 3.x, low is the cheapest documented level
    GEMINI_FAST_MODEL: {"reasoning_effort": "low"},
    GEMINI_STRONG_MODEL: {"reasoning_effort": "low"},
}


def available(provider):
    if provider == "omniroute":
        return os.environ.get("OMNIROUTE") != "0"
    return bool(os.environ.get(PROVIDERS[provider][1]))


class Client:
    def __init__(self, timeout=120, max_retries=2, cache=True, provider="nim"):
        url, key = PROVIDERS[provider]
        # a local OmniRoute does not check the key unless REQUIRE_API_KEY is set on it
        self.api = OpenAI(base_url=url, api_key=os.environ.get(key, "unused") if provider == "omniroute" else os.environ[key], timeout=timeout,
                          max_retries=max_retries)
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
