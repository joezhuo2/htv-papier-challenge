"""uv run python -m harness.eager_error: where PyTorch eager's error against the FP32 reference comes from, one rounding at a time.
Writes runs/eager_error.json. Quoted in docs/results.md."""

import json

import torch

from harness import data
from harness.runner import RUNS

F = torch.nn.functional


def err(y, ref):
    return ((y.float() - ref).abs() / ref.abs().clamp_min(1)).max().item()


def stages(x, w, b, ref):
    z = x.float() @ w.float() + b.float()
    mm = x @ w
    out = {
        "round once at the end": err(F.gelu(z).half(), ref),
        "fp16 matmul output": err(F.gelu(mm.float() + b.float()).half(), ref),
        "fp16 matmul and bias add": err(F.gelu((mm + b).float()).half(), ref),
        "eager (fp16 matmul, bias add and gelu)": err(data.eager(x, w, b), ref),
    }
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    out["eager, reduced precision reduction off"] = err(data.eager(x, w, b), ref)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
    return out


if __name__ == "__main__":
    res = {}
    for name, M, N, K, scale in data.CASES:
        x, w, b = data.make(M, N, K, scale)
        res[name] = stages(x, w, b, data.reference(x, w, b))
    (RUNS / "eager_error.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({c: {k: f"{v:.2e}" for k, v in r.items()} for c, r in res.items()}, indent=2))
