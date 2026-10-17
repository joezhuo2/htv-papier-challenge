import torch

# (name, M, N, K, pre-activation scale). The standard batch is first and is the one that is timed.
CASES = [
    ("standard", 2048, 2048, 2048, 1.0),
    ("large_k", 512, 512, 16384, 1.0),
    ("large_mag", 2048, 2048, 2048, 8.0),
]
# max |y - ref| / max(|ref|, 1): absolute for small outputs, relative for large ones, since fp16 output rounding alone is ~2e-3 absolute at |y| ~ 4.
BOUND = 1e-3


def make(M, N, K, scale, seed=0):
    """fp16 inputs where x @ w + b has std ~= scale, so outputs are O(scale)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", generator=g) * scale
    w = torch.randn(K, N, device="cuda", generator=g) / K**0.5
    b = torch.randn(N, device="cuda", generator=g) * 0.1 * scale
    return x.half(), w.half(), b.half()


def reference(x, w, b):
    torch.backends.cuda.matmul.allow_tf32 = False
    return torch.nn.functional.gelu(x.float() @ w.float() + b.float())


def eager(x, w, b):
    return torch.nn.functional.gelu(x @ w + b)
