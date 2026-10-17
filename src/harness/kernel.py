from dataclasses import asdict, dataclass

import torch
import triton
import triton.language as tl

ACC = ("fp32", "fp16")
GELU = ("exact", "approx")
GROUP_M = 8
SMEM_LIMIT = 101376  # bytes of shared memory per block on the RTX 5060 (Triton's reported hardware limit)


def smem_bytes(p):
    """Shared memory the pipelined FP16 tiles need, (stages - 1) buffers of an x tile and a w tile. Matches Triton's
    OutOfResources figure for every logged failure, for example 147456 for 128x256x64 with 4 stages."""
    return (p.stages - 1) * (p.block_m * p.block_k + p.block_k * p.block_n) * 2


@dataclass(frozen=True)
class Params:
    """The constrained edit surface, matching Params in src/contract/types.ts. The agent may only change these fields."""

    block_m: int = 16
    block_n: int = 16
    block_k: int = 16
    warps: int = 1
    stages: int = 1
    acc: str = "fp32"
    gelu: str = "exact"

    def __post_init__(self):
        for name in ("block_m", "block_n", "block_k"):
            v = getattr(self, name)
            if v < 16 or v > 256 or v & (v - 1):
                raise ValueError(f"{name} must be a power of two in [16, 256], got {v}")
        if self.warps not in (1, 2, 4, 8, 16):
            raise ValueError(f"warps must be one of 1, 2, 4, 8, 16, got {self.warps}")
        if not 1 <= self.stages <= 6:
            raise ValueError(f"stages must be in [1, 6], got {self.stages}")
        if self.acc not in ACC or self.gelu not in GELU:
            raise ValueError(f"acc must be in {ACC} and gelu in {GELU}")

    def dict(self):
        return asdict(self)


NAIVE = Params()


@triton.jit
def _gelu(z, APPROX: tl.constexpr):
    if APPROX:
        # tanh(u) = 2 * sigmoid(2u) - 1
        u = 0.7978845608028654 * (z + 0.044715 * z * z * z)
        return z * tl.sigmoid(2.0 * u)
    else:
        return 0.5 * z * (1.0 + tl.math.erf(z * 0.7071067811865476))


@triton.jit
def fused_kernel(
    x_ptr, w_ptr, b_ptr, y_ptr, M, N, K,
    sxm, sxk, swk, swn, sym, syn,
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GM: tl.constexpr,
    ACC16: tl.constexpr, APPROX: tl.constexpr,
):
    pid = tl.program_id(0)
    npm = tl.cdiv(M, BM)
    npn = tl.cdiv(N, BN)
    per_group = GM * npn
    first = (pid // per_group) * GM
    gsize = min(npm - first, GM)
    pm = first + (pid % gsize)
    pn = (pid % per_group) // gsize

    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    xp = x_ptr + (rm % M)[:, None] * sxm + rk[None, :] * sxk
    wp = w_ptr + rk[:, None] * swk + (rn % N)[None, :] * swn

    acc = tl.zeros((BM, BN), dtype=tl.float16 if ACC16 else tl.float32)
    for k in range(tl.cdiv(K, BK)):
        xv = tl.load(xp, mask=rk[None, :] < K - k * BK, other=0.0)
        wv = tl.load(wp, mask=rk[:, None] < K - k * BK, other=0.0)
        acc = tl.dot(xv, wv, acc, out_dtype=tl.float16 if ACC16 else tl.float32)
        xp += BK * sxk
        wp += BK * swk

    z = acc.to(tl.float32) + tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)[None, :]
    z = _gelu(z, APPROX)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(y_ptr + rm[:, None] * sym + rn[None, :] * syn, z.to(tl.float16), mask=mask)


def run(p: Params, x, w, b):
    M, K = x.shape
    N = w.shape[1]
    y = torch.empty((M, N), device=x.device, dtype=torch.float16)
    grid = (triton.cdiv(M, p.block_m) * triton.cdiv(N, p.block_n),)
    fused_kernel[grid](
        x, w, b, y, M, N, K,
        x.stride(0), x.stride(1), w.stride(0), w.stride(1), y.stride(0), y.stride(1),
        BM=p.block_m, BN=p.block_n, BK=p.block_k, GM=GROUP_M,
        ACC16=p.acc == "fp16", APPROX=p.gelu == "approx",
        num_warps=p.warps, num_stages=p.stages,
    )
    return y
