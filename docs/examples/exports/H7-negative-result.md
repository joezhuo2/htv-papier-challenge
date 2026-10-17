# Negative result: H7 rejected

**Hypothesis:** Block_k 64 to 128, warps 4 to 8, stages 4 to 2, giving 128x128x128 tiles, 8 warps, 2 stages, FP32 accumulation, exact GELU, is at least 1.03x faster than the best configuration on this branch, with standard max error under 1.0e-3, the Precision lens bound.

**Parameters:** `{'block_m': 128, 'block_n': 128, 'block_k': 128, 'warps': 8, 'stages': 2, 'acc': 'fp32', 'gelu': 'exact'}`

**Judged against:** standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision)

**Expected:** Speedup of at least 1.03x vs the best so far with max error under 1.0e-3.

**Observed:** 0.95x vs the parent (+/- 0.04x) at 0.825 ms, standard error 4.9e-4.

**Mechanism:** It ran at 0.95x vs its parent, slower than what it was timed against. Changed block_k 64 to 128, warps 4 to 8, stages 4 to 2 (255 to 166 registers and 24 to 0 spilled words per thread). Likely cause: registers fell from 255 to 166 and spills from 24 to 0 words per thread, so block_k 64 to 128, warps 4 to 8, stages 4 to 2 did relieve register pressure, but that was not the bottleneck; what the change costs instead: a longer K step loads more per iteration and raises shared memory per stage; more warps split the same tile thinner, so each thread does less math per load; a shallower pipeline leaves less load latency hidden behind the math.

**Pivot:** Revert block_k 64 to 128, warps 4 to 8, stages 4 to 2: register pressure is not what limits this config and try block_n 128 to 64 next.
