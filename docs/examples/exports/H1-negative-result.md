# Negative result: H1 rejected

**Hypothesis:** FP16 accumulation on 128x128x64 tiles, 4 warps, 4 stages with exact GELU cuts latency by at least 1.20x against the same config with FP32 accumulation, with standard max error under 2.0e-2, the Speed lens bound.

**Parameters:** `{'block_m': 128, 'block_n': 128, 'block_k': 64, 'warps': 4, 'stages': 4, 'acc': 'fp16', 'gelu': 'exact'}`

**Judged against:** standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision)

**Expected:** Speedup of at least 1.20x vs the best so far with max error under 2.0e-2.

**Observed:** Speedup of 1.54x (+/- 0.10x), standard error 1.3e-2, adversarial error 7.1e-2.

## Error versus K

| K | adversarial error |
|---|---|
| 256 | 1.9e-2 |
| 512 | 2.4e-2 |
| 1024 | 3.2e-2 |
| 2048 | 5.8e-2 |
| 4096 | 8.1e-2 |

**Mechanism:** standard max error 1.3e-2 missed < 1.0e-3; adversarial max error 7.1e-2 missed < 1.0e-3. Adversarial error goes from 1.9e-2 at K=256 to 8.1e-2 at K=4096, above the strict 1.0e-3 bound at every K.

**Pivot:** Drop FP16 accumulation, keep the other parameters, and test whether the gain survives.
