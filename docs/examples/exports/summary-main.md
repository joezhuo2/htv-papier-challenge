# Branch summary: main

**What I tried:** Block_m 16 to 128, block_n 16 to 128, block_k 16 to 64, warps 1 to 4, stages 1 to 4, giving 128x128x64 tiles, 4 warps, 4 stages, FP32 accumulation, exact GELU, is at least 2.00x faster than naive, with standard max error under 1.0e-3, the default bound. (block_m 16 to 128, block_n 16 to 128, block_k 16 to 64, warps 1 to 4, stages 1 to 4)

| metric | before | after | unit |
|---|---|---|---|
| latency | 2.009 | 0.7827 | ms |
| throughput | 8.552 | 21.95 | TFLOPS |
| max error | 0.000493 | 0.000493 |  |
| adversarial error | 0.0005225 | 0.0005225 |  |

**Verdict:** Confirmed: 2.52x (noise +/- 0.12x) against a predicted >= 2.00x. Prediction holds.

**Confidence:** high. 2.52x vs naive clears the 2.00x threshold by 25.8%, against this run's measured spread of +/- 4.9% (decision band +/- 4.9%); every error check passed.

**Uncertainties:**
- Measured on one shape (2048 x 2048 x 2048) only.
- Adversarial error 5.2e-4 is measured but not gated by any lens yet.

**What would change my mind:** A same-window rerun below 1.90x vs naive.

**Next step:** Stop this branch and wait for a lens directive.
