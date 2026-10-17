# Branch summary: prec

**What I tried:** Block_k 64 to 128, warps 4 to 8, stages 4 to 2, giving 128x128x128 tiles, 8 warps, 2 stages, FP32 accumulation, exact GELU, is at least 1.03x faster than the best configuration on this branch, with standard max error under 1.0e-3, the Precision lens bound. (block_k 64 to 128, warps 4 to 8, stages 4 to 2)

| metric | before | after | unit |
|---|---|---|---|
| latency | 0.7827 | 0.825 | ms |
| throughput | 21.95 | 20.82 | TFLOPS |
| max error | 0.000493 | 0.000493 |  |
| adversarial error | 0.0005225 | 0.0005225 |  |

**Verdict:** Rejected: 0.95x (noise +/- 0.04x) against a predicted >= 1.03x. Slower than its parent: 0.95x vs best (noise +/- 0.04x) against >= 1.03x. Within every error bound.

**Confidence:** medium. 0.95x vs the parent misses the 1.03x threshold by 7.7%, outside this run's measured spread of +/- 4.4% (decision band +/- 4.4%).

**Uncertainties:**
- Measured on one shape (2048 x 2048 x 2048) only.

**What would change my mind:** A same-window rerun at or above 1.08x vs the parent.

**Next step:** Stop this branch and wait for a lens directive.
