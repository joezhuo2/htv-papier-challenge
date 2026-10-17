# Branch summary: speed2

**What I tried:** Switching to approx GELU and increasing warps to 8 will eliminate register spills and improve instruction throughput. Relative to the pivot config (128x128x64 tiles, 4 warps, 4 stages, FP32 accumulation, exact GELU) it changes warps 4 to 8, gelu exact to approx. (warps 4 to 8, gelu exact to approx)

| metric | before | after | unit |
|---|---|---|---|
| latency | 0.7827 | 0.7295 | ms |
| throughput | 21.95 | 23.55 | TFLOPS |
| max error | 0.000493 | 0.0006922 |  |
| adversarial error | 0.0005225 | 0.0007199 |  |

**Verdict:** Inconclusive: 1.10x (noise +/- 0.08x) against a predicted >= 1.10x. The speedup is inside the noise band around the predicted threshold.

**Confidence:** low. 1.10x vs the parent sits inside this run's measured spread of +/- 7.2% (decision band +/- 7.2%) around the 1.10x threshold.

**Uncertainties:**
- The speedup is within this run's measured spread of +/- 7.2% (decision band +/- 7.2%).
- Measured on one shape (2048 x 2048 x 2048) only.

**What would change my mind:** A same-window rerun clearly above 1.18x or below 1.02x vs the parent.

**Next step:** Stop this branch, the iteration budget is spent.
