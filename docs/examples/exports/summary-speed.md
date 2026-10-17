# Branch summary: speed

**What I tried:** FP16 accumulation on 128x128x64 tiles, 4 warps, 4 stages with exact GELU cuts latency by at least 1.20x against the same config with FP32 accumulation, with standard max error under 2.0e-2, the Speed lens bound. (acc fp32 to fp16)

| metric | before | after | unit |
|---|---|---|---|
| latency | 0.7827 | 0.514 | ms |
| throughput | 21.95 | 33.43 | TFLOPS |
| max error | 0.000493 | 0.01318 |  |
| adversarial error | 0.0005225 | 0.07083 |  |

**Verdict:** Rejected: 1.54x (noise +/- 0.10x) against a predicted >= 1.20x. standard max error 1.3e-2 missed < 1.0e-3; adversarial max error 7.1e-2 missed < 1.0e-3.

**Confidence:** high. Standard max error 1.3e-2 against < 1.0e-3. Errors are deterministic with fixed-seed inputs, so a rerun gives the same value.

**Uncertainties:**
- Measured on one shape (2048 x 2048 x 2048) only.

**What would change my mind:** An error under 1.0e-3 on the same inputs, which would mean the harness changed.

**Next step:** Fork: keep what worked in H1 and drop FP16 accumulation.
