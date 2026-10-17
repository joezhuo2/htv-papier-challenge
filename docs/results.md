# Results

Every number here comes from a logged run on the build machine (RTX 5060 Laptop GPU, WSL2, PyTorch 2.11 + CUDA 12.8, Triton 3.6). Speedups are same-window: the candidate is timed interleaved with naive, eager and its parent in one child process, and the +/- is that window's measured spread. Logs live in `runs/` (gitignored), so the file names below are local evidence, not repo files.

## Stable numbers

From `harness.stability` (10 separate processes each, `runs/stability.log`, also saved to `src/harness/headroom.json`), plus the approximate GELU runs from live sessions.

| Config | vs naive | vs eager | vs FP32 tiles | Standard error | Adversarial error |
| --- | --- | --- | --- | --- | --- |
| Best FP32 tiles (128x128x32, 4 warps, 3 stages, exact GELU) | 2.75x (2.70 to 2.78) | 1.03x (1.02 to 1.04) | 1.00x | 4.9e-4 | 5.2e-4 |
| Same tiles, approximate (tanh) GELU | 2.79x +/- 0.10x (live1 H9), 2.88x +/- 0.09x (record1 H7) | 1.05x +/- 0.06x, 1.06x +/- 0.06x | 1.03x +/- 0.05x, 1.03x +/- 0.06x (inside the noise band) | 6.9e-4 | 7.2e-4 |
| Same tiles, FP16 accumulation | 4.03x (3.99 to 4.09) | 1.52x (1.49 to 1.54) | 1.48x (1.47 to 1.51) | 1.32e-2 | 7.08e-2 |
| PyTorch eager | about 2.67x (2.75 / 1.03, same window) | 1.00x | | 1.5e-3 | 2.5e-3 |
| Naive Triton start (16x16x16, 1 warp, 1 stage) | 1.00x | about 0.37x | | 4.9e-4 | 5.2e-4 |

The approximate GELU numbers are single same-window runs, not a 10 process stability measurement: it passes 1e-3 on both batches, and its gain over the FP32 tiles (about 3%) is inside the measured noise, so the agent marks it inconclusive. That is the honest result: on this GPU the erf epilogue is not the bottleneck.

## Why PyTorch eager fails the 1e-3 gate and the naive kernel passes

The error metric is `max |y - ref| / max(|ref|, 1)` against an FP32 reference computed from the same FP16 inputs. One rounding of an O(1) output to FP16 costs up to half a unit in the last place, 2^-11, about 4.9e-4, which is exactly the naive kernel's standard error (4.93e-4). The fused Triton kernel keeps the accumulator in FP32 through the bias add and the GELU and rounds to FP16 once, at the final store, so it can never do better or worse than that one rounding.

PyTorch eager runs `gelu(x @ w + b)` as three FP16 operations, and each writes an FP16 tensor. `uv run python -m harness.eager_error` (output in `runs/eager_error.json`) adds one rounding at a time and measures each step against the same reference:

| Step | standard | large_k (K = 16384) | large_mag (8x pre-activations) |
| --- | --- | --- | --- |
| Everything in FP32, round once at the end (what the fused kernel does) | 4.9e-4 | 4.9e-4 | 4.9e-4 |
| Matmul output rounded to FP16, bias and GELU in FP32 | 1.0e-3 | 2.3e-3 | 1.5e-3 |
| Matmul output and bias add both rounded to FP16 | 1.5e-3 | 2.5e-3 | 2.0e-3 |
| PyTorch eager (matmul, bias add and GELU in FP16) | 1.5e-3 | 2.5e-3 | 2.0e-3 |
| Eager with `allow_fp16_reduced_precision_reduction = False` | 1.5e-3 | 1.5e-3 | 2.0e-3 |

So the standard 1.5e-3 is two extra roundings before the GELU (the matmul output and the bias add each cost about one more half ulp), and the third rounding at the GELU output adds nothing measurable to the maximum. The adversarial 2.5e-3 is the large_k case (the adversarial error is the worse of large_k and large_mag): with K = 16384, cuBLAS reduces partial sums in reduced precision, PyTorch's default for FP16 GEMMs, and turning that off brings large_k back to 1.5e-3. So eager is a speed reference, not a correctness reference, and the README and UI say so.

## The designed failure (FP16 accumulation)

- Standard error 1.318e-2 and adversarial 7.083e-2 on every run (fixed-seed inputs), so it passes the Speed tolerance of 2e-2 and fails 1e-3 on both batches.
- Error versus K on large-magnitude inputs: about 1.9e-2 at K=256 rising to 8.1e-2 at K=4096, over 1e-3 at every K.
- Reproduced provisional then rejected, from logged metrics with no new run, in every session where the Precision directive came after the FP16 run: live1 (1.48x +/- 0.05x over its parent), live2 (1.33x), live4 (1.41x), record1 (1.50x), record2 (1.50x). In the presenter's own recording (fallback1) the Precision directive came first, so FP16 was rejected outright with no provisional stage. Since 10:50 the seeded variant is the first Speed hypothesis, which makes the order deterministic.

## Replay candidates

Durations are real and compressed with the current player (idle gaps over 4 s shortened, order unchanged). "Crashed" counts runs that produced no measurement.

| File | Real | Compressed | Hypotheses | Flip | Help asked / answered | Hybrid | Crashed | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| runs/record2.jsonl | 3:16 | 1:38 | 9 | yes | 2 / 2 | proposed, approved | 3 | newest code; the GPU dropped out of WSL at 167 s, so the hybrid and the last two runs failed at CUDA startup |
| runs/live4.jsonl | 10:22 | 2:06 | 10 | yes | 1 / 1 | no | 0 | older code (before commands apply at once), clean runs |
| runs/live1.jsonl | 9:15 | 3:06 | 15 | yes | 3 / 3 | no | 0 | older code, long, best approximate GELU number |
| runs/record1.jsonl | 7:43 | 1:41 | 8 | yes | 1 / 1 | no | 1 | shared memory compile failure (now prevented) |
| runs/live2.jsonl | 6:24 | 1:46 | 8 | yes | 0 / 0 | no | 0 | no help request |
| runs/fallback1.jsonl | 8:55 | 1:54 | 8 | no | 2 / 1 | no | 2 | presenter recording; Precision before FP16, two shared memory failures |

Recommendation: none of these shows every beat on the final code. Record one more session with the current build and the recording plan (Start recording, Speed directive, wait for the provisional card, Precision directive, hunch, answer the help request, approve the hybrid), with the GPU healthy, and use it as `runs/demo.jsonl`. Until then the UI falls back from `demo.jsonl` to `fallback1.jsonl`, `fallback2.jsonl` and on.

## Models per role

| Role | First choice | Fallbacks | Notes |
| --- | --- | --- | --- |
| Hypotheses and kernel edits | OmniRoute `auto/coding` (local gateway) | gemini-3.5-flash-lite, then NIM `openai/gpt-oss-20b` | OmniRoute's connected providers all refuse right now (OpenCode free tier, Felo, auggie), so each session's first call falls through to Gemini in about 6 s and the route is skipped afterwards |
| Post-mortem prose | gemini-3.8-flash, 15 s timeout | none, a factual template stands in | gemini-3.8-flash's quota was exhausted (HTTP 429) during the 10:52 recording, so its post-mortems used the template |
| Summaries | none | | every summary field is computed in code from the verdict, checks and measured spread |
| Verdicts | none | | always computed in code |

Gemini is used (both Gemini models above). NVIDIA NIM is the last fallback for hypotheses.

## Recording record2 (10:52, final code before the GPU dropped)

- Total 196 s, 10 model calls answered in 64.1 s total, 8.1 s spent in failed attempts (OmniRoute once, gemini-3.8-flash twice).
- Speed directive at 45.6 s, seeded FP16 provisional at 58.2 s, Precision directive applied the moment it arrived at 66.4 s with the flip and conflict in the same millisecond, post-mortem with the K sweep at 81.7 s.
- Help request hq1 1 ms after the inconclusive verdict, the answer applied the moment it arrived.
- Hybrid of prec and speed2 proposed at 143.5 s and approved, then the GPU dropped out ("Found no NVIDIA driver"), so H6 to H8 crashed at CUDA startup. Zero crashes came from configs: shared memory is checked before every run.
- At the cap the agent reported which promised iterations the cap cut.

## For the pitch and the writeup

- Quote 2.75x vs naive and 1.03x vs eager for the best FP32 config, with eager failing the 1e-3 gate for the reason above. The story is the process, not a large win over eager.
- Disclose: one person plays both lenses; the FP16 opener on the Speed lens is harness-seeded and says so in its note; error bounds are lens-specific and set in code; Papiers is read-only, findings are exported as files; replays compress idle gaps and say so.
- The WSL GPU passthrough dropped twice today (`nvidia-smi` fails with "Failed to initialize NVML"). `wsl --shutdown` from Windows brings it back. Check `nvidia-smi` before recording or demoing live.
