# Results

Every number here comes from a logged run on the build machine (RTX 5060 Laptop GPU, WSL2, PyTorch 2.11 + CUDA 12.8, Triton 3.6). Speedups are same-window: the candidate is timed interleaved with naive, eager and its parent in one child process, and the +/- is that window's measured spread. Logs live in `runs/` (gitignored), so the file names below are local evidence, not repo files.

## Frozen demo run

`demo/replay.jsonl` (tag `demo-freeze`) is the 20:00 UTC recording, `runs/sessions/20261017-155752`: 106 events, 2:53 real, about 1:30 in the player with idle gaps compressed. It is the only recording committed to the repo, so its numbers can be checked from a fresh clone. The table is generated from the log by `uv run python scripts/results_table.py --write`, not typed. Verdicts are the final ones after the Precision directive re-judged the logged runs, and "vs parent" is the same-window speedup over the run's parent (naive for r1). The best run, in bold, is the lowest latency among runs inside every active bound, whatever its verdict, so the rejected FP16 run r2 is faster but not eligible.

<!-- results-table -->
| Run | Hypothesis | Branch | Config | Latency (ms) | vs parent | vs naive | Standard error | Adversarial error | Registers | Spills | Verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| r1 | H0 | main | 128x128x64, 4 warps, 4 stages, FP32 acc, exact GELU | 0.783 | 2.52x (naive) | 2.52x | 4.9e-4 | 5.2e-4 | 255 | 24 | confirmed |
| r2 | H1 | speed | 128x128x64, 4 warps, 4 stages, FP16 acc, exact GELU | 0.514 | 1.54x | 3.73x | 1.3e-2 | 7.1e-2 | 255 | 10 | rejected |
| r3 | H2 | speed | 128x128x64, 4 warps, 3 stages, FP32 acc, exact GELU | 0.735 | 1.02x | 2.47x | 4.9e-4 | 5.2e-4 | 255 | 24 | inconclusive |
| r4 | H3 | prec | 128x128x64, 8 warps, 4 stages, FP32 acc, exact GELU | 0.729 | 1.05x | 2.58x | 4.9e-4 | 5.2e-4 | 160 | 0 | inconclusive |
| **r5** | H4 | speed2 | 128x128x64, 4 warps, 4 stages, FP32 acc, approx GELU | 0.719 | 1.07x | 2.59x | 6.9e-4 | 7.2e-4 | 255 | 0 | inconclusive |
| r6 | H5 | speed2 | 128x128x64, 8 warps, 4 stages, FP32 acc, approx GELU | 0.730 | 1.10x | 2.63x | 6.9e-4 | 7.2e-4 | 171 | 0 | inconclusive |
| r7 | H6 | prec | 128x128x64, 8 warps, 3 stages, FP32 acc, exact GELU | 0.731 | 1.07x | 2.57x | 4.9e-4 | 5.2e-4 | 160 | 0 | inconclusive |
| r8 | H7 | prec | 128x128x128, 8 warps, 2 stages, FP32 acc, exact GELU | 0.825 | 0.95x | 2.29x | 4.9e-4 | 5.2e-4 | 166 | 0 | rejected |
<!-- /results-table -->

Measurement noise: each speedup comes with that window's per-round spread, and here the band was about 4% to 8% of the parent ratio (roughly 5%). Five of the eight results (H2 to H6) are inconclusive, their gains of 1.02x to 1.10x over their parents sit inside that band, so the agent could not call them either way. Only the big moves are clear: H0 over naive (2.52x, confirmed), FP16 accumulation (1.54x faster, rejected on error) and H7's 128-wide K tile (0.95x, rejected as slower). Run to run variation is of the same order: within this recording the vs-naive ratios of nearly identical FP32 configs span 2.47x to 2.63x, and r6 reports a higher vs-naive ratio than the best run r5 while being slower in absolute latency, because each ratio is measured in its own window. Across recordings the best FP32 result ranged from 2.5x to 2.8x vs naive and 0.93x to 1.05x vs PyTorch eager, so the honest summary is about 2.6x over the naive start and roughly on par with eager.

Next steps: rerun the inconclusive configs (H2 to H6, starting with the best run H4) with more timing rounds to shrink the band, and test whether `tl.dot` runs in TF32 or IEEE mode, which is still the open question behind the eager comparison.

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

So the standard 1.5e-3 is two extra roundings before the GELU (the matmul output and the bias add each cost about one more half ulp), and the third rounding at the GELU output adds nothing measurable to the maximum. The adversarial 2.5e-3 is the large_k case (the adversarial error is the worse of large_k and large_mag): with K = 16384, cuBLAS reduces partial sums in reduced precision, PyTorch's default for FP16 GEMMs, and turning that off brings large_k back to 1.5e-3. So eager is a speed reference, not a correctness reference, and the README and UI say so. These measurements point at the FP16 intermediates, but the explanation is still treated as open until the fused kernel's `tl.dot` precision mode (TF32 or IEEE) has been checked, since the comparison assumes both sides accumulate in IEEE FP32.

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
| 17:07, `war-room-2026-10-17-17-07-33.jsonl` (suggested slot: fallback1) | 2:32 | 1:34 | 8 | yes | 1 / 1 | not proposed | 0 | recorded on the code before the 16:30 fixes (not pulled); flip at 65.2 s; annotated highlights cited; best confirmed H0 at 2.72x vs naive and 1.02x vs eager, nothing after it beat it; the GELU hunch was not retested |
| runs/fallback1.jsonl (16:00, `war-room-2026-10-17-16-00-14.jsonl`) | 3:11 | 1:45 | 9 | yes | 3 / 3 | proposed, approved | 0 | hand played on the code before the 16:30 fixes; flip at 90.9 s; H6 (128x256 on 4 warps) spilled 1442 words at 0.10x; the GELU hunch arrived after the cap and never ran; best confirmed H3 at 2.65x vs naive and 0.97x vs eager |
| runs/fallback2.jsonl (was fallback1) | 8:55 | 1:54 | 8 | no | 2 / 1 | no | 2 | presenter recording; Precision before FP16, two shared memory failures |
| runs/fallback3.jsonl (was fallback2) | | | | | | | | not described in these notes |

Recommendation: none of these shows every beat on the final code. Record one more session with the current build (after the 16:30 fixes) and the recording plan (Start recording, Speed directive, wait for the provisional card, Precision directive, the tile hunch "Try 128x128x32 tiles with 3 stages", answer the help requests, approve the hybrid), with the GPU healthy, and use it as `runs/demo.jsonl`. Until then the UI falls back from `demo.jsonl` to `fallback1.jsonl` (the 16:00 recording), `fallback2.jsonl` and on.

## Recording 17:07 (code before the 16:30 fixes)

- 152 s real, 94 s compressed, 8 hypotheses, 0 crashed runs, no spills (the most was 173 registers per thread). It passes the UI's log validation and folds to a finished session. The log shows the old code: the Speed bound line after the Precision directive, "not both under", `acc=fp16`, no `Summary.hypothesis`, no re-judge of H0, and the Precision branch opening 13.6 s after its directive.
- H0 (64x64x32, 4 warps, 3 stages) confirmed at 2.72x +/- 0.07x vs naive and 1.02x vs eager, the best result of any recording so far, and nothing after it beat it: H2, H3, H5 and H6 were speed misses (0.95x to 1.00x), H4 (approximate GELU) and H7 (64x128x32) inconclusive at 1.03x.
- Seeded FP16 (H1) provisional at 58.2 s, 1.41x +/- 0.02x over H0, standard 1.3e-2 and adversarial 7.1e-2; the Precision directive at 65.2 s flipped it within 2 ms, post-mortem with the K sweep at 78.9 s using the factual template (the prose model did not answer).
- Papiers: the annotations show. hl_da32 and hl_dcdb carry the user's notes, the H1 post-mortem cites hl_dcdb (the VRR breakdown note) instead of hl_4784, and "tile size shared memory" now returns the Triton tutorial conversation. hl_ad37 (GELU) and hl_4d5d have no annotation yet.
- No hybrid: each lens's fastest correct run was an approximate GELU variant (H5 on speed2, H4 on prec), the union is H5 itself and the other tile shape gives H4 itself, so every candidate was skipped without a log line.
- The GELU hunch (d3) was not retested because H4 on prec had already run that exact config; the message said "already has gelu=approx or that exact config was run", which does not say which.
- One help request (hq1 on prec), answered "keep tuning" after 26.7 s; prec ran both promised iterations (H6, H7). The cap of 8 ended the session.

## Recording fallback1 (16:00, hand played, code before the 16:30 fixes)

- 191 s real, 105 s compressed, 9 hypotheses (8 plus the hybrid), 0 crashed runs, help requests hq1 to hq3 all answered "keep tuning".
- Seeded FP16 (H1) provisional at 78.0 s, 1.51x +/- 0.03x over its FP32 parent, standard 1.3e-2 and adversarial 7.1e-2; the Precision directive at 90.9 s flipped it to rejected with the conflict in the same millisecond, post-mortem with the K sweep at 105.2 s.
- The model proposed approximate GELU by itself three times (H2, H5, and the H7 hybrid), each inconclusive inside the noise band (1.09x, 1.06x, 1.01x).
- H6, 128x256x64 tiles on 4 warps (256 FP32 accumulator elements per thread), compiled to 40 registers and 1442 spilled words per thread and ran at 7.373 ms, 0.10x of its parent: the first real register post-mortem and the first real off-scale chart point.
- Best confirmed result: H3 (128x256x64, 8 warps, 3 stages), 2.65x vs naive and 0.97x vs eager. The agent never tried block_k 32, the measured headroom config (2.75x, 1.03x).
- Problems it exposed, all fixed at 16:30 (see docs/status.md): the late hunch opened an empty branch and was dropped by the cap, the Speed bound message ignored the active Precision bound, the cap was blamed for an iteration that the two-miss rule cut, the hybrid text named changes its candidate did not have, the hybrid summary used a different baseline from its claim, register overflow was only checked for hybrids, re-judging skipped every run that was not provisional, and the Precision branch opened 14 s late behind the post-mortem.

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
- Disclose: one person plays both lenses; the FP16 opener on the Speed lens is harness-seeded and says so in its note; error bounds are lens-specific and set in code; findings are exported as Markdown files, and Papiers write-back happens only for verdicts a human approves after a live run; replays compress idle gaps and say so.
- The WSL GPU passthrough dropped twice today (`nvidia-smi` fails with "Failed to initialize NVML"). `wsl --shutdown` from Windows brings it back. Check `nvidia-smi` before recording or demoing live.
