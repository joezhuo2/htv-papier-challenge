# Findings

Measured results and what they mean, from logged runs on the build machine (RTX 5060 Laptop GPU, WSL2, PyTorch 2.11 + CUDA 12.8, Triton 3.6). Numbers from before the interleaved timing change (04:10 Saturday) are marked as cross-process and are not reliable for speedups. Final demo numbers must come from the recorded demo session.

## Baselines and the kernel (harness bench, cross-process)

| Run | Latency | Standard error | Passes 1e-3 |
| --- | --- | --- | --- |
| PyTorch eager | about 0.67 ms | 1.5e-3 | no |
| Naive Triton (16x16x16, 1 warp, 1 stage) | about 2.0 ms | 4.9e-4 | yes |
| Best sweep config (128x128x32, 4 warps, 3 stages) | about 0.64 ms | 4.9e-4 | yes |
| Same config, FP16 accumulation | about 0.45 ms | 1.3e-2 | no |
| Same config, approximate GELU | about 0.64 ms | 6.9e-4 | yes |

Eager itself fails the strict bound on this metric. Approximate GELU gave no speedup on this GPU, so the mock's H4 story (approximate GELU as the big win) does not hold on real hardware.

## Same-window timing (harness.stability, 10 runs each, runs/stability.log)

| Speedup | Median | Range | Across-run CV | Within-run half-range |
| --- | --- | --- | --- | --- |
| FP32 best tiles vs naive | 2.75x | 2.70 to 2.78 | 0.7% | 3.6% |
| FP32 best tiles vs eager | 1.03x | 1.02 to 1.04 | 0.7% | 5.1% |
| FP16 vs naive | 4.03x | 3.99 to 4.09 | 0.8% | 3.7% |
| FP16 vs eager | 1.52x | 1.49 to 1.54 | 0.9% | 4.6% |
| FP16 vs FP32 tiles | 1.48x | 1.47 to 1.51 | 0.7% | 3.9% |

Before interleaving, the FP16 vs FP32 comparison measured 1.28x, 1.43x, 1.88x and 2.36x in different runs, and one overnight H0 measured 5.24x over naive against 2.7x to 3.9x earlier. The per-round spread (about 4%) is used as the verdict noise, which is conservative: the median itself is stable to under 1%.

## The designed failure (FP16 accumulation)

- Standard error 1.318e-2 and adversarial error 7.083e-2 on every run (fixed seed), so it passes the Speed tolerance of 2e-2 with a 34% margin and fails 1e-3 on both batches.
- Error versus K on the large-magnitude inputs: about 1.9e-2 at K=256, 2.4e-2 at 512, 3.2e-2 at 1024, 5.8e-2 at 2048, 8.1e-2 at 4096, above the strict bound at every K.
- GPU test (tests/test_agent_gpu.py): three runs in a row are provisional under 2e-2, then the Precision directive flips the verdict to rejected from the logged metrics with no run_started between the directive and the flip, followed by a post-mortem with the 5 point sweep.

## Model benchmark (NIM, before Gemini)

| Model | Median latency | Valid JSON | Edit compiles | Edit passes 1e-3 |
| --- | --- | --- | --- | --- |
| openai/gpt-oss-20b | 13 s | 100% | 100% | 50% |
| google/gemma-4-31b-it | 13 s | 100% | 100% | 100% |
| nvidia/nemotron-3.5-lightning-30b-a3b | 165 s | 100% | 60% | 20% |

gpt-oss reaches for FP16 accumulation by itself, gemma repeats one safe edit and overpredicts, Nemotron is too slow. DeepSeek v4.1 flash and GLM 5.3 flash took minutes per call. Gemini (added at 04:15) answered test calls in 1.8 s (gemini-3.5-flash-lite) and 2.7 s (gemini-3.8-flash) and is now the default, with NIM as fallback.

## Real sessions so far (behavior only, none is the demo replay)

| Session | Code | Agent proposed FP16 itself | Seeded FP16 on Speed | Flip | Notes |
| --- | --- | --- | --- | --- | --- |
| rep1 to rep3 | before lens bounds | in 2 of 3, on main from the Papiers Triton conversation | rep1 only | n/a | rep1 exposed 15x and 0.15x predictions and pivots on every speed miss, both fixed |
| night1 | lens bounds, cross-process timing, NIM | no | rejected on the speed check | no | miss caused by timing noise |
| night2 | same | yes, on main before any directive (rejected at strict default) | inconclusive | no | help request fired |
| night3 | interleaved timing, Gemini | no | provisional, then rejected by Precision | yes | |
| candidate1 (05:00) | final code | no | 1.54x +/- 0.12x over FP32 parent, provisional, then rejected 3.6 s later | yes | best confirmed result only r1 (2.78x vs naive, 0.90x vs eager), two help requests timed out to defaults, two shared-memory compile failures |

| fallback1 (16:00) | before the 16:30 fixes, hand played | no | 1.51x +/- 0.03x over FP32 parent, provisional, then rejected at the Precision directive | yes | model tried approximate GELU three times (all inconclusive), H6 spilled at 0.10x, best confirmed 2.65x vs naive and 0.97x vs eager |

Across all sessions: predictions stayed in range after the fix, every papiers_context carries the real query, note id and excerpt, and the Precision directive's query returns no Papiers notes (logged as no matching notes). Screenshots of candidate1 are in docs/screenshots.

## Observations worth knowing

- Shared memory is limited to 101,376 bytes per block on this GPU. Tiles like 128x256x64 with 4 stages fail to compile, the agent treats that as an ordinary miss.
- The best FP32 configuration is only about 3% faster than PyTorch eager in the same window, and eager does not pass the strict bound. The honest headline is the agent's process (prediction, code verdict, self-rejection, human steering), not a large speedup over eager.
- The agent rarely proposes FP16 on a Speed branch by itself, because the Papiers context and the ruled-out list steer it away after a strict rejection on main. The seeded variant makes the failure path reproducible and is disclosed in the hypothesis note.
