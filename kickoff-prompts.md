# Kickoff prompts for Claude Code

Setup before pasting anything:

1. Open a WSL2 terminal and make the project folder, then copy `CLAUDE.md` and `war-room-plan.md` into its root.
2. Create a `.env` with `NVIDIA_API_KEY=...` and make sure `.env` is in `.gitignore` (the first prompt will also check this).
3. Start Claude Code from the project folder.

## Prompt 1: checkpoint and harness (Phase 0 and the start of Phase 1)

```
Read CLAUDE.md and war-room-plan.md, focusing on Sections 3, 4, and 5. We are doing the hour 0 checkpoint and the benchmark harness only. Before writing any code, ask me every question you have about ambiguity or design decisions. Do not commit or push anything.

Part A, checkpoint (report each result as you go, and stop and tell me if anything fails):
1. Run nvidia-smi and report the GPU.
2. Set up the project with uv, install a PyTorch build for CUDA 12.8 or newer, print torch.cuda.get_device_capability(), and time a 4096 x 4096 FP16 matmul.
3. Compile and run a trivial Triton vector add, then the official Triton matmul tutorial kernel. If Triton fails on this GPU, stop here and tell me, do not continue to Part B.
4. Read papiers.ai/docs and report how to install the CLI and the exact commands for retrieving reading context and writing findings back. Do not wire anything in yet.
5. Call each NIM model once (nvidia/nemotron-3.5-lightning-30b-a3b and deepseek-ai/deepseek-v4.1-flash) with NVIDIA_API_KEY from .env, verify the base URL and model IDs, and report latency. Make sure .env is gitignored.

Part B, harness (only after Part A passes):
1. Scaffold the layout from CLAUDE.md.
2. In src/harness, build: a PyTorch eager reference (gelu(x @ w + b)), an FP32 reference, input scaling so outputs are O(1), a standard test batch plus adversarial inputs (large K, large magnitudes), and a naive Triton fused matmul + bias + GELU kernel as specified in Section 4 (tiny tiles, one warp, one stage, FP32 accumulation, exact GELU, no autotune).
3. Benchmark with CUDA events, warmup, and median of N runs. Output one JSON object per run with latency_ms, tflops, max_error, vram_mb, compiled_ok, correct_ok.
4. Run each experiment through a subprocess runner with a 15 second hard timeout and kill on hang.
5. Check that the naive kernel passes max_error < 1e-3. If it does not, tell me and propose relative error instead of loosening the threshold.
6. Run a quick tile size, warps, and stages sweep to show how much headroom the naive config leaves. Report speedups against both eager and the naive kernel. If the headroom is only a few percent, tell me and propose a worse starting config.

Acceptance: `uv run python -m harness.bench` prints JSON for eager, the naive kernel, and the sweep. Stop after that and give me a short report.
```

## Prompt 2: model benchmark (after Prompt 1 is done and reviewed)

```
Read war-room-plan.md Section 5. Write a small script in src/agent that runs the same prompt against both NIM models about 10 times each: given the current kernel parameters, a directive, and the last result, return a JSON hypothesis with a falsifiable prediction, thresholds, and a constrained parameter edit. Record median latency, valid JSON rate, and the rate of edits that compile and pass the correctness check in the harness. Print two example explanations from each model. Use generous timeouts for the deepseek model, cache responses to disk, and respect rate limits. Report a recommendation for the model split. Do not commit or push anything, and stop when the report is done.
```

## After that

Give Claude Code one phase at a time using the plan's Section 14 schedule: the minimal agent loop with programmatic verdicts, then lens tabs and forking with the failure path, then summaries and help requests, then Papiers, then the UI, then dry runs. For each, paste the relevant plan sections, ask it to ask its questions first, and end with a runnable acceptance check.
