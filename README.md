# Multi-POV Research War Room

A solo build for Hack the Valley 11, entered in the Papiers Autoresearch Challenge. An autonomous agent tunes a fused matmul + bias + GELU Triton kernel under two competing lenses, Speed and Precision. One person switches between the two lens tabs in a web UI, and in a real team these would be two people. The agent states a falsifiable prediction before every run, the verdict is computed in code from measured metrics (never by the LLM), it rejects its own failed hypotheses with a post-mortem, and it asks the relevant lens owner for help while it keeps exploring other branches.

`war-room-plan.md` is the full spec, `decisions.md` records decisions that override it, and `status.md` tracks progress.

## Status

Working today: the benchmark harness, the constrained kernel edit surface, the subprocess runner, the test suites, and the NIM model benchmark. The agent loop, orchestrator, server, and UI are in progress. Every number below comes from a logged run on the build machine.

## Hardware and measured baseline

RTX 5060 Laptop GPU (Blackwell, compute capability 12.0, 8 GB), WSL2 on Windows, PyTorch 2.11 for CUDA 12.8, Triton 3.6. Standard batch M = N = K = 2048 in FP16.

| Run | Latency | max_error | Passes 1e-3 |
| --- | --- | --- | --- |
| PyTorch eager | 0.69 ms | 1.5e-3 | no |
| Naive Triton start (16x16x16 tiles, 1 warp, 1 stage) | 2.0 ms | 4.9e-4 | yes |
| Best sweep config (128x128x32, 4 warps, 3 stages) | 0.64 ms | 4.9e-4 | yes |
| Same config, FP16 accumulation | 0.45 ms | 1.3e-2 | no |

The best FP32 config is about 3.1x faster than the naive start and about 1.07x faster than eager. FP16 accumulation is faster still but fails the error bound, which is the failure the agent has to discover and reject by itself. Run to run latency noise across separate processes is about 0.6% (coefficient of variation), and drift over 3 minutes of sustained load is under 0.2%.

## Model choice

All models run through NVIDIA NIM's hosted OpenAI compatible API. Each model got the same hypothesis prompt 10 times (current params, a Speed directive, and the naive kernel's measured result), and every proposed edit was run through the harness.

| Model | Median latency | Valid JSON | Edit compiles | Edit passes 1e-3 |
| --- | --- | --- | --- | --- |
| openai/gpt-oss-20b | 13.0 s | 100% | 100% | 50% |
| google/gemma-4-31b-it | 12.9 s | 100% | 100% | 100% |
| nvidia/nemotron-3.5-lightning-30b-a3b | 165 s | 100% | 60% | 20% |

DeepSeek v4.1 flash and GLM 5.3 flash were dropped because single calls took several minutes. The defaults are `openai/gpt-oss-20b` for hypotheses and edits and `google/gemma-4-31b-it` for post-mortems and summaries. Hosted latency varied a lot during the hackathon, so responses are cached in `runs/llm_cache/`.

## How correctness is measured

Inputs are scaled so pre-activations have unit variance and outputs are O(1). Outputs are compared against an FP32 reference computed from the same FP16 inputs. The error is `max |y - ref| / max(|ref|, 1)`, which is absolute for small outputs and relative for large ones, with a bound of 1e-3. A plain absolute bound was unreachable even for a correct kernel, because FP16 output rounding alone is about 2e-3 at |y| around 4. The standard batch is the default gate. Two adversarial cases (512 x 512 x 16384 for large K, and 8x larger pre-activations) are reported as `adv_error` and gated by the Precision lens.

## Setup

Requires an NVIDIA GPU with a recent driver, [uv](https://docs.astral.sh/uv/), and Node 22 or newer.

```bash
uv sync
```

```bash
cd src/ui && npm install
```

Create a gitignored `.env` in the repo root:

```
NVIDIA_API_KEY=...
PAPIERS_TOKEN=...
```

`FAST_MODEL` and `STRONG_MODEL` optionally override the NIM model IDs. For Papiers, install the read-only CLI with `npm install -g papiers` and check it with `papiers doctor`.

## Running

Benchmark eager, the naive kernel, and a tile, warps, and stages sweep (one JSON line per run, evidence in `runs/`):

```bash
uv run python -m harness.bench
```

Compare NIM models on the hypothesis prompt:

```bash
uv run python -m agent.model_bench
```

Tests. The default run is the fast suite, and GPU, Papiers, and long running tests are opt in:

```bash
uv run pytest
```

```bash
uv run pytest -m gpu
```

```bash
uv run pytest -m slow
```

```bash
cd src/ui && npx vitest run
```

## Layout

```
src/
  contract/  shared event and state types (types.ts) and the mock demo script
  harness/   Triton kernel template, inputs and FP32 reference, experiment child process, runner, bench, noise band
  agent/     NIM client and model benchmark (agent loop in progress)
  server/    FastAPI app (in progress)
  ui/        Vite React app (in progress)
runs/        evidence logs keyed by run id (gitignored)
tests/       pytest suites, fixtures, and vitest tests in tests/ui
```

Each experiment runs in its own subprocess. The child imports torch and initializes CUDA (about 9 s here), prints `ready`, and then gets a hard 15 second timeout, after which its whole process group is killed.

## Credits

The kernel's tiling and grouped program ordering follow the official Triton matrix multiplication tutorial (triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Built with Triton, PyTorch, NVIDIA NIM, FastAPI, React, and Papiers.
