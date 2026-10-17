# Multi-POV Research War Room

An autonomous agent tunes a GPU kernel under two human lenses, Speed and Precision, and code, not the model, judges every result.

A solo build for Hack the Valley 11, entered in the Papiers Autoresearch Challenge. An autonomous agent tunes a fused matmul + bias + GELU Triton kernel under two competing lenses, Speed and Precision. One person switches between the two lens tabs in a web UI, and in a real team these would be two people. The agent states a falsifiable prediction before every run, the verdict is computed in code from measured metrics (never by the LLM), it rejects its own failed hypotheses with a post-mortem, and it asks the relevant lens owner for help while it keeps exploring other branches.

Demo video: [VIDEO_URL](https://youtu.be/x-hu9OjLhmY). Devpost: [DEVPOST_URL](https://devpost.com/software/research-war-room).

`war-room-plan.md` is the full spec, `decisions.md` records decisions that override it, `docs/status.md` tracks progress, and `docs/results.md` has the measured results.

## Try the replay

No GPU and no `.env` needed, only Node 22 or newer. `npm ci` needs the network once to download the UI dependencies; after that the build and the replay run fully offline. From the repo root:

```bash
cd src/ui && npm ci && npm run build:judging && npx vite preview
```

Open `http://localhost:4173/`. The judging build plays the frozen recording `demo/replay.jsonl` (tag `demo-freeze`), labelled recorded run, with idle gaps over 4 s compressed and disclosed. The 60-second tour, in player time:

- 0:00 to 0:18: the eager and naive baselines, then H0 (128x128x64 tiles, 4 warps, 4 stages) confirmed at 2.52x over naive.
- 0:22: a Speed directive with a 2e-2 tolerance opens the Speed branch, and at 0:28 the FP16 accumulation run H1 is provisionally confirmed at 1.54x over its parent.
- 0:35: a Precision directive sets 1e-3 on both batches. The agent re-judges the logged runs without re-running them, flips H1 to rejected, raises the conflict banner and opens the Precision branch.
- 0:43 to 0:47: a help request to the Speed lens (it keeps exploring while it waits) and the H1 post-mortem with the error versus K sweep.
- 0:49 to 1:27: the typed GELU hunch runs as H4, and the remaining runs land inside the noise band or are rejected as slower.
- 1:30: the final report: best run, what was ruled out, what to try next, and what a human has to decide.

The findings exported from this run are in [docs/examples/exports](docs/examples/exports), and its run table is in [docs/results.md](docs/results.md).

## Honest numbers

- The best kernel is about 2.6x faster than the naive start (2.59x in the frozen run, 2.5x to 2.8x across recordings).
- It is roughly on par with PyTorch eager, 0.93x to 1.05x depending on the timing window (0.97x in the frozen run).
- FP16 accumulation is about 1.5x faster still, but its error is 1.3e-2 on the standard batch and 7.1e-2 on the adversarial batch, so the Precision lens rejects it.
- PyTorch eager itself fails the 1e-3 gate, at 1.5e-3 standard and 2.5e-3 adversarial. The rounding measurements in docs/results.md point at its FP16 intermediates, but the explanation is still open; the next check is whether `tl.dot` runs in TF32 or IEEE mode.
- Five of the eight results in the frozen run are inconclusive inside a noise band of roughly 5%.

## Status

Working today: the benchmark harness, the agent loop with code computed verdicts, lens directives with lens-specific error bounds, branching and conflict detection, a hybrid step that proposes a merge when two lenses both find gains and otherwise declines with a logged reason, post-mortems with an error versus K sweep, help requests with a deadline and default, read-only Papiers context, findings export as Markdown files for manual import (the Papiers CLI is read-only today, so write-back is pending, not working), the FastAPI WebSocket server, and the React war room UI (mock, recorded replay, and live). Every number below comes from a logged run on the build machine. Final numbers come from the recorded demo session, so treat anything not in a session log as provisional.

## How the agent decides

Before each run the agent states a prediction (minimum speedup against naive or the best so far). The error bound is not the agent's choice: the lens sets it in code. With no directive the strict 1e-3 applies to the standard batch. A Speed directive gives Speed branches a looser standard-batch tolerance (2e-2 by default, parsed from the directive text, never above 5e-2). A Precision directive sets 1e-3 on the standard and adversarial batches for every branch. Adversarial error is measured on every run, so a Precision directive re-judges logged runs without re-running them. Verdicts are `confirmed`, `rejected`, or `inconclusive` (inside the noise band), and a result that only passes under the Speed tolerance is marked provisional.

FP16 accumulation is the designed failure: it is faster on this GPU, its standard error (1.3e-2, identical on every run thanks to fixed-seed inputs) passes the Speed tolerance, and the Precision directive flips it to rejected with a post-mortem and an error versus K sweep. The first hypothesis on the Speed lens is this FP16 variant of the best tiles so far, seeded by the harness and disclosed in its note, so the story is the same in every session.

## Hardware and measured baseline

RTX 5060 Laptop GPU (Blackwell, compute capability 12.0, 8 GB), WSL2 on Windows, PyTorch 2.11 for CUDA 12.8, Triton 3.6. Standard batch M = N = K = 2048 in FP16.

Speedups are same-window: the candidate is timed interleaved with naive, eager and its parent in one child process, over 10 separate processes per row (`harness.stability`, saved to `src/harness/headroom.json`).

| Config | vs naive | vs eager | max_error | adv_error | Passes 1e-3 |
| --- | --- | --- | --- | --- | --- |
| Naive Triton start (16x16x16 tiles, 1 warp, 1 stage) | 1.00x | about 0.37x | 4.9e-4 | 5.2e-4 | yes |
| PyTorch eager | about 2.67x | 1.00x | 1.5e-3 | 2.5e-3 | no |
| Best sweep config (128x128x32, 4 warps, 3 stages) | 2.75x (2.70 to 2.78) | 1.03x (1.02 to 1.04) | 4.9e-4 | 5.2e-4 | yes |
| Same config, FP16 accumulation | 4.03x (3.99 to 4.09) | 1.52x (1.49 to 1.54) | 1.3e-2 | 7.1e-2 | no |

This table is the offline sweep, not an agent session. Its best FP32 config is about 2.75x faster than the naive start and about 3% faster than eager in that timing window, while the agent's recordings land at 2.5x to 2.8x and 0.93x to 1.05x (see Honest numbers), and eager itself fails the 1e-3 gate (the explanation is still open, see above). FP16 accumulation is about 1.48x faster still but fails the error bound by 13x on the standard batch and 71x on the adversarial batch, which is the failure the agent has to discover and reject by itself. Across processes the speedups vary by under 1% (coefficient of variation), and within one timing window the per-round spread is about 4%, which the verdicts use as the noise band.

## Model choice

The first model benchmark ran through NVIDIA NIM's hosted OpenAI compatible API. Each model got the same hypothesis prompt 10 times (current params, a Speed directive, and the naive kernel's measured result), and every proposed edit was run through the harness.

| Model | Median latency | Valid JSON | Edit compiles | Edit passes 1e-3 |
| --- | --- | --- | --- | --- |
| openai/gpt-oss-20b | 13.0 s | 100% | 100% | 50% |
| google/gemma-4-31b-it | 12.9 s | 100% | 100% | 100% |
| nvidia/nemotron-3.5-lightning-30b-a3b | 165 s | 100% | 60% | 20% |

DeepSeek v4.1 flash and GLM 5.3 flash were dropped because single calls took several minutes. That benchmark set the NIM fallbacks (`openai/gpt-oss-20b` for hypotheses, `google/gemma-4-31b-it` for post-mortems). The current order is a local OmniRoute gateway, then `gemini-3.5-flash-lite`, then NIM for hypotheses, and `gemini-3.8-flash` for post-mortem prose; summaries and verdicts are computed in code. See docs/results.md for the models per role. Hosted latency varied a lot during the hackathon, so responses are cached in `runs/llm_cache/`.

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

Run a headless research session. Each `--directive N:lens:text` is sent once N hypotheses have verdicts, and the event log lands in `runs/sessions/<id>/events.jsonl` with exported findings in `exports/` next to it:

```bash
uv run python -m agent.run --directive "1:speed:Push throughput harder" --directive "3:precision:Max error under 1e-3 on both batches"
```

Run the war room live. Start the agent server first, then the UI, and open `http://localhost:5173/?mode=live`. The page shows a pre-flight check (GPU, Gemini, NIM, Papiers, each checked against the real endpoint) and the session starts when you press Start; set `AUTOSTART=1` to skip that. Start session is the table demo path and stays capped at one hypothesis (`MAX_HYPS`, empty for no cap). Start recording is the separate recording mode: it runs H0, then waits for a lens directive (after `DIRECTIVE_WAIT_S`, default 90 s, with none it continues on main as a flagged default, only while budget remains and a lens has yet to speak), and runs until Stop or `RECORD_HYPS` hypotheses (default 8); once the budget is spent, or both lenses have spoken and nothing runs or waits, it ends after `END_WAIT_S` (default 15 s) unless a late hunch arrives, accepting directives and help answers at any point, with the log saved as a replay under `runs/sessions/<id>/events.jsonl` (the last agent log line gives the replay URL). Stop takes effect at once, even during a model call, and New session stops the current session and starts a fresh one through `POST /restart`. `MAIN_ITERS`, `BRANCH_ITERS`, `IDLE_S` and `HELP_DEADLINE_S` are configurable. Model calls go to a local OmniRoute gateway first (`OMNIROUTE_URL`, default `http://localhost:20128/v1`, model `OMNIROUTE_MODEL`, default `auto/coding`, disable with `OMNIROUTE=0`), then Gemini (`GEMINI_API_KEY`), then NIM, and every fallback is logged. Start OmniRoute before the server; the pre-flight check shows whether it answers and lists the model.

```bash
uv run uvicorn server.app:app --port 8000
```

```bash
cd src/ui && npm run dev
```

The UI also plays the mock script (`?mode=mock`, the default in development, always tagged mock) and recorded sessions (`?mode=replay&file=/runs/sessions/<id>/events.jsonl`, labelled recorded run, with idle gaps over 4 s compressed and disclosed). For judging, build without mock mode; it opens the first recorded run that loads, the committed `demo/replay.jsonl` then the local `runs/fallback1.jsonl`, `runs/fallback2.jsonl` and on. The 19:41 UTC recording is committed too and plays with `?mode=replay&file=/demo/fallback-1541.jsonl`. Freeze any new demo recording with `scripts/freeze.sh <session-folder>` (`--as <file>` for a fallback), which checks the log is complete and the copies hold no secrets:

```bash
cd src/ui && npm run build:judging && npx vite preview
```

Measure timing stability (each candidate is timed interleaved with naive, eager and its parent in one process):

```bash
uv run python -m harness.stability 10
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
cd src/ui && npm test
```

Lint and type checks:

```bash
uv run ruff check src tests
```

```bash
cd src/ui && npm run lint && npm run typecheck
```

## Layout

```
src/
  contract/  shared event and state types (types.ts) and the mock demo script
  harness/   Triton kernel template, inputs and FP32 reference, experiment child process, runner, bench, noise band
  agent/     orchestrator (loop, branches, help requests), verdicts, hypothesis prompts, Papiers adapter, export, NIM client
  server/    FastAPI app: /ws streams the session's events and accepts commands
  ui/        Vite React war room, with its vitest suite in src/ui/test
runs/        evidence logs, sessions, and exports (gitignored)
tests/       pytest suites and fixtures, plus the contract smoke test in tests/ui
```

Each experiment runs in its own subprocess. The child imports torch and initializes CUDA (about 9 s here), prints `ready`, and then gets a hard 15 second timeout, after which its whole process group is killed.

## Credits

The kernel's tiling and grouped program ordering follow the official Triton matrix multiplication tutorial (triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html). Built with Triton, PyTorch, NVIDIA NIM, FastAPI, React, and Papiers.
