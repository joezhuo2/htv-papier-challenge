# CLAUDE.md

## Project

Multi-POV Research War Room, a solo hackathon build for the Papiers Human-Centred Autoresearch Challenge. An autonomous agent tunes a fused matmul + bias + GELU Triton kernel under two competing lenses (Speed and Precision) that one person switches between in a web UI. `war-room-plan.md` in the repo root is the source of truth for scope, architecture, schedule, and the demo, read the relevant sections before starting any task and follow it over your own assumptions.

Deadlines (America/Toronto): initial Devpost submission Saturday 9pm, final Sunday 9:30am. Tier 1 scope must work by Saturday 9pm, Tier 2 is polish for Sunday. Cut order when behind: Papiers deep to light, ledger table, conflict banner, hybrid merge. Never cut the failure path, the summary cards, or the help request flow.

## Standing rules

- Never run `git commit`, `git push`, or any command that publishes code unless I explicitly say so in that message. Staging nothing and leaving changes in the working tree is the default.
- Never use em dashes in code, comments, docs, or chat replies.
- Write paragraphs in docs, comments, and replies as continuous strings with no mid-paragraph line breaks.
- Do not comment obvious code. Keep comments concise and limited to complex edge cases.
- Use concise, idiomatic variable and function names, not verbose self-narrating ones.
- Write straightforward logic. No redundant defensive null checks and no excessive try/except boilerplate where normal error propagation is fine.
- If anything is ambiguous, or a design decision is not already settled in the plan or this file, ask me before starting the task. Ask as many design questions up front as you can, and do not assume what I mean.
- Work one phase at a time. At the end of each phase, stop with something runnable and a short report of what works, what does not, and what you decided. Do not start the next phase on your own.

## Environment

- WSL2 on Windows, RTX 5060 laptop GPU (Blackwell, compute capability 12.0, 8GB VRAM). Needs a PyTorch build for CUDA 12.8 or newer.
- Python managed with `uv`. Add dependencies with `uv add`, run code with `uv run`.
- Kernels are written in Triton. If Triton does not work on this machine, stop and tell me, the fallback domain is quantization and pruning (see the plan, Section 3).
- Every experiment runs in a subprocess with a hard 15 second timeout and a kill on hang.

## Stack and layout

FastAPI backend with WebSocket updates, React (Vite) front end, single package with subfolders:

```
src/
  harness/   benchmark, FP32 reference, naive Triton kernel, metrics JSON, subprocess runner
  agent/     hypothesis loop, verdict logic, branching, summaries, help requests, LLM client
  server/    FastAPI app, WebSocket, state, Papiers adapter
  ui/        Vite React app (DAG, metric chart, summary cards, help cards, lens tabs)
runs/        evidence logs keyed by hypothesis ID (gitignored)
tests/
```

## LLM

OpenAI compatible APIs, keys in a gitignored `.env` (`GEMINI_API_KEY`, `NVIDIA_API_KEY`), never hardcode or print them. The original plan was NIM with Nemotron and DeepSeek; the model benchmark and Saturday's changes replaced that (details in `docs/results.md` and `docs/architecture.md`). Current routing, all configurable via env:

- Hypotheses: a local OmniRoute gateway first (`OMNIROUTE_URL`, `OMNIROUTE_MODEL` default `auto/coding`, off with `OMNIROUTE=0`), then `gemini-3.5-flash-lite` (`GEMINI_FAST_MODEL`), then NIM `openai/gpt-oss-20b` (`FAST_MODEL`).
- Post-mortem prose: `gemini-3.8-flash` (`GEMINI_STRONG_MODEL`) with a 15 s timeout and a factual template as fallback, NIM `google/gemma-4-31b-it` (`STRONG_MODEL`) only without a Gemini key.
- Summaries and verdicts use no model, they are computed in code. Responses are cached in `runs/llm_cache/`.

## Invariants that must not be broken

- The agent states a falsifiable prediction with thresholds before each run. The verdict (`confirmed`, `rejected`, `inconclusive`) is computed in code from measured metrics, never by asking the LLM.
- The agent edits only a constrained surface of the kernel template: tile sizes, warps, stages, accumulation dtype, GELU formulation, fusion boundaries. No free-form kernel rewrites.
- The starting kernel is correct and fused but deliberately poorly configured (tiny tiles, one warp, one stage, FP32 accumulation, exact GELU, no `@triton.autotune`) and must pass `max_error < 1e-3` before any agent work begins.
- Inputs are scaled so outputs are O(1), and error is measured against an FP32 reference. If the baseline cannot meet the bound, tell me and propose relative error instead of loosening the threshold silently.
- Every metric shown in the UI, the video, or the writeup comes from a real logged run. Nothing is hardcoded for the demo.
- The failure path (an aggressive low-precision hypothesis that the agent itself rejects with a post-mortem) must be reproducible.
- Papiers is used for reading notes before hypotheses. Its CLI is read-only, so findings are exported as Markdown files for manual import (see `decisions.md`). Keep a local log as the source of truth.
