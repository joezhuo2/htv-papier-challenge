# Decisions

Project: Multi-POV Research War Room. Solo build by Joe for Hack the Valley 11, entered in the Papiers Autoresearch Challenge. An autonomous agent tunes a fused matmul + bias + GELU Triton kernel under two competing lenses (Speed and Precision) that one person switches between in a web UI. `war-room-plan.md` has the full spec and `CLAUDE.md` has the standing rules. Where this file disagrees with the plan, this file wins (see Overrides).

## Locked

- **Domain:** GPU kernel tuning, fused matmul + bias + GELU in Triton, baseline is PyTorch eager (`gelu(x @ w + b)`).
- **Hardware and env:** RTX 5060 laptop (8GB VRAM, compute capability 12.0), WSL2 on Windows, Python with `uv`, PyTorch for CUDA 12.8 or newer. Fallback domain if Triton fails: quantization and pruning.
- **Starting kernel:** correct and fused but deliberately weak (tiny tiles, 1 warp, 1 stage, FP32 accumulation, exact GELU, no autotune). Must pass `max_error < 1e-3` before any agent work. Inputs scaled so outputs are O(1), error measured against an FP32 reference.
- **Agent edit surface:** tile sizes, warps, stages, accumulation dtype, GELU formulation (exact or approximate), fusion boundaries. No free-form kernel rewrites.
- **Verdicts:** the agent states a falsifiable prediction before each run, and the verdict (`confirmed`, `rejected`, `inconclusive`) is computed in code from metrics, never by the LLM. Noise band comes from measured run to run spread.
- **Lens-specific error bounds (option A):** the bound is set in code by the lens, never by the agent. With no directive the strict 1e-3 applies to the standard batch. A Speed directive sets a looser standard-batch tolerance for Speed branches, 2e-2 by default, parsed from the text by regex and clamped to at most 5e-2. A Precision directive sets 1e-3 (it can only tighten) on the standard and adversarial batches for every branch. Adversarial error is measured on every run, so a Precision directive re-judges logged runs without re-running. Each prediction and verdict carries a `Bound` (standard, adversarial, set_by) in the contract.
- **Failure by design:** FP16 accumulation is about 1.45x faster than the best FP32 tiles on this GPU but has standard error 1.3e-2 and adversarial error 7.1e-2 (fixed seed, identical every run). Under the Speed tolerance it is provisionally confirmed, and the Precision directive flips it to rejected, with a post-mortem and an error-vs-K sweep (1.9e-2 at K=256 up to 8.1e-2 at K=4096). If the agent does not propose FP16 itself, the harness adds it as a disclosed variant (plan Section 9), and the pitch says so: a seeded stretch hypothesis the agent tested, explained and judged in code.
- **Help requests:** routed to the relevant lens owner (badge on that tab). While blocked, the agent keeps exploring other branches. After the deadline it proceeds on a flagged default assumption.
- **Roles:** Joe plays both lenses by switching tabs. Say so in the pitch.
- **LLM:** originally NVIDIA NIM with Nemotron and DeepSeek. After the model benchmark (both were too slow), hypotheses go to OmniRoute, then `gemini-3.5-flash-lite`, then NIM `openai/gpt-oss-20b`, and post-mortem prose to `gemini-3.8-flash` with a factual fallback. Keys in a gitignored `.env`. See `CLAUDE.md` and `docs/results.md`.
- **Stack:** FastAPI backend with WebSocket, React (Vite) front end, layout `src/{harness,agent,server,ui}`, `runs/` gitignored, `tests/`.
- **Testing:** pytest and vitest. Markers `gpu` and `papiers` skipped by default. Golden test compares orchestrator events to a JSON export of `src/contract/demo-script.ts`. Per phase prompts are in `test-prompts.md`.
- **Repo rules:** commit and push are allowed, in small commits, but never `.env`, `runs/`, tokens or API keys, and the repo visibility stays as it is. No em dashes, minimal comments, concise names, no defensive boilerplate. Ask before assuming.
- **Demo freeze:** the demo replay is `demo/replay.jsonl`, committed and tagged `demo-freeze`, with the exports in `docs/examples/exports/`. Any new demo recording is frozen with `scripts/freeze.sh <session-folder>`, which checks the log ends with `session_finished` and has a `final_report`, and fails on secret patterns. The judging build opens `demo/replay.jsonl` first.

## Contract (src/contract)

`types.ts` defines state objects, `DemoEvent` and `Command`. `demo-script.ts` is a 66 event, 3 minute mock (flagged `mock: true`) with hypotheses H0 to H5: H0 tuned tiles confirmed, H1 FP16 accumulation provisional then rejected after the Precision directive, H2 inconclusive and triggers the help request, H3 safe pipeline confirmed, H4 approximate GELU confirmed after Joe's typed hunch, H5 hybrid confirmed. The UI's player (`src/ui/src/player.ts`) replays events with idle gaps over 4 s compressed, with presenter controls (pause, step, step back, chapters, speed).

## Demo

Default demo is replay of a recorded real run, frozen in `demo/replay.jsonl` (log every event as JSONL in the contract format, no mock flag), labeled "recorded run" on screen. Idle gaps compressed and disclosed. A "run one real iteration" option at the table was recommended for the Authenticity criterion but is not yet confirmed by Joe. A demo video is optional, skip unless time is left.

## Scope tiers

Tier 1 by Saturday 9pm: harness, loop with programmatic verdicts, lens tabs, directives, one fork, reproducible self rejection, summary cards, one help request, Papiers read and export, UI items 1 to 4. Tier 2 by Sunday 9:30am: hybrid merge, conflict banner, ledger table, comparative report, polish. Cut order: Papiers depth, ledger table, conflict banner, hybrid merge. Never cut the failure path, summary cards or the help request.

## Overrides to war-room-plan.md

- Findings are exported as Markdown for manual import (see `papiers.md`); the mock event is `findings_exported` and carries the exported file name. From CLI 0.1.6 final verdicts can also be written back as highlights, after a run, only for items a human approves, only for verdicts citing a stored passage, never from a replay or mock log, and read back in later sessions as context that is never evidence.
- UI is React with Vite, not the plain page suggested in plan Sections 12 and 19.
- Demo is replay based, no recording hour on Saturday, and no re-record step on Sunday.
