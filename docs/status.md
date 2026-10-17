# Running log (Saturday Oct 17)

Read this first. Newest entries first in each list. Numbers, replay candidates and models per role are in docs/results.md.

## Done
- 11:05 Recorded runs/record2.jsonl on the final code: seeded FP16 provisional 13 s after the Speed directive, flip and conflict the instant the Precision directive arrived, post-mortem 15 s later, two help requests answered, hybrid proposed and approved, 196 s total with 64 s in model calls. The GPU then dropped out of WSL, so the hybrid run and two others crashed at CUDA startup. Not usable as the demo.
- 11:00 Replay fallback: demo.jsonl, then fallback1.jsonl, fallback2.jsonl and on (a missing or invalid file is skipped, the search stops at the first missing fallback number). The top bar shows which file is playing.
- 10:55 UI: Papiers ids render as note chips with the cleaned title, a provisional result shows a card and a "check" badge on the Precision tab (the cue for the Precision directive), Save is labeled "save partial replay" until session_finished.
- 10:50 Agent rework: commands (directives, help answers, hybrid approval) apply the moment they arrive through a listener thread, and help and hybrid deadlines run on the wall clock; the first Speed hypothesis is the harness-seeded FP16 variant of the best tiles, with its claim built from the bound and params; hybrid proposals after a conflict merge the fastest correct run of each lens (approve runs it once, past the cap); summaries are computed in code (confidence from the measured spread, no LLM); failed runs get a templated post-mortem with the shared memory arithmetic; a typed hunch becomes the next hypothesis on its lens branch or gets a plan_change saying why not; two failed runs in a row ask for help; duplicate answers are ignored silently; the cap reports promised iterations it cut; every branch gets a final status and the replay (runs/sessions/<id>/replay.jsonl) is written only after session_finished; failover and file paths go to debug.log, not the replay.
- 10:45 Shared memory is checked before any run: (stages - 1) x (block_m x block_k + block_k x block_n) x 2 bytes against 101376, matching both logged failures. The limit is in the prompt.
- 10:40 Models: post-mortems pinned to gemini-3.8-flash with a 15 s timeout and a factual fallback; a failed route is skipped for 300 s per model, or for the session on permission or bad request errors; OmniRoute auto/coding is first for hypotheses but its connected providers all refuse, so it falls through to Gemini.
- 10:20 Recording mode (Start recording): H0, then waits for a lens directive (90 s, then a flagged default on main), runs until Stop or 8 hypotheses. Start session stays the one-hypothesis table demo. Stop acts at once, New session restarts through POST /restart, the clock starts at 0:00 on Start, Start locks while starting, baseline failures show on the pre-flight card.
- 09:00 Live sessions live1 to live4 and record1 (see docs/results.md): the flip reproduced in every session where Precision came after FP16. Seeding FP16 on the naive tiles was a bug, fixed.

## Pending
- Record the demo session on the final code with a healthy GPU, following the recording plan, and copy it to runs/demo.jsonl. Check nvidia-smi first; run wsl --shutdown from Windows if it fails.
- Replay player work requested at 10:30, not started: per-gap caps (2.5 s, 5 s for a run), a real-time toggle, chapter labels per hypothesis and verdict, n and p shortcuts, exact-state seeks, and the 60 second tour.
- Part B checks not run yet: the fresh-clone README walk-through and the gpu and papiers marked test suites.
- OmniRoute needs a working provider connected in its dashboard (or OMNIROUTE=0 in .env).
- gemini-3.8-flash quota was exhausted at 10:54, post-mortems use the factual template until it resets.

## Earlier (overnight, 04:05 to 05:15)
- 05:15 Docs: everything from the overnight chat is now in docs/ (README index, architecture, runbook, findings, contract, ui, ui-plan, pitch-notes) with screenshots of candidate1 in docs/screenshots.
- 05:05 candidate1 (real session, final code, Gemini, interleaved timing): the flip reproduced (seeded FP16 on the Speed branch at 1.54x +/- 0.12x over its FP32 parent, provisional under 2e-2, rejected 3.6 s later by the Precision directive from logged metrics, post-mortem with the K sweep). Weak as a showcase: the best confirmed result is r1, 2.78x vs naive and 0.90x vs eager, seven of nine hypotheses missed, two help requests timed out to defaults. Not copied to runs/demo.jsonl.
- 04:55 Repo cleanup: oxlint for the UI (typescript-eslint does not support TypeScript 7) and Ruff for Python, both clean, with the flagged code fixed rather than silenced except two documented exceptions. README documents the judging build, pre-flight, stability script and lint commands. Tests: 84 vitest, 50 pytest, GPU tests pass.
- 04:55 Recording a fresh real session on the final code as a demo candidate (runs/sessions/candidate1), not yet copied to runs/demo.jsonl.
- 04:35 Polish list started in the UI: chapters (jump plus auto-pause), plain-language hover definitions, mode pill with the replay compression note, final result screen, clean first load and page title, live pre-flight panel (GPU, Gemini, NIM, Papiers checked for real by GET /preflight), visible timeout state with Retry and Stop, Save as replay, noise band (same-window speedups with +/- spread on cards, run details and chart error bars), judging build (`npm run build:judging`, mock hidden, falls back to /runs/demo.jsonl), replay log schema validation with line-numbered errors. Tests: 84 vitest, 50 pytest.
- 04:30 Golden test: tests/test_golden.py drives a fixed fake session and pins its event types and payload keys to tests/fixtures/golden_session.jsonl; src/ui/test/golden.test.ts checks that fixture passes UI log validation, folds to a finished session, and uses the same keys as the mock for every shared event type.
- 04:25 Live controls in the agent and server: the session waits for a Start command (AUTOSTART=1 skips it), LLM failures emit agent_error and wait for Retry or retry by themselves after 30 s, harness timeouts show as "timed out", Stop finishes the session, MAX_HYPS caps a live run (default 1), directives are rate limited (one per lens per 10 s, at most 6).
- 04:20 Gemini is the default model provider (gemini-3.5-flash-lite for hypotheses, gemini-3.8-flash for prose, reasoning_effort low), NIM is the per-call fallback and each fallback is logged. Key verified: models endpoint 200, test calls 1.8 s and 2.7 s.
- 04:20 Step 3 verified on the GPU: FP16 standard error 1.318e-2 on three runs (34% margin under 2e-2), provisional under Speed three times, then rejected under 1e-3 by the Precision directive with no run_started between the directive and the flip, post-mortem with a 5 point K sweep all over the bound.
- 04:18 Step 1 timing rework measured (runs/stability.log): every candidate is timed in one child process interleaved with naive, eager and its parent best over 7 rounds. Across 10 separate processes the speedups now vary by 0.7 to 0.9% (CV), versus 1.28x to 2.36x for the same comparison before. Verdicts use max(calibrated band, this run's within-window half-range, about 4%) as the noise, which is conservative.
- 04:17 Step 2 batch report (behavior only, numbers provisional, no replay taken from it): night1 and night2 ran the older code with NIM, night3 the interleaved timing with Gemini. The agent proposed FP16 itself once (night2, on main before any directive, rejected at the strict default). Every session needed the seeded FP16 variant on the Speed branch; it flipped provisional then rejected only in night3, the misses in night1 and night2 came from the old cross-process speed check. No implausible predictions (1.02 to 1.5x vs best, 3.5x vs naive). Every papiers_context event carries the real query, note id (cnv_...) and a 240 character excerpt, and the Precision query's empty result is logged as no matching notes. Help requests fired in night2 and night3, exports were written in all three.
- 04:05 Moved this file to docs/status.md.

## Key numbers (interleaved timing, 10 runs each)
- Best FP32 tiles (128x128x32, 4 warps, 3 stages): 2.75x vs naive (2.70 to 2.78), 1.03x vs eager (1.02 to 1.04).
- FP16 accumulation, same tiles: 4.03x vs naive, 1.52x vs eager, 1.48x vs the FP32 tiles (1.47 to 1.51), standard error 1.318e-2, adversarial 7.08e-2.
- Within-run spread about +/- 4%, across-run variation under 1%.
- More detail in docs/findings.md.

# Status and next steps (written 04:00)

As of Saturday Oct 17, about 4:00am ET. Update this file as work lands.

## Done

- Plan, decisions, contract and mock, Papiers notes (see `decisions.md`, `papiers.md`). Node 22 via nvm (`nvm use 22`), the system Node is still v18.
- Harness: kernel template, FP32 reference, subprocess runner with timeouts, eager and naive baselines, sweep, noise band, and a `sweep` target that measures error versus K for post-mortems.
- Model benchmark: `openai/gpt-oss-20b` for hypotheses, `google/gemma-4-31b-it` for post-mortems and summaries (defaults in `src/agent/llm.py`).
- Agent (`src/agent/`): hypothesis prompts with plausibility bounds on predictions (1.02 to 2.0x vs best, 1.1 to 10x vs naive), verdicts computed in code, lens-specific error bounds (option A in `decisions.md`), Speed and Precision branches, conflict detection, re-judging logged runs on a Precision directive, post-mortems with the K sweep and a pivot fork (only for error failures, speed misses and compile failures are ordinary misses), a seeded FP16 variant on Speed branches when the agent does not try it, help requests on inconclusive verdicts with a 45 s default, read-only Papiers context before each hypothesis, and Markdown export of negative results, branch summaries, help answers and the ledger to `runs/sessions/<id>/exports/`.
- Server (`src/server/app.py`): FastAPI, one session per process, `/ws` sends history then streams events and accepts validated commands.
- UI (`src/ui`): war room with mock, replay and live modes, presenter controls including prev, mock/live toggle, branch graph, metric chart with predicted markers, adversarial panel gated by the active bound, per-verdict bounds and history, help card, post-mortem, Papiers panel with reads and library search, ledger, report.
- Tests: 40 fast pytest tests, GPU tests (`-m gpu`), 63 vitest tests.

## Measured tonight

- FP16 accumulation: standard error 1.318e-2 and adversarial 7.08e-2 on every run (fixed seed), so it passes the Speed tolerance with a 34% margin and fails Precision on both batches. K sweep 1.9e-2 at K=256 to 8.1e-2 at K=4096.
- Speedup noise is far larger than the recorded 1.65% band: the same FP16 versus FP32 comparison measured 1.28x, 1.43x, 1.88x and 2.36x across runs in separate processes. Speedup verdicts near a threshold are unreliable until this is fixed.
- In 2 of 3 real sessions the agent proposed FP16 itself after reading the Papiers Triton conversation, but on main before any directive, where the strict bound rejects it. The seeded variant now also runs on Speed branches in that case. The full Speed then Precision flip has passed in fake-engine and GPU tests, but has not yet happened in a real LLM session on the current code.
- Best confirmed so far (session rep1, older code): 256x128x32, 16 warps, approximate GELU, 3.9x vs naive and 1.28x vs eager. Tentative until the final recorded run.

## Running overnight

- Three scripted headless sessions (`runs/sessions/night1` to `night3`, Speed directive after 1 verdict, Precision after 3) to check that the flip reproduces and to provide recorded replay candidates. Log of the batch: `runs/night.log`.

## Next, in order

1. Check the overnight sessions: did H on a Speed branch go provisional and then rejected by Precision, 3 of 3? Pick the best log as the recorded replay.
2. Fix the noise model: time the reference and the candidate in the same child process, back to back, and re-measure the band that way.
3. Live rehearsal: answer a help request from the UI, let one hit the deadline, check the export files.
4. Golden test of orchestrator event shapes against the mock contract.
5. Three dry runs, feature freeze at 5:30pm, record the demo, Devpost writeup and README, submit by 8:30pm. Disclose the seeded FP16 variant and the lens-specific bounds.

## Tier 2 (Sunday)

- Hybrid merge (`approve_hybrid` is only logged by the agent), comparative report (`report_ready` is never emitted), polish.

## Open items for Joe

- Confirm whether to add a "run one real iteration" option at the demo table.
- Load more of the Papiers reading set (the account has one conversation, no notes or highlights).
- Ask the Papiers team about a write path.
