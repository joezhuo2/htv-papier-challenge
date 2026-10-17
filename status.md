# Status and next steps

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
