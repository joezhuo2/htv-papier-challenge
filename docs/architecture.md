# Architecture

How the war room works end to end, as built on Saturday Oct 17. `war-room-plan.md` is the original spec, `decisions.md` records where the build overrides it, and this file describes what the code actually does.

## Components

```
src/
  harness/   Triton kernel template, inputs and FP32 reference, experiment child process, runner, bench, noise band, stability
  agent/     orchestrator (loop, branches, help, export), verdict, hypothesis prompts, Papiers adapter, LLM client and session wiring
  server/    FastAPI: /ws streams the session's events and takes commands, /preflight checks the environment, /health
  contract/  shared TypeScript types (types.ts) and the mock demo script
  ui/        Vite React war room (mock, recorded replay, live)
runs/        evidence, sessions and exports (gitignored)
tests/       pytest suites, the golden session fixture, and the contract smoke test
```

## Harness

The kernel is a fused matmul + bias + GELU Triton kernel whose only editable surface is `Params` (tile sizes, warps, stages, accumulation dtype, GELU formulation). The naive start is 16x16x16 tiles, 1 warp, 1 stage, FP32 accumulation, exact GELU. Every experiment runs in a fresh child process (`harness.experiment`) started by `harness.runner`: the child imports torch and initializes CUDA (about 9 s here), prints `ready`, then gets a hard 15 s timeout after which its whole process group is killed. Evidence for each run lands in `runs/<run_id>/` (spec, stdout, stderr, metrics).

Correctness is `max |y - ref| / max(|ref|, 1)` against an FP32 reference built from the same FP16 inputs, with fixed-seed inputs so errors are identical on every run. The standard batch is M = N = K = 2048. The two adversarial cases are 512 x 512 x 16384 (long K) and 8x larger pre-activations, reported together as `adv_error` on every run.

Timing (since 04:10 Saturday): a candidate is timed in the same child process as its references (naive, eager and, when it has one, the parent best config), interleaved over 7 rounds of 20 repetitions with the order rotated each round. Speedups are the median of per-round ratios, and each result carries `spreads`, the relative half-range of those ratios. Before this change speedups compared against baselines timed in other processes and swung wildly (the same comparison measured 1.28x to 2.36x). `harness.stability` measures the result: under 1% variation across processes.

The `sweep` target measures one config's error across K (256 to 4096) on the large-magnitude inputs, for post-mortems.

## Agent loop (`src/agent/orchestrator.py`)

One session runs branches one at a time on the single GPU, applying queued commands between iterations.

1. Start: measure eager and the naive kernel, refuse to start if naive fails the 1e-3 gate, create the `main` branch (lens `both`), emit `session_started`.
2. Each iteration picks the running branch with the fewest iterations, reads Papiers for that branch's topic, and asks the fast model for a hypothesis (the prompt quotes the measured headroom from `src/harness/headroom.json`, the best sweep config at 2.75x vs naive and 1.03x vs eager, written by `harness.stability`, as information only): a claim, a prediction (`min_speedup` against `naive` or `parent_best`) and params. The parser refuses non-JSON, free-form code (any key outside Params) and implausible predictions (1.02 to 2.0x against the best so far, 1.1 to 10x against naive), and retries up to three times. A config already tried anywhere in the session is refused.
3. The error bound in the prediction is not the model's choice: the code sets it from the branch's lens (see Lens bounds).
4. The run is measured with same-window references, and `verdict.judge` decides in code: `confirmed`, `rejected` or `inconclusive` (a speedup within the noise band, the larger of the calibrated band and the run's own spread). A result that passes only because the Speed tolerance is looser, or whose adversarial error is ungated, is marked provisional.
5. Outcomes: an error-bound failure on a lens branch rejects the branch, runs the K sweep post-mortem, exports a negative result, and forks a pivot branch that rules out the failing change (at most 3 branches per lens). A missed speedup or a compile failure is an ordinary miss. An inconclusive verdict asks for help (once per branch). Two misses in a row or the iteration budget closes a branch.
6. A summary with fixed fields is emitted after every iteration. Its prose fields (uncertainties, what would change my mind) come from the strong model, with a factual fallback if that model fails. The post-mortem cause and pivot likewise.
7. The session ends when nothing is runnable for `idle_s`, when the live cap is reached, or on Stop, and exports the ledger and branch summaries.

### Lens bounds (option A in `decisions.md`)

With no directive the strict 1e-3 applies to the standard batch. A Speed directive gives Speed branches a looser standard-batch tolerance, 2e-2 by default, parsed from the directive text by regex and clamped to at most 5e-2. A Precision directive sets 1e-3 (it can only tighten, never loosen) on the standard and adversarial batches for every branch, and re-judges every provisional result from its logged metrics without re-running anything. If that rejects a Speed candidate it emits `conflict_detected` and runs the post-mortem. Every prediction and verdict carries the `Bound` it was judged against.

### Seeded FP16 variant

If a Speed branch reaches its second iteration and FP16 accumulation has not been tested on a Speed branch, the harness proposes it on the branch's best config with a 1.2x prediction, and the hypothesis note says so ("Harness variant from the precision sweep (plan Section 9)"). A human suggestion from a help answer takes priority over it.

### Help requests

An inconclusive verdict sends a templated request to the branch's lens (main asks Speed): the question, three pieces of evidence, three options (keep tuning, accept, suggest something else), the default, and a deadline (45 s by default). The branch waits while others continue. After the deadline the agent proceeds on the default and says so in a plan change. Accept closes the branch, keep tuning adds two iterations, and a suggestion becomes the next hypothesis on that branch with a citation note. Late answers still revise the plan, and unknown or repeated answers are logged and ignored.

### Papiers (read-only)

`agent/papiers.py` calls the `papiers` CLI (`papiers search ... --json`) under Node 22. Before each hypothesis the agent searches once per query (the branch's latest directive text, or a lens default topic), logs one `papiers_context` per returned item with the query, id, title and excerpt (or one without a note when the search is empty), puts the excerpts in the prompt, and cites the top item on the hypothesis ("Context from Papiers note <id>"). A search failure is logged and the loop continues. Papiers cannot be written to, so findings are exported as Markdown under `runs/sessions/<id>/exports/` (negative results with the K sweep, branch summaries, help answers, the ledger) and announced as `findings_exported`. Wording everywhere is "exported", never "written".

### Models (`agent/llm.py`, `agent/session.py`)

Gemini is the default provider when `GEMINI_API_KEY` is set: `gemini-3.5-flash-lite` for hypotheses and `gemini-3.8-flash` for post-mortems and summaries, through `https://generativelanguage.googleapis.com/v1beta/openai/` with `reasoning_effort: low` (thinking cannot be turned off on 3.x). Each call falls back to NIM for the same role (`openai/gpt-oss-20b` fast, `google/gemma-4-31b-it` strong) and logs the fallback. Responses are cached on disk in `runs/llm_cache/`. If every provider fails, the session emits `agent_error` and waits for a Retry command, retrying by itself after 30 s.

### Live controls

Commands: `add_directive` (rate limited to one per lens per 10 s and six per session), `answer_help`, `start` (optional `record: true` for the recording mode, capped by `RECORD_HYPS` instead of `MAX_HYPS`), `stop` (acts at once, an in-flight model call is abandoned), `retry`, and `approve_hybrid` (logged only, not implemented in the agent). `MAX_HYPS` caps a live session (default 1 in the server). Harness timeouts produce an `agent_error` of kind `harness_timeout` and a failed run with `error: "timed out"`.

## Server (`src/server/app.py`)

One session at a time. It waits for a `start` command from the UI unless `AUTOSTART=1`, and `POST /restart` stops it and replaces it with a new session waiting for Start. `/ws` sends the session history as one array on connect (it starts with `session_started`, so the UI resets) and then each new event, and accepts validated commands. `/preflight` runs real checks: `nvidia-smi`, Gemini and NIM `/models` with the keys, and a Papiers search. Every session's log is `runs/sessions/<id>/events.jsonl`, the source of truth.

## UI (`src/ui`)

See `docs/ui.md`. One reducer folds `DemoEvent`s into view state for every mode. Mock plays `src/contract/demo-script.ts` (always tagged mock), replay plays a JSONL log (validated on load, idle gaps over 4 s compressed and disclosed), and live connects to `/ws` through the Vite proxy.

## Tests

`uv run pytest` (50 fast tests: verdicts, parsing, directive bounds, the loop with fake engine and fake LLM, the Precision flip, help requests, export, Papiers, live controls, providers, server, the golden session), `uv run pytest -m gpu` (real kernel: K sweep, FP16 determinism and the flip from logged metrics), `cd src/ui && npm test` (84 vitest tests). `tests/test_golden.py` pins a fake session's event types and payload keys to `tests/fixtures/golden_session.jsonl`, and `src/ui/test/golden.test.ts` checks that fixture against the mock's keys and the UI's log validation.
