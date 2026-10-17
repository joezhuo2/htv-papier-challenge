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

Every Triton run also records `n_regs` (registers per thread) and `n_spills` (32-bit words of local memory per thread, Triton's `CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES / 4`) from the compiled kernel that `kernel.launch` returns. They are in `metrics.json` and in `run_finished` metrics, and the run details show them. `harness.eager_error` measures where PyTorch eager's error comes from, one FP16 rounding at a time (see docs/results.md).

## Agent loop (`src/agent/orchestrator.py`)

One session runs branches one at a time on the single GPU. A listener thread applies commands (directives, help answers, hybrid approval) the moment they arrive and checks help and hybrid deadlines on the wall clock. The loop holds a state lock except while it waits on a model or the GPU, and every GPU run takes a GPU lock, so a post-mortem sweep never overlaps a timed run.

1. Start: measure eager and the naive kernel, refuse to start if naive fails the 1e-3 gate, create the `main` branch (lens `both`), emit `session_started`.
2. Each iteration picks the running branch with the fewest iterations, reads Papiers (see Papiers below), and asks the fast model for a hypothesis (the prompt quotes the measured headroom from `src/harness/headroom.json`, the best sweep config at 2.75x vs naive and 1.03x vs eager, written by `harness.stability`, as information only): a claim, a prediction (`min_speedup` against `naive` or `parent_best`) and params. The parser refuses non-JSON, free-form code (any key outside Params), and implausible predictions (1.02 to 2.0x against the best so far, 1.1 to 10x against naive), and retries up to three times with the reason. A config over the shared memory budget ((stages - 1) x (block_m x block_k + block_k x block_n) x 2 bytes against 101376, stated in the prompt) is not refused: it is cut to the largest stage count that fits (`kernel.fit_stages`), and the claim is rebuilt in code from the final params (`claim_text`) with a note saying how many stages were cut, so claim and params never disagree. A config already tried anywhere in the session is refused. Model claims spell GELU in capitals.
3. The error bound in the prediction is not the model's choice: the code sets it from the branch's lens (see Lens bounds).
4. The run is measured with same-window references, and `verdict.judge` decides in code: `confirmed`, `rejected` or `inconclusive` (a speedup within the noise band, the larger of the calibrated band and the run's own spread). A result that passes only because the Speed tolerance is looser, or whose adversarial error is ungated, is marked provisional.
5. Outcomes: an error-bound failure on a lens branch rejects the branch, runs the K sweep post-mortem, exports a negative result, and forks a pivot branch that rules out the failing change (at most 3 branches per lens). A missed speedup is an ordinary miss, except that a run under 0.5x its parent gets a code-generated post-mortem naming the registers per thread, the spills and the accumulator elements per thread, with the parent's counts when they are logged, and a negative-result export. A run that does not compile or measure is a miss with a templated post-mortem (the shared memory arithmetic when Triton ran out of it, no model call), and two in a row ask for help. An inconclusive verdict asks for help (once per branch). Two misses in a row or the iteration budget closes a branch.
6. A summary with fixed fields is emitted after every iteration, computed entirely in code: confidence and its reason from the verdict, the checks and the run's measured spread, uncertainties from what is ungated or within noise, and what would change the agent's mind from the thresholds. Only the post-mortem cause and pivot use a model (the strong role, 15 s, factual fallback).
7. The session ends when nothing is runnable for `idle_s` (never in recording mode), when the cap is reached, or on Stop. It writes summaries for anything not yet summarized, reports promised iterations the cap cut, gives every open branch a final status, exports the ledger and summaries, emits `session_finished`, and only then writes `replay.jsonl` next to the events.

### Lens bounds (option A in `decisions.md`)

With no directive the strict 1e-3 applies to the standard batch. A Speed directive gives Speed branches a looser standard-batch tolerance, 2e-2 by default, parsed from the directive text by regex and clamped to at most 5e-2. A Precision directive sets 1e-3 (it can only tighten, never loosen) on the standard and adversarial batches for every branch, and re-judges every provisional result from its logged metrics without re-running anything. If that rejects a Speed candidate it emits `conflict_detected` and runs the post-mortem. Every prediction and verdict carries the `Bound` it was judged against.

### Seeded FP16 variant

The first hypothesis on the Speed lens is always the harness-seeded FP16 accumulation variant of the best tiles so far (the best confirmed run, or the fastest run that passes the strict bound), timed against those tiles with a 1.2x prediction and judged against the Speed bound. Its claim is built in code from the params and the bound, and its note says "Harness-seeded variant, not proposed by the model". A pending hunch runs right after it, with a plan_change saying so.

### Hunches and bounds

`hypothesis.hunch_change` reads a directive in code. A directive that names a concrete change (tanh or erf GELU, FP16 or FP32 accumulation, NxNxN tiles, N warps, N stages, or phrases like "bigger tiles" or "a cheaper GELU") is a hunch: it becomes the next hypothesis on the lens's open branch (or the branch it forks), noted "Hunch from the <lens> lens in directive <id>", and a plan_change says so. If that branch closes first the hunch moves to another open branch of the lens, and if it cannot be tested (cap, stop, no valid config after three tries, the model's first reading invalid) a plan_change says why. A constraint-only directive ("push throughput, I accept 2e-2") is not a hunch: its plan_change says it sets a bound, and the next hypothesis on that branch is noted "Tested under the <lens> lens bound set by <id>".

When a hunch's change was already measured (a logged hypothesis whose own change from its starting point was that change), the plan_change names the earlier hypothesis and its result, and the hunch is rerun in code as that change on the branch's current best params ("Retest of H2: ..."), predicting at least 1.02x against those params. A suggestion typed into a help answer works the same way.

### Hybrid

After a conflict and a Precision directive, once both lenses have a run that passes every active bound, the agent proposes merging their fastest such runs: each parameter that lens changed from main's start point, with the Precision lens winning where both changed the same one (`hybrid_proposed`). The union of two tile changes can overflow the registers (the 11:00 recording merged 256x128 and 128x256 into 256x256x64 with 8 warps, 256 accumulator elements per thread, 1492 spilled words, 0.09x), so a candidate must keep its tile area within the larger parent's and at most 128 FP32 (256 FP16) accumulator elements per thread. When the full union breaks that, the candidate keeps the faster parent's tile shape (block sizes, warps, stages), then the slower parent's, with the merged dtype and GELU, and the proposal text says which rule the union broke. Approval (`approve_hybrid`) forks a `merge` branch (parents both, label Hybrid) that runs the merged config once, allowed past the cap. A confirmed hybrid marks both parents merged. Without approval in the help deadline it is not run, and a later approval still runs it.

### Help requests

An inconclusive verdict sends a templated request to the branch's lens (main asks Speed): the question, three pieces of evidence, three options (keep tuning, accept, suggest something else), the default, and a deadline (45 s by default). The branch waits while others continue. After the deadline the agent proceeds on the default and says so in a plan change. Accept closes the branch, keep tuning adds two iterations, and a suggestion becomes the next hypothesis on that branch with a citation note. Late answers still revise the plan, and unknown or repeated answers are logged and ignored.

When a Precision directive flips a verdict, pending requests are refreshed. A request on a branch that the flip rejected is reissued with the same id, fresh evidence and a question saying the answer now applies to the successor (the open pivot branch of the same lens, which keeps running), or withdrawn with `help_cancelled` when no branch continues it. A request whose evidence lists the flipped run is reissued with the current verdicts. Answers and deadlines always land on the successor, a rejected branch is never reopened. Evidence lines say "provisionally confirmed" for provisional runs.

### Papiers (read-only)

`agent/papiers.py` calls the `papiers` CLI under Node 22, and `Library.retrieve(query)` follows the papiers-research skill in the npm package: the user's own judgement first. It runs `search QUERY --kind highlight` and `--kind note` in parallel, reads the highlights it found (`read hl_... --format plain`, the only call that returns `annotationText`), and when the search finds no marks it matches the session's `list --kind highlight` and `list --kind note` on shared words. Ranking: annotated highlights, then notes, then plain highlights. With no marks it falls back to `search --kind passage --filter annotated.eq=true`, then any passage, then conversations, stopping at the first that returns something. Each item carries `kind`, `document_title`, `highlight_text` (for highlights) and `annotation` (the user's words) besides `id`, `title` and `excerpt`.

Queries name what is being tested, not the directive text. Before the model proposes, the agent reads for the hunch's change or the lens topic (main "tile size shared memory", Speed "accumulation precision FP16", Precision "accumulation error grows with length") and puts the items in the prompt. Once a hypothesis exists (model, seeded, retest or hybrid) it reads for that hypothesis's own change (`facet`: accumulation dtype first, then the GELU formula, then tiling) and cites the top item in the note ("Context from Papiers highlight hl_..."), or cites nothing when that query finds nothing. Before a post-mortem it reads for the measured cause: "accumulation error grows with length" when the K sweep rises, "register pressure spills" for a slow run, "tile size shared memory" for a shared memory failure, and the change otherwise. The items go into the strong model's prompt and the top id into `post_mortem.note`. Each query is read once per session and logged as one `papiers_context` per item (or one without a note when nothing matched). The CLI is released from the state lock while it runs, and a failure is logged and the loop continues.

Papiers cannot be written to, so findings are exported as Markdown under `runs/sessions/<id>/exports/` (negative results with the K sweep, branch summaries, help answers, the ledger) and announced as `findings_exported`. Wording everywhere is "exported", never "written".

### Models (`agent/llm.py`, `agent/session.py`)

Hypotheses go to a local OmniRoute gateway first (`auto/coding`), then `gemini-3.5-flash-lite`, then NIM `openai/gpt-oss-20b`. Post-mortem prose is pinned to `gemini-3.8-flash` with a 15 s timeout (NIM `google/gemma-4-31b-it` only without a Gemini key), and a factual template stands in when it fails. A route that fails is skipped for 300 s, per model, or for the session on permission or bad request errors. Failovers, per-call durations and explanation fallbacks go to `runs/sessions/<id>/debug.log`, not the replay. Gemini runs through `https://generativelanguage.googleapis.com/v1beta/openai/` with `reasoning_effort: low`. Responses are cached on disk in `runs/llm_cache/`. If every provider fails, the session emits `agent_error` and waits for a Retry command, retrying by itself after 30 s.

### Live controls

Commands: `add_directive` (rate limited to one per lens per 10 s and six per session), `answer_help`, `start` (optional `record: true` for the recording mode, capped by `RECORD_HYPS` instead of `MAX_HYPS`), `stop` (acts at once, an in-flight model call is abandoned), `retry`, and `approve_hybrid` (runs a proposed hybrid). Duplicate help answers are ignored silently, and a plan_change that promised iterations the cap cut says so at the end. `MAX_HYPS` caps a live session (default 1 in the server). Harness timeouts produce an `agent_error` of kind `harness_timeout` and a failed run with `error: "timed out"`.

## Server (`src/server/app.py`)

One session at a time. It waits for a `start` command from the UI unless `AUTOSTART=1`, and `POST /restart` stops it and replaces it with a new session waiting for Start. `/ws` sends the session history as one array on connect (it starts with `session_started`, so the UI resets) and then each new event, and accepts validated commands. `/preflight` runs real checks: `nvidia-smi`, OmniRoute, Gemini and NIM `/models` with the keys, and the Papiers counts (highlights, notes and documents the CLI can see, with a warning when there are no highlights). Every session's log is `runs/sessions/<id>/events.jsonl`, the source of truth.

## UI (`src/ui`)

See `docs/ui.md`. One reducer folds `DemoEvent`s into view state for every mode. Mock plays `src/contract/demo-script.ts` (always tagged mock), replay plays a JSONL log (validated on load, idle gaps over 4 s compressed and disclosed), and live connects to `/ws` through the Vite proxy.

## Tests

`uv run pytest` (84 fast tests: verdicts, parsing, directive bounds, the loop with fake engine and fake LLM, the Precision flip, help requests, export, Papiers retrieval against a mocked CLI (highlight, note, empty, error), the hybrid guard, stage cuts, attribution, stale help requests, live controls, providers, server, the golden session), `uv run pytest -m gpu` (real kernel: K sweep, FP16 determinism and the flip from logged metrics), `cd src/ui && npm test` (101 vitest tests). `tests/test_golden.py` pins a fake session's event types and payload keys to `tests/fixtures/golden_session.jsonl`, and `src/ui/test/golden.test.ts` checks that fixture against the mock's keys and the UI's log validation.
