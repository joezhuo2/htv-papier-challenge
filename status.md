# Status and next steps

As of Saturday Oct 17, about 2:45am ET. Update this file as work lands.

## Done

- Plan v3 (`war-room-plan.md`), `CLAUDE.md`, kickoff prompts (`kickoff-prompts.md`), per phase test prompts (`test-prompts.md`).
- Shared contract and mock in `src/contract/` (`types.ts`, `demo-script.ts`). `replay.ts` was never added to the repo, the UI has its own player in `src/ui/src/player.ts`.
- Hackathon rules read and categories chosen (see `hackathon.md`). Papiers docs read (see `papiers.md`).
- Node 22 is installed through nvm (v22.23.3). Put it on PATH with `nvm use 22` before any npm or vitest command, the system Node is still v18.
- Phase 1a harness tests and the measured noise band (`src/harness/noise_band.json`, cv 0.55%, band 1.65%), commit `aacd2e6`.
- UI in `src/ui`, built against the mock (commit `a77eaa7`): three column war room, player with pause, resume, step and speed, reducer, branch graph, metric chart with predicted markers, help card, post-mortem, hybrid card, ledger, report view, and `?mode=mock`, `?mode=replay&file=...`, `?mode=live`. 48 vitest tests, run with `npm test` in `src/ui`. Dev server with `npm run dev`, recorded logs are served from `runs/` at `/runs/...` in dev.
- Papiers is read-only: the export event is now `findings_exported`, each item carries a `file`, and the UI labels it "Exported for Papiers import" (commit `4b13cea`).

## In progress

- **Prompt 2, model benchmark:** running in another session (`runs/model_bench.log`), with faster NIM fallback models alongside Nemotron and DeepSeek. The edits to `src/agent/model_bench.py` are not committed yet.

## Next, in order

1. Finish the model benchmark and pick the fast and strong models.
2. Phase 1b: the hypothesis loop with a falsifiable prediction, edits limited to the kernel template knobs, a subprocess run, and a verdict computed in code. Log every event as JSONL in the contract format without the mock flag. Gate (was Saturday 3:30am): one autonomous iteration with a code computed verdict.
3. Failure path: H1 with FP16 accumulation rejected by the Precision directive, with a post-mortem and an error versus K sweep, reproducing 3 of 3. First confirm that FP16 accumulation is actually faster on the 5060. Gate: Saturday noon.
4. FastAPI server with `/ws` streaming `DemoEvent`s and accepting `Command`s. Decide whether the server resends the session history on reconnect (the UI resets on `session_started`, so either works) and whether to add a `dismiss_hybrid` command.
5. Help flow and golden test by 2:30pm, two clean full live runs and a saved replay by 5:30pm, final run with the hybrid and chosen replay file by Sunday 7:30am.

## Notes

- The mock numbers are placeholders on a different scale from real runs (the harness measures about 0.64 ms for a 128x128x32 config, the mock shows 5 to 41 ms). The demo must use a recorded real run, the UI labels mock data "mock" and recorded logs "recorded run".
- Remaining UI contract requests: a hypothesis or run id on `Summary`, and a field on `PostMortem` for what the agent will not try again (the UI derives both for now).

## Open items for Joe

- Sleep plan is undecided (the 4 hour version, the full block, or none). Set hard no-code-change times: Saturday 5:30pm and Sunday 7:30am.
- Confirm whether to add a "run one real iteration" option at the demo table.
- Load the real Papiers reading set with highlights and notes (see `papiers.md`).
- Add the README and submit a minimal initial Devpost entry early Saturday with the chosen categories.
- Ask the Papiers team about a write path.
