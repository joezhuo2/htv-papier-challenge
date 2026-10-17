# Status and next steps

As of Saturday Oct 17, about 1:45am ET. Update this file as work lands.

## Done

- Plan v3 (`war-room-plan.md`), `CLAUDE.md`, kickoff prompts (`kickoff-prompts.md`), per phase test prompts (`test-prompts.md`).
- Shared contract and mock in `src/contract/` (`types.ts`, `demo-script.ts`, `replay.ts`), type checked and consistency checked.
- Hackathon rules read and categories chosen (see `hackathon.md`). Papiers docs read (see `papiers.md`).

## Pending (reported by Claude Code)

- **vitest:** blocked on the Node 22 upgrade, since WSL2 is on Node v18.19.1. The Papiers CLI also needs Node 22 or newer.
- **Phase 1a harness tests:** the six GPU tests, the noise band file and the 3 minute thermal drift check are not written yet.
- **Prompt 2, model benchmark:** not started. It will include faster NIM fallback models alongside Nemotron and DeepSeek.
- The Part A checkpoint results (GPU, Triton, NIM key) are not recorded here, so add them once confirmed.

## Next, in order

1. Upgrade Node in WSL2: install nvm (github.com/nvm-sh/nvm), then `nvm install 22`, `nvm use 22`, `node -v`. Reinstall the Papiers CLI afterward (`npm install -g papiers`).
2. Add vitest and finish the shared test setup.
3. Write the Phase 1a harness tests and run `uv run pytest -m gpu`. Gate: Saturday 2:00am, Triton works and the baseline passes.
4. Run Prompt 2 (model benchmark), then Phase 1b. Gate: Saturday 3:30am, one autonomous iteration with a code computed verdict.
5. Later gates: failure path reproduces 3 of 3 by Saturday noon, help flow and golden test by 2:30pm, two clean full live runs and a saved replay by 5:30pm, final run with the hybrid and chosen replay file by Sunday 7:30am.

## Open items for Joe

- Sleep plan is undecided (the 4 hour version, the full block, or none). Set hard no-code-change times: Saturday 5:30pm and Sunday 7:30am.
- Confirm whether to add a "run one real iteration" option at the demo table.
- Copy `src/contract/` into the repo. Build the UI session against the mock once it is there.
- Load the real Papiers reading set with highlights and notes (see `papiers.md`).
- Create the public GitHub repo with a README and `.gitignore` for `.env` and `runs/`. Submit a minimal initial Devpost entry early Saturday with the chosen categories.
- Presenter controls (pause, resume, step) in `replay.ts` are not added yet.
- Ask the Papiers team about a write path.

Joe commits and pushes. Do not run `git commit` or `git push`.
