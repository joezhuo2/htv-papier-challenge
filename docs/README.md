# Docs

Start with `status.md` (running log, what is done, what is pending, key numbers).

- `status.md`: progress log and next steps.
- `architecture.md`: how the harness, agent loop, verdicts, lens bounds, help requests, Papiers, models, server and UI fit together.
- `runbook.md`: every command (UI modes, live server, headless sessions, measurements, tests, lint, screenshots) and troubleshooting.
- `findings.md`: measured results, timing stability, the FP16 failure, model benchmark, and what each real session showed.
- `contract.md`: the shared event contract and every change made to it, plus what is still open.
- `ui.md`: the war room UI as built, the feedback that shaped it, and assumptions.
- `ui-plan.md`: the original UI spec.
- `pitch-notes.md`: disclosures, safe numbers and story beats for the pitch and Devpost.
- `screenshots/`: captures of real sessions.

`war-room-plan.md` (repo root) is the original plan, `decisions.md` (repo root) records decisions that override it, `papiers.md` and `hackathon.md` hold the Papiers and challenge notes.

## Working rules for anyone continuing this

- Commit and push to main after each significant change. Check `git status` before each push and never commit `.env`, `runs/` or keys.
- No em dashes anywhere, paragraphs as continuous lines, minimal comments, concise names, no defensive boilerplate.
- Use Node 22 (`nvm use 22`) for anything in `src/ui` and for the Papiers CLI.
- Do not run tests or builds while a GPU session, the overnight batch or a timing measurement is running.
- Every number shown in the UI, the video or the writeup must come from a logged real run.
- Keep the mock script, the golden test and the UI reducer consistent with any contract change.
