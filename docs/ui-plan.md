# UI plan (original spec, as given on Saturday Oct 17)

Scope: `src/ui`, React with Vite, built against the mock in `src/contract/` first, then pointed at the real WebSocket. Where this plan disagrees with `war-room-plan.md`, this plan wins for the UI. `docs/ui.md` records what was built and how it changed.

## Principles

A clean working tool, not a landing page. The first screen is the war room itself. Calm, quiet, readable from a judge's table at 125 to 150 percent zoom. Match the Papiers look: black text on light beige, thin borders, generous whitespace, almost no shadows, no gradients, no emoji. Every state is understood from text or a glyph, never from color alone. No network requests at load: fonts and assets are bundled.

## Tokens

- Background `#F6F2EA`, surface `#FBF9F4`, ink `#111111`, muted text `#6B665C`, border `#E0DACB`, hover surface `#EFE9DC`.
- Lens accents used only as a 2px line, a tab underline and a faint background tint (about 4 percent): Speed ink blue `#2F4B7C`, Precision terracotta `#B5573A`. Switching tabs shifts the tint and the accent line, nothing else.
- Verdict colors, always paired with a glyph and a word: confirmed `#2E6B4F` (check), rejected `#9B2C2C` (cross), inconclusive `#8A6D1F` (tilde), provisional is an ink outline with a dotted border.
- Type: bundled sans (Inter or system-ui fallback), tabular numerals for all metrics, a mono face for ids and params. Sizes 12, 14, 16, 20, 28. One weight step for emphasis, no all caps except tiny labels.
- Radius 6px, borders 1px, spacing on a 4px grid, no drop shadows except a 1px border on focused cards.

## Layout (three columns, fixed viewport, no page scroll)

- Top bar: project name, a "recorded run" or "live" pill, elapsed time, replay controls (pause, resume, step, speed 1x/2x/4x, restart). Presenter controls are small and out of the way.
- Left column (320px), steering: lens tabs (Speed, Precision) with a badge dot when a help request is waiting for that lens. Under the tabs: the active lens's directive list, a directive input with a Send button, and help cards addressed to that lens with an answer box and a countdown to the default assumption.
- Center column (flexible), the research: branch graph on top (about 45 percent height), metric chart below. Clicking a graph node or chart point selects that run and opens it in the right column.
- Right column (380px), the record: summary feed of structured cards (what was tried, prediction, result, verdict, why, next step), the hypothesis ledger table below a divider, and the post-mortem card pinned when a hypothesis is rejected.

## Components

1. Lens tabs and directive panel. Directives show text, lens, time and the branch they affected. After a directive that conflicts with an active branch, show a one-line inline notice "Conflicts with branch X, forked" with a link to the new node. Typed hunches use the same input.
2. Branch graph (hand-built SVG). Nodes are branches laid out left to right by creation order, with edges for forks and the merge. Node states: running (pulsing 1px ring, reduced motion respected), done (solid), rejected (dashed outline, struck-through label, small cross), blocked (hollow with a lock glyph and "waiting for Speed owner"), merged (double outline). Run ticks sit on each branch line and are clickable.
3. Metric chart (hand-built SVG). Latency or speedup against run index, one line per branch in neutral grays with the selected branch in ink, eager and naive baselines as thin dashed reference lines labeled. Before each run a ghost marker shows the predicted value, and after the run it resolves to the measured point with a short connector. Adversarial error appears as a second small panel for the selected run, with the 1e-3 threshold line, and the error-versus-K sweep lives inside the post-mortem.
4. Summary card. Fixed order: title with verdict chip, prediction (stated before the run), measured result with units, delta versus the previous best and versus eager, one sentence of reasoning, and the next step the agent chose. Rejected cards add the post-mortem link. Cards fade in without moving earlier ones.
5. Help card. Appears on the addressed lens tab with a badge. It shows the question, why the agent is blocked, what it keeps doing meanwhile, the default assumption, and the deadline countdown. Answering produces a plan change entry in the feed. After the deadline the card shows "proceeding on assumption" and downstream results carry an "assumption pending" tag.
6. Post-mortem card. For a rejected hypothesis: what was predicted, what happened, the error-versus-K sweep as a small SVG line chart, the mechanism in two sentences, and what the agent will not try again.
7. Hybrid proposal (Tier 2). A card with Approve and Dismiss, showing the merged params from both lenses and the predicted result. Approve sends the `approve_hybrid` command.
8. Ledger table (Tier 2). One row per hypothesis: id, lens, prediction, result, verdict, branch, Papiers note id when present. Chronological order, no sorting.
9. Papiers badge. Hypotheses that cite a note show a small "Papiers note" tag. Findings export is labeled "exported" with the file name, never "written".

## State and data

- A single reducer folds `DemoEvent` into view state: branches, hypotheses, runs, verdicts, summaries, help requests, directives, ledger, plan changes, current lens, selected run. The same reducer serves replay and live.
- The source is switchable by a query param: `?mode=mock` (default for dev, plays `demoScript`), `?mode=replay&file=...` (a recorded JSONL log in the contract format), `?mode=live` (WebSocket to FastAPI, sends `Command`s).
- The replay helper gains pause, resume and step controls that never drop or reorder events. Mock events render a visible "mock" tag. Recorded logs show "recorded run" with a note that idle gaps are compressed.

## Interactions and states

- Keyboard: 1 and 2 switch lens, Space pauses replay, Right arrow steps, Esc clears selection. Focus rings are visible.
- Empty states: short quiet text ("No runs yet"), never spinners that lie. The agent log is a collapsed drawer at the bottom, off by default.
- Errors: a thin banner at the top for a dropped WebSocket with reconnect, no modal dialogs.
- Motion: only fades and the running ring, under 200ms, disabled under `prefers-reduced-motion`.

## Acceptance (manual checklist at presentation resolution)

- Rejected branch visible in the graph and in the feed.
- Ghost prediction marker resolves to the measured point.
- Help badge appears on the correct lens tab and the blocked branch shows a lock.
- Lens tint and accent change on switch, and text stays black on beige and readable.
- Verdicts readable without color.
- Readable at 125 to 150 percent zoom with no page scroll at 1920x1080 and 1440x900.
- No console errors, no requests to other origins on load.

## Out of scope

Landing page, marketing copy, login, settings screens, dark mode, mobile layout, animations beyond the running ring and fades, any chart library.
