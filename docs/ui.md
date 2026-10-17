# UI

The war room in `src/ui`: a Vite React TypeScript app with hand-built SVG charts, fonts bundled locally (Inter, JetBrains Mono) and no requests to other origins. `docs/ui-plan.md` is the original spec. This file records what was built, the feedback that shaped it, and the assumptions made.

## Layout

A top bar, then three columns that each scroll on their own (no page scroll at 1920x1080 or 1440x900, readable at 125 to 150% zoom).

- Top bar: title, a mode pill (`mock`, `recorded run` with "idle gaps compressed, A to B", or `live`), a mock/live toggle (replay/live in the judging build), chapters, elapsed time, presenter controls (pause/resume, prev, step, 1x/2x/4x, auto-pause, restart) in mock and replay, Save as replay and Stop in live.
- Left (steering): Speed and Precision tabs with a help badge, the active lens's help cards (compact: question, options and Send answer visible together, details folded), directives with a quiet metadata line and an inline conflict notice, the directive input, and the Papiers panel (every read with time, query, id and excerpt, highlighted briefly when read, "No matching notes" for empty searches, plus an on-demand library search through the dev server so the token never reaches the browser).
- Center (research): branch graph scaled to the panel (lanes only for lenses that have branches, rejected dashed and struck through with a red corner badge, blocked with a lock and "waiting for Speed owner", merged double outline, running pulse, dashed pending ticks), and the speedup chart (same-window speedup vs naive with noise bars, predicted rings that resolve to measured points, a legend for predicted, measured, inconclusive (half filled), provisional (dotted) and rejected) next to the adversarial panel ("not gated" until a Precision bound applies, then the bound line labelled with who set it, red crosses and "over bound" labels on failing runs whatever is selected). Tabs for Research, Report (Tier 2) and Result.
- Right (record): the pinned post-mortem with the error-vs-K sweep, the selected run's details (one block per verdict with its bound and check rows), a pending card while a run is measuring, the summary feed (older cards collapsed to one line, the newest expanded and kept in view, "meanwhile" tags, "Informed by note" links that jump to the Papiers read, verdict history with bounds), and the ledger (its "was ..." line lists only earlier verdicts that differ from the current one, since a Precision re-judge can repeat a verdict under the new bound).

## Behavior

- One reducer folds `DemoEvent`s for every mode. Every verdict is kept with its bound, the latest verdict's bound is the active bound.
- Mock plays the contract demo script and always shows the mock tag. Replay loads a JSONL file, validates every line (clear, line-numbered error banner), and compresses idle gaps over 4 s. Live connects to `/ws`, queues commands while disconnected, reconnects with backoff, and shows a banner that says how to start the server.
- The player fires events at t / speed and never drops or reorders them. Prev, seek and chapters re-fold the event prefix from scratch.
- Working states come only from real events: a pending card and tick between `run_started` and `run_finished` (live and mock, never recorded replay), and "Choosing next hypothesis" in live mode only, between a verdict and the next proposal.
- Live extras: pre-flight panel with real checks, Start, and a fallback link to the recorded run; a visible stalled card for model timeouts with Retry and Stop; failed runs show "timed out (15 s harness limit)"; Save as replay downloads the received events.
- Plain-language hover definitions for adversarial, bound, provisional, inconclusive, rejected, confirmed, noise band, speedup, eager and naive.
- Final result screen when a session finishes: the best result with params, same-window speedups and the bound it was judged against, every rejection with its reason, and the lens owners' directives, answers and plan changes with the exported files.
- Judging build (`npm run build:judging`): mock mode is never reachable and the app opens the first recorded run that loads, `runs/demo.jsonl`, then `runs/fallback1.jsonl`, `runs/fallback2.jsonl` and on (the top bar shows the file).

## Feedback applied (Saturday, in order)

- Right column: one scrolling column, cards never compressed; labels above values; graph scaled to its panel; ghost ring distinct from the eager line plus a legend; quieter directive metadata.
- Legend entry for provisional; adversarial panel ungated until the Precision directive; collapsed older cards with auto-follow; thin scrollbars; eager label off the edge.
- Help card that fits with its submit button; thin selection rings instead of focus boxes; half-filled inconclusive marker; rejected runs keep their red cross in the adversarial panel; "Adversarial error by run"; "meanwhile" tags.
- Per-verdict bounds everywhere (reducer, cards, run details, adversarial line).
- Prev button and the mock/live toggle; Papiers reads with query and "Informed by note" links; working states; chapters, definitions, honest labels, result screen, clean first load, live controls, judging build, log validation, lint.

## Assumptions

- The active bound is the latest verdict's bound, so the adversarial panel stays ungated between a Precision directive and the next verdict.
- A summary card shows only verdicts up to its own time, the earlier card of a flipped hypothesis keeps its snapshot plus a "superseded" tag.
- Pending cards appear in mock as well as live (mock events are scripted and labelled), never in recorded replay.
- Each Papiers read is its own card, a note returned by two queries appears twice and "Informed by" jumps to its first read.
- In replay the help countdown runs on compressed timeline time. Once the session has finished an unanswered card shows "closed" with the default it proceeded on, never a countdown (tested in `help.test.tsx`).
- The chart's eager reference line comes from the session baselines, while points use each run's own same-window speedup.
