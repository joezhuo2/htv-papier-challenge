# Contract

`src/contract/types.ts` is the shared language between the agent (Python, which emits it as JSON) and the UI. `src/contract/demo-script.ts` is a 3 minute scripted mock in the same format (every event has `mock: true`). Every session log is JSONL in this format. Changes must keep the mock, the golden test (`tests/test_golden.py` and `src/ui/test/golden.test.ts`) and the UI reducer consistent.

## Changes made on Saturday Oct 17

1. `papiers_write` renamed to `findings_exported`, each item gains `file` (Papiers is read-only, findings are exported for manual import). The mock's last next step no longer says it writes to Papiers.
2. `Bound { standard, adversarial | null, set_by: Lens | 'default' }` added to `Prediction.bound` and to every `verdict` event, so a run judged under the Speed tolerance is distinguishable from one judged under Precision. `Prediction.max_error` equals `bound.standard`. The mock carries bounds (H0 default, H1 first verdict Speed 2e-2, everything after the Precision directive Precision 1e-3 on both batches).
3. `papiers_context` gains an optional `query`, and `note` became optional (absent when a search returns nothing). The mock's two reads carry queries.
4. `Metrics` gains optional `speedups` and `spreads` (keys `naive`, `eager`, `parent`), measured in the same timing window as the run.
5. `run_finished` gains an optional `error` (`"timed out"` for the harness timeout, otherwise the failure text). A failed run's metrics fields are null.
6. New event `agent_error { kind: 'llm_timeout' | 'llm_error' | 'harness_timeout', text, branch?, retry_in_s? }`.
7. `Command` gains `start`, `stop` and `retry`.
8. `start` takes an optional `record: true` (recording mode, up to `RECORD_HYPS` hypotheses, the log is saved as a replay). The default start keeps the live demo cap. No event shapes changed.

## Still open

- A field on `PostMortem` for what the agent will not try again: the UI derives it from the pivot.
- A `dismiss_hybrid` command, and agent support for `report_ready` (the comparative report is only in the mock).

## Changes made at 10:50

9. The agent emits `hybrid_proposed` and acts on `approve_hybrid` (a `merge` branch with both parents, label Hybrid, status `merged` on the parents when it is confirmed).
10. `plan_change.help` is the id it answers: a help request (`hq1`) or a directive (`d2`) for hunch replies.
11. Failed runs get a `post_mortem` with an empty sweep. Every branch has a final `branch_status` before `session_finished`, and provider failover lines are no longer `agent_log` events.

## Changes made at 11:10 to 11:45

12. `Metrics` gains optional `n_regs` and `n_spills` from the compiled Triton kernel (absent for eager).
13. New event `help_cancelled { help, reason }`: a pending request withdrawn after a flip rejected its branch with no successor. A reissued request is a second `help_requested` with the same id, which the UI replaces in place.
14. `papiers_context.note` is `PapiersNote { id, title, excerpt, kind?, document_title?, highlight_text?, annotation? }`. Old logs without the new fields stay valid.
15. `PostMortem` gains optional `note`, the Papiers id retrieved for the measured cause.
16. Hypothesis notes cite "Context from Papiers <kind> <id>", and constraint-only directives are noted "Tested under the <lens> lens bound set by <id>" instead of "Hunch from".

## Changes made at 16:30

17. A Precision directive re-judges every measured run whose last verdict has a different bound, so `verdict` events can repeat a hypothesis's verdict with the new bound (for example H0 confirmed under default, then confirmed under Precision). Old logs stay valid. The Precision branch's `branch_created` now comes before the re-judge's `verdict` and `conflict_detected` events.
18. `conflict_detected.reason` names the batches that failed, `verdict.reason` for a provisional result names only the errors over the strict bound, and claims and pivots use readable settings ("FP16 accumulation", "approx GELU") instead of `acc=fp16`.
19. `run_finished.run.metrics` is `Metrics | FailedMetrics`: a run that failed to compile, crashed or timed out sends null for latency, throughput, both errors and VRAM, and the type now says so (it was typed as numbers).
20. `Summary` gains optional `hypothesis`, the hypothesis it is about. The UI used to tie a summary to the newest finished run on its branch, which put H1's post-flip summary on H2 in the 16:00 recording; logs without the field still use that fallback.

## Changes made after 17:45

21. New event `final_report { report: FinalReport }`, sent just before `session_finished`: `best` (run, hypothesis, branch, params, latency and speedups, or null), `ruled_out`, `next`, `human` (decisions that need a lens owner) and `hunches` (one `{ directive, lens, text, outcome }` per typed hunch). Logs without it stay valid and the result screen renders without it.
22. Every rejected verdict now has a `post_mortem`, speed misses included (empty sweep, cause from registers, spills and the parameter diff). `post_mortem.note` is the retrieved item that best matches the cause, not the first search hit.
23. A hunch directive gets exactly one `plan_change`, when it arrives, saying whether it is a retest; later notices about it are `agent_log` lines. A help request still open when the session ends gets one `plan_change` "Session ended with hqN unanswered: <why>".
24. `help_requested` is reissued at most once per re-judge and only when it changed. A request for a branch rejected while its run was measured is filed against the successor (`blocks` names it), and a rejected branch never gets another `branch_status`.
25. `papiers_context.note.annotation` has LaTeX dollar delimiters removed.
26. `hybrid_proposed` no longer needs a `conflict_detected` first: it combines the best confirmed run with the best confirmed single-knob change from another branch, and can arrive at the cap, where the session waits for approval or the deadline.
27. `Hypothesis` gains optional `papiers`, the id of the Papiers item its note cites; the note names the document instead of the id. Older logs keep the id in the note.
28. `session_finished.best` and `final_report.best` give vs naive and vs eager as the session baselines' latency over the best run's latency (the best is the fastest confirmed run), not its same-window ratios.

## Changes made after the 14:38 recording

29. Every inconclusive verdict also gets a `post_mortem` (empty sweep). The UI badge on a post-mortem card shows the hypothesis's verdict, `rejected` or `inconclusive`.
30. A directive whose concrete change gives an exact config that already ran, is being measured or is queued gets one `plan_change` naming that run ("I am not running it again on speed2 because that exact config ... is being measured right now as H2 on speed"), and `final_report.hunches[].outcome` is that run's result. A constraint-only directive at the cap or after Stop gets one `plan_change` saying the bound applies to the logged runs only.
31. A help answer, or the deadline default, promises at most what the hypothesis cap leaves after iterations already promised. When none fit, its `plan_change` says so and the branch closes. `help_requested.what_changes` states the same number.
32. `hybrid_proposed` merges only the changes of a run at 1.00x or better over its parent that passes every active bound, and the best config has to pass them too. When no run qualifies, an `agent_log` line before `final_report` lists each candidate and why it did not qualify.
33. `papiers_context.note.document_title` for a conversation is the library document whose title shares the most words with it (at least two), otherwise the conversation's own title. Hypothesis claims, notes and post-mortem prose never contain raw Papiers ids.
