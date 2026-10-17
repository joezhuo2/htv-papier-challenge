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
