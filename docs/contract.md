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

- A hypothesis or run id on `Summary`: the UI ties a summary to the newest finished run on its branch.
- A field on `PostMortem` for what the agent will not try again: the UI derives it from the pivot.
- A `dismiss_hybrid` command, and agent support for `approve_hybrid`, `hybrid_proposed` and `report_ready` (Tier 2, not implemented in the agent).
- Failed runs send null metric fields although `Metrics` types them as numbers.
