# Pitch and Devpost notes

What to say, what to disclose, and which numbers are safe to quote.

## Disclosures (say these out loud or in the writeup)

- One person plays both lenses by switching tabs. In a real team the Speed and Precision owners would be two people.
- The FP16 accumulation hypothesis is the harness-seeded first hypothesis on the Speed lens (its note says "Harness-seeded variant, not proposed by the model", and its claim is built in code). The agent tests it, its verdict is computed in code, and it writes the post-mortem. In the logged sessions the agent proposed FP16 on its own once, on main before any directive, after reading the Papiers Triton tutorial conversation.
- Error bounds are lens-specific and set in code: the Speed lens tolerates 2e-2 on the standard batch, the Precision lens demands 1e-3 on the standard and adversarial batches. That is why FP16 is first provisional and then rejected.
- Papiers is read-only through its CLI. The agent reads it before each hypothesis and cites what it used. Findings are exported as Markdown files for manual import, the UI says "exported", never "written".
- Recorded replays compress idle gaps over 4 s, and the top bar says so with the before and after durations.
- Mock data is only for development and is always tagged mock; the judging build cannot show it.

## Numbers

- Quote only numbers from the recorded demo session, with the same-window speedups and their spread.
- Safe framing from the stability measurement: the agent's best FP32 configuration is about 2.75x faster than the naive start and about 3% faster than PyTorch eager; FP16 accumulation would be about 1.48x faster still but fails precision by 13x on the standard batch and 71x on the adversarial batch.
- Do not use the mock's 6.72x, the overnight batch numbers, or any cross-process speedup from before the interleaved timing change.

## Story beats (chapters in the UI)

1. First prediction, stated before the run, with the bound it will be judged against.
2. Speed directive, Speed branch, FP16 provisionally confirmed under the Speed tolerance.
3. Precision directive: the same logged run is re-judged and rejected, no new run, conflict detected, branch rejected.
4. Post-mortem with the error-versus-K sweep and the pivot that rules out FP16 accumulation.
5. Help request on an inconclusive result while another branch keeps working; answer it or let the default apply.
6. Final result screen: best result, what was rejected and why, what the humans decided, exported files.

## Screenshots

`docs/screenshots/candidate-*.png` are from candidate1 (a real session on the final code, 05:00 Saturday). It shows the flip well but its best result is weak (0.90x vs eager), so retake the screenshots from the chosen demo recording.
