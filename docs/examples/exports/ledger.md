# Hypothesis ledger

Naive 2.009 ms, PyTorch eager 0.665 ms. Best run r5 (inconclusive): 2.59x vs naive, 0.97x vs eager, same-window.

| id | branch | prediction | result | verdicts | final bound | note |
|---|---|---|---|---|---|---|
| H0 | main | 2.00x vs naive | 0.783 ms, std 4.9e-4, adv 5.2e-4 | confirmed then confirmed | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Claim rebuilt from the final params after three mismatched claims, the last one said it names 16x16x16, the params do not. Context from the Papiers conversation "Matrix Multiplication — Triton  documentation" |
| H1 | speed | 1.20x vs parent_best | 0.514 ms, std 1.3e-2, adv 7.1e-2 | confirmed then rejected | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Harness-seeded variant, not proposed by the model: low-precision accumulation tested under the active Speed bound. Context from your highlight in "MIXED PRECISION TRAINING" |
| H2 | speed | 1.02x vs parent_best | 0.735 ms, std 4.9e-4, adv 5.2e-4 | inconclusive | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Tested under the Speed lens bound set by d1. Context from the Papiers conversation "Matrix Multiplication — Triton  documentation" |
| H3 | prec | 1.05x vs parent_best | 0.729 ms, std 4.9e-4, adv 5.2e-4 | inconclusive | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Tested under the Precision lens bound set by d2. Context from the Papiers conversation "Matrix Multiplication — Triton  documentation" |
| H4 | speed2 | 1.05x vs parent_best | 0.719 ms, std 6.9e-4, adv 7.2e-4 | inconclusive | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Hunch from the Speed lens in directive d3. Context from your highlight in "Gaussian Error Linear Units (GELUs)" |
| H5 | speed2 | 1.10x vs parent_best | 0.730 ms, std 6.9e-4, adv 7.2e-4 | inconclusive | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Context from your highlight in "Gaussian Error Linear Units (GELUs)" |
| H6 | prec | 1.03x vs parent_best | 0.731 ms, std 4.9e-4, adv 5.2e-4 | inconclusive | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Context from the Papiers conversation "Matrix Multiplication — Triton  documentation" |
| H7 | prec | 1.03x vs parent_best | 0.825 ms, std 4.9e-4, adv 5.2e-4 | rejected | standard under 1.0e-3, adversarial under 1.0e-3 (set by Precision) | Stages cut from 4 to 2, the largest count that fits in shared memory (196608 bytes needed, 65536 of 101376 after the cut). Context from the Papiers conversation "Matrix Multiplication — Triton  documentation" |
