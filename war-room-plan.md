# Multi-POV Research War Room (v3, solo build)

## 1. Concept

One researcher (you) plays two perspectives, Speed (Person A) and Precision (Person B), switching between lens tabs in one UI on a single GPU kernel investigation. An autonomous agent proposes falsifiable hypotheses, runs Triton experiments in under 15 seconds, and judges its own results against predictions it stated beforehand. When the two lenses collide, the agent forks into parallel branches, explains the evidence for each in a structured summary, rejects its own failed hypotheses with a post-mortem, and asks the relevant lens owner a specific question when human judgment would help, while it keeps exploring other branches. Findings and context flow through Papiers: the agent reads annotations to seed hypotheses and writes the ledger and summaries back.

Honesty note for the demo and writeup: the system is designed for two humans, and in the demo one person plays both roles by switching tabs. Say so plainly, it is a feature of the prototype, not a flaw.

## 2. Locked decisions

| Decision | Choice |
| --- | --- |
| Domain | GPU kernel tuning, fused matmul + bias + GELU |
| Kernel language | Triton |
| Hardware | RTX 5060 laptop (8GB VRAM), WSL2 on Windows |
| Team | Solo, two lens tabs, you switch |
| LLM | NVIDIA NIM (Nemotron fast model and DeepSeek flash model), final choice after an hour 1 benchmark |
| Papiers integration | Deep: read annotations to form hypotheses, write findings back |
| Help routing | Ask the relevant lens owner (the other tab shows the card) |
| While waiting for an answer | Keep exploring other branches |
| Failure | Agent rejects its own hypothesis from evidence |
| Deadlines | Initial Devpost: Saturday 9pm. Final: Sunday 9:30am |
| Initial submission | Working demo, video, and writeup |
| Plan format | Markdown |

## 3. Hour 0 hardware checkpoint (go/no-go at 2:00am)

The 5060 laptop is viable but the software stack is the risk. Verify in this order and stop at the first failure.

1. In WSL2, confirm `nvidia-smi` shows the GPU.
2. Install a PyTorch build for CUDA 12.8 or newer and print `torch.cuda.get_device_capability()` (expect 12.0). Time a 4096 x 4096 FP16 matmul.
3. Compile and run a trivial Triton kernel (vector add), then a Triton matmul from the official tutorial.
4. Laptop settings: plugged in, high performance power mode, close other GPU apps. Thermal throttling will add noise to timings.
5. Test the NIM API key with one chat completion call (the hosted API is OpenAI compatible, verify the base URL and model IDs in the NVIDIA catalog).

If step 3 is not working by 2:00am, fall back to the quantization and pruning domain and keep the rest of the architecture unchanged. Solo with a hard deadline, do not spend the morning fighting the toolchain.

## 4. Research problem

Target: a fused matmul + bias + GELU Triton kernel, baseline is PyTorch eager (`torch.nn.functional.gelu(x @ w + b)`). Shapes are small enough for 8GB VRAM and sweeps finish in under 15 seconds.

Metrics returned as JSON from every run: `latency_ms` (median of N runs with warmup, CUDA events), `tflops`, `max_error`, `vram_mb`, `compiled_ok`, `correct_ok`.

Starting kernel (iteration 0): a deliberately unoptimized but correct Triton kernel. It is already fused (matmul, bias, and GELU in one kernel), uses FP32 accumulation and exact GELU, and passes `max_error < 1e-3` from the start, but it is poorly configured: tiny tiles, one warp, one pipeline stage, and no `@triton.autotune`. This gives the agent real headroom for early wins, keeps the first hypothesis failure attributable to the hypothesis and not a broken starting point, and leaves parameter ownership with the agent so every change is a visible hypothesis. Do not start from an unfused or scalar loop kernel, since it can blow the 15 second budget and hides the fusion and accumulation trade-off. Three reference points appear on the charts: PyTorch eager as the external reference, the naive Triton kernel as the agent's start, and the best surviving branch. Report speedups against both eager and the naive start, since a large win over a weak baseline is easy to discount. It is fine if the agent never beats eager's cuBLAS backed matmul, the challenge does not require measurable improvement, and the writeup should say so plainly.

Error metric note: FP16 has roughly 1e-3 relative precision, so a bare absolute bound of 1e-3 fails for outputs larger than about 1 even with a perfect kernel. Scale inputs so outputs are O(1), compare against an FP32 reference, and verify that the known-good baseline kernel passes `max_error < 1e-3`. If it does not, switch to relative error.

Speed lens (Person A): maximize TFLOPS and minimize latency.
Precision lens (Person B): enforce `max_error < 1e-3` on the full test batch plus adversarial inputs (large K, large magnitudes).

The agent edits a constrained surface, not free-form kernels: tile sizes, number of warps and stages, accumulation dtype, activation formulation (exact vs approximate GELU), and fusion boundaries inside a known-good kernel template. This keeps LLM written code correct while leaving real room for trade-offs.

## 5. Model choice: hour 1 benchmark (15 minutes max)

Run the same prompt on both NIM models: given the current kernel parameters, a directive, and the last result, return a JSON hypothesis with a prediction and a parameter edit. Run about 10 trials each and record:

- Median latency per call (the loop needs to feel fast).
- Valid JSON rate.
- Rate of edits that compile and pass the correctness check.
- Quality of the written explanation (read two outputs from each).

Default split if both are usable: the fast model for hypothesis and edit generation inside the loop, the stronger model for post-mortems, branch summaries, and the final report, since those run less often and matter most for the demo. Check rate limits on the hosted API and cache responses during rehearsals so dry runs do not burn quota.

## 6. Architecture

```
        Lens tab: Speed (A)          Lens tab: Precision (B)
                  \                        /
                   v                      v
+---------------------------------------------------------------+
|              Control Plane + Agent Orchestrator                |
|  directives, conflict detection, branching, hypothesis ledger  |
+--------+----------------+------------------+------------------+
         |                |                  |
         v                v                  v
+----------------+ +---------------+ +----------------------------+
| Experiment     | | Explanation   | | Help Request Manager       |
| Engine         | | Layer         | | (triggers, routing,        |
| (Triton +      | | (branch       | | blocking state, timeouts)  |
| benchmark)     | | summaries)    | +----------------------------+
+----------------+ +---------------+
         ^                |                  ^
         |                v                  |
         |        +---------------+          |
         +--------| Papiers       |----------+
                  | Context       |
                  | Adapter       |
                  +---------------+
                          |
                          v
              Web UI (DAG, metrics, summaries, help cards)
```

### Component 1: Control Plane and Orchestrator
Ingests directives from the two lens tabs (one web client, a lens switcher, `user_id` still `person_a` or `person_b` so the data model works unchanged for two real people later), tracks research state, detects conflicts, forks branches, and owns the hypothesis ledger.

State objects:
- `ExperimentState`: code version, metrics, branch parent ID, status (`running`, `waiting_on_human`, `rejected`, `merged`, `done`).
- `UserDirective`: `{ user_id, lens, timestamp, constraint_type, threshold, priority }`.
- `BranchTree`: DAG of the main thread and forks.
- `HypothesisRecord`: `{ id, branch_id, claim, prediction, thresholds, status, evidence_refs, verdict_reason }` where status is `proposed`, `confirmed`, `rejected`, or `inconclusive`.
- `BranchSummary`: `{ branch_id, what_was_tried, what_moved, confidence, uncertainties, what_would_change_my_mind, next_step }`.
- `HelpRequest`: `{ id, target_user, question, evidence_refs, options, default_assumption, blocking_branch, deadline, status, answer }`.

### Component 2: Autonomous Experiment Engine
Runs each candidate in an isolated subprocess with a hard 15 second timeout and kill on hang. Captures stdout, stderr, and structured metrics, and writes everything to the evidence log keyed by hypothesis ID.

### Component 3: Explanation Layer
Generates a `BranchSummary` after every completed iteration and a comparative report when branches finish (Section 8).

### Component 4: Help Request Manager
Monitors the agent loop for uncertainty triggers, formats a request with evidence, routes it to the lens owner (a badge on that tab), marks the branch blocked, and ingests the answer (Section 10).

### Component 5: Papiers Context Adapter
Retrieves reading and thinking context before hypothesis generation and writes results back afterward (Section 11).

### Component 6: Web UI
See Section 12. Branches run sequentially and interleave on one GPU, which is enough to show "keeps exploring other branches while waiting" without concurrency bugs.

## 7. Agent loop (one iteration)

The loop starts from the deliberately unoptimized baseline in Section 4 (iteration 0), so the first hypotheses have visible headroom.

1. Retrieve context: active directives, branch history, relevant Papiers notes.
2. Propose a hypothesis with a falsifiable prediction and thresholds stated before running, for example: "Accumulating in FP16 will give at least 1.3x speedup with max_error under 1e-3."
3. Generate a constrained code change (parameters and template regions only).
4. Run the engine and collect metrics.
5. Compare measured results to the prediction in code, not by LLM judgment. The harness sets the verdict: `confirmed`, `rejected`, or `inconclusive` (inside the noise band or near a threshold).
6. Write a branch summary and, for rejections, a post-mortem.
7. Choose the next step from the evidence: refine, pivot, fork, merge, or request help.

## 8. Agent explanation and summary

Every branch has a live summary card, regenerated after each iteration, with fixed fields:

- **What I tried**: one sentence plus a link to the code diff.
- **What moved**: metrics before and after for latency, TFLOPS, max_error, VRAM, shown as deltas.
- **Verdict against my prediction**: predicted vs observed, and the verdict.
- **Confidence**: high, medium, or low, with the reason (run-to-run variance, small sample of adversarial inputs).
- **Uncertainties**: what the agent does not know and could not test.
- **What would change my mind**: the specific result that would flip the conclusion.
- **Next step and why**: the chosen action tied to the evidence.

When branches finish, the agent produces a Comparative Divergence Report: a side-by-side table with PyTorch eager and the naive start as reference rows, a latency vs max_error trade-off curve across the parameter sweep, and a three sentence plain-language recommendation. All claims link to logged runs.

## 9. Failure by design: the agent rejects its own hypothesis

The failure is produced by the rules of the loop, not staged by a human.

1. The Speed directive pushes the agent toward aggressive options. The agent proposes H1: use FP16 accumulation (or a low-precision approximate GELU) for a speedup, with the prediction and thresholds written down before the run.
2. The harness runs it on the standard batch and the adversarial inputs. With large K, low-precision accumulation drives `max_error` above 1e-3 while latency improves.
3. The programmatic verdict marks H1 `rejected` because the prediction was violated. No human tells the agent this.
4. The agent writes a post-mortem: expected vs observed, the likely cause (accumulation error growing with K, backed by an error-vs-K sweep it runs itself), and the pivot (accumulate in FP32, keep the other fusion gains, try a better tile configuration).
5. The rejected hypothesis stays visible in the DAG and ledger with its evidence, and is written back to Papiers as a negative result.

Reliability: reproduce this path at least three times in rehearsal and keep a recorded run as backup. If the agent avoids the trap, the harness still tests a lower-precision variant in the sweep and reports it as rejected with the same post-mortem.

## 10. Asking for help

### Triggers
- **Unresolvable lens conflict**: the evidence shows a real trade-off and no option satisfies both directives.
- **Near-threshold result**: a metric falls inside the noise band, so the verdict is `inconclusive`.
- **Ambiguous requirement**: for example whether the error bound applies to max, mean, or per-row relative error, or whether approximate GELU is acceptable.
- **Repeated failure**: the same hypothesis family fails twice.

### Routing
Questions about speed, hardware, or measurement go to the Speed lens. Questions about error bounds, numerics, or acceptable approximations go to the Precision lens. The card appears in that lens tab with a notification badge, and you switch hats to answer.

### Request format
Each card contains a specific question (one sentence), two or three evidence items with links to runs, two or three answer options with the agent's default marked, what the answer would change, and a deadline.

### While waiting
The blocked branch shows `waiting on Precision lens` in the DAG. The agent continues the least blocked branch and logs what it did. If the deadline passes, it proceeds with the default assumption and flags downstream results as `assumption pending`, then revises when the answer arrives.

### Ingesting the answer
The answer becomes a `UserDirective` or annotation. The agent states explicitly how it changes the plan (for example: "Because the Precision lens accepts per-row relative error, I am reopening the approximate GELU branch") and records the human contribution in the ledger and in Papiers.

## 11. Papiers deep integration

Verify exact CLI commands in the docs at papiers.ai/docs during the hour 0 checkpoint. The intended flow:

- **Before the hackathon work (if the rules allow)**: load a small reading set into the library with highlights and short notes, for example the Triton paper, FlashAttention, mixed precision training, and a few notes on GELU approximations and accumulation error.
- **Before each hypothesis**: the adapter retrieves notes relevant to the current directive and branch, includes them in the agent prompt, and cites them in the hypothesis ("seeded from your note on FP16 accumulation").
- **After each iteration or branch completion**: write the branch summary, ledger entries (including rejections and post-mortems), and human answers back to Papiers.
- **Demo moment**: show one hypothesis visibly traced to your own annotation.

Fallback: if the CLI is awkward, keep a local log as the source of truth and sync to Papiers on top. Downgrade deep to light (write back only) at the 2:00am checkpoint if you are behind.

## 12. UI (build in this priority order)

1. **Branch DAG** with status badges (running, waiting on human, rejected, merged) and the rejected branch clearly visible.
2. **Live metric chart**: latency and max_error per iteration, per branch.
3. **Branch summary card** (Section 8) with expandable evidence.
4. **Help request cards** with a badge on the relevant lens tab, with answer buttons.
5. **Hypothesis ledger** table with verdicts.
6. **Conflict banner** with the Merge and Hybridize action.

Suggested stack for a solo build: small FastAPI backend with WebSocket updates and a single page front end. Keep it plain, no heavy framework setup.

## 13. Scope tiers

**Tier 1, must work by Saturday 9pm (initial submission):**
- Benchmark harness and constrained Triton edit surface.
- Autonomous loop with falsifiable predictions and programmatic verdicts.
- Two lens tabs, directives, conflict detection, and one fork.
- Reproducible self rejection with post-mortem.
- Branch summary cards.
- One help request that blocks a branch while the agent explores another, and the answer changing the plan.
- Papiers read before a hypothesis and write back at the end.
- UI items 1 to 4.
- Demo video, writeup, README, screenshots.

**Tier 2, for Sunday 9:30am (final):**
- Hybrid merge run and conflict banner.
- Hypothesis ledger table.
- Comparative Divergence Report with the trade-off curve.
- More hypotheses per branch, better prompts, cleaner video.
- Deeper Papiers use if time allows.

Cut order if behind: Papiers deep to light, then ledger table, then conflict banner, then hybrid merge. Never cut the failure path, the summary cards, or the help request, since those are what the judging criteria reward.

## 14. Schedule

Times are America/Toronto. This assumes you sleep, adjust to your own plan.

**Block 1: Saturday 12:30am to 3:30am**
- 12:30 to 1:15: hardware checkpoint (Section 3), Papiers docs, NIM key test.
- 1:15 to 2:00: benchmark harness, FP32 reference, baseline kernel, error bound check, 15 minute model benchmark (Section 5). Go/no-go on Triton and on deep Papiers at 2:00.
- 2:00 to 3:30: minimal agent loop, one autonomous iteration end to end with a verdict.
- 3:30: stop and sleep. Save your working state (you decide when to commit or push).

**Sleep: 3:30am to 10:30am**

**Block 2: Saturday 10:30am to 5:30pm**
- 10:30 to 12:00: lens tabs, directives, conflict detection, fork, failure path (reproduce it).
- 12:00 to 1:30: summary generator and help request manager with wait and explore.
- 1:30 to 2:30: Papiers read and write.
- 2:30 to 4:00: UI items 1 to 4.
- 4:00 to 5:30: three dry runs, fix bugs. Feature freeze at 5:30pm.

**Block 3: Saturday 5:30pm to 8:30pm**
- 5:30 to 6:30: record the demo video, two or three takes.
- 6:30 to 7:30: Devpost writeup, screenshots, README, repo cleanup.
- 7:30 to 8:30: buffer. Submit by 8:30pm, not 8:59.

**Block 4: Saturday 9:30pm to Sunday 9:30am**
- 9:30pm to 1:30am: Tier 2 features and polish, keep the Tier 1 build working at all times (work on a copy or branch).
- 1:30am to 7:30am: sleep.
- 7:30 to 9:00am: re-record video if needed, update writeup, final submission by 9:00am, not 9:30.

## 15. Devpost deliverables checklist

Check the Devpost page for the exact required fields and video length limits tonight, since I do not have them. Likely needs:

- Project name, tagline, and short description.
- Writeup covering the problem, how it works, what the agent investigated, the failed hypothesis, how human judgment changed the course, and what you learned.
- Demo video showing a complete run (see Section 16).
- Public repo link with a README (setup steps, how to run, architecture diagram).
- Screenshots of the DAG, summary card, help request, and ledger.
- Papiers usage described explicitly.
- Built with list: Triton, PyTorch, NVIDIA NIM, FastAPI, Papiers.

## 16. Demo script (3 minutes, record as a screen capture with voiceover)

Every number on screen comes from a real run. Do not quote results in the narration before they exist, read them off the screen. Open with one line: "I'm playing both researchers here, switching lens tabs, in a real team these would be two people."

| Time | Scene | What happens | Script highlight |
| --- | --- | --- | --- |
| 0:00 to 0:25 | Baseline and context | Baseline benchmark runs. Agent pulls a note from Papiers and states hypothesis H1 with a prediction. | "The agent starts from what I've read and commits to a prediction before it runs anything." |
| 0:25 to 1:00 | Speed lens | You submit a throughput directive in the Speed tab. Agent runs H1 (low precision accumulation) and latency drops. | "The speed lens pushes for throughput and the agent finds real gains." |
| 1:00 to 1:35 | Self rejection | Switch to the Precision tab and attach the error bound. The harness shows max_error over bound, H1 flips to rejected, the agent writes its post-mortem and forks the tree. | "Nobody told it it was wrong. It checked its own prediction, rejected its hypothesis, and explained why." |
| 1:35 to 2:10 | Asking for help | On the precision branch a result lands near the threshold. A card appears with a badge on the Speed or Precision tab depending on the question. The agent keeps exploring the other branch. You switch tabs and answer. The agent states how the plan changes. | "It knew what it didn't know, asked the right lens, and didn't sit idle." |
| 2:10 to 2:40 | Evidence and trade-off | Side by side branch summaries with confidence and "what would change my mind", plus the latency vs error curve. | "Every claim links back to a logged run." |
| 2:40 to 3:00 | Consensus | Approve hybrid run (Tier 2, otherwise show the best surviving branch), agent executes. The ledger shows confirmed, rejected, and inconclusive entries and the write back to Papiers. | "Human judgment steered the search, and the failed branch is part of the result." |

For the Saturday video, if the hybrid merge is not done, end on the best surviving branch and the ledger. Re-record with the hybrid for the Sunday final.

## 17. Judging criteria map

| Criterion | Where it is shown |
| --- | --- |
| Autonomous research | Falsifiable hypotheses, programmatic verdicts, agent chosen next steps, self rejection (Sections 7 and 9) |
| Human to agent collaboration | Lens directives, help requests with evidence, answers that change the plan, Papiers notes seeding hypotheses (Sections 10 and 11) |
| Technical execution | Constrained Triton edit surface, timeouts, reproducible failure path, three dry runs |
| Originality | Multi perspective forking plus an agent that asks lens owners targeted questions and keeps working while it waits |
| Evidence and demonstration | Summary cards, ledger, trade-off curve, links to raw runs, backup video (Sections 8 and 16) |

## 18. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| Triton or PyTorch fails on Blackwell under WSL2 | Hour 0 checkpoint, fallback to quantization domain at 2:00am |
| LLM writes broken kernels | Constrained edit surface on a known-good template, auto revert on failure |
| NIM rate limits or slow responses | Hour 1 benchmark, response caching in rehearsals, fast model for the loop |
| Noisy timings and thermal throttling | Warmup, median of N, plugged in, high performance mode, noise band in verdicts |
| FP16 error bound unreachable by design | Baseline check at hour 1, normalize inputs or switch to relative error |
| Failure does not reproduce live | Reproduce three times in rehearsal, harness fallback variant, recorded backup |
| Help request unanswered in the demo | Deadline with default assumption, rehearse the tab switch |
| Hung experiment | 15 second hard timeout and process kill |
| Solo burnout and scope creep | Sleep blocks, Tier 1 and Tier 2 split, feature freeze at 5:30pm, submit 30 minutes early |
| Breaking the working build during Tier 2 | Keep a frozen Tier 1 copy, work on a separate copy or branch |
| Papiers CLI behaves differently than expected | Verify at hour 0, local log as source of truth, downgrade to light |

## 19. Assumptions to confirm

- UI stack (suggested default: FastAPI backend with WebSocket and a plain single page front end).
- Final model split after the hour 1 benchmark, and the exact NIM model IDs available to you.
- Whether the fused matmul + bias + GELU op is the final target.
- Exact Devpost requirements (video length, required fields, public repo rule).
- Whether you are comfortable with the sleep blocks as scheduled.
