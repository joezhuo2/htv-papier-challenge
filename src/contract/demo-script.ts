import type { Branch, Check, DemoEvent, Delta, Hypothesis, Metrics, Params, PostMortem, Report, Run, Summary } from './types'

const p = (block_m: number, block_n: number, block_k: number, warps: number, stages: number, acc: Params['acc'], gelu: Params['gelu']): Params =>
  ({ block_m, block_n, block_k, warps, stages, acc, gelu })

const m = (latency_ms: number, tflops: number, max_error: number, adv_error: number, vram_mb: number): Metrics =>
  ({ latency_ms, tflops, max_error, adv_error, vram_mb, compiled_ok: true, correct_ok: true })

const d = (metric: string, before: number, after: number, unit = ''): Delta => ({ metric, before, after, unit })
const c = (name: string, predicted: string, observed: string, pass: boolean | null): Check => ({ name, predicted, observed, pass })

const eager = m(6.9, 19.9, 2.9e-4, 3.8e-4, 301)
const naive = m(41.0, 3.35, 3.1e-4, 4.2e-4, 238)

const branch = (id: string, parents: string[], label: string, lens: Branch['lens']): Branch =>
  ({ id, parents, label, lens, status: 'running' })

const main = branch('main', [], 'Baseline tuning', 'both')
const speed = branch('speed', ['main'], 'Speed: FP16 accumulate', 'speed')
const prec = branch('prec', ['main'], 'Precision: strict bound', 'precision')
const fast = branch('fast', ['speed'], 'Speed: safe accumulate', 'speed')
const merge = branch('merge', ['prec', 'fast'], 'Hybrid', 'both')

const h = (id: string, branch: string, claim: string, min_speedup: number, vs: 'naive' | 'parent_best', params: Params, note?: string): Hypothesis =>
  ({ id, branch, claim, prediction: { min_speedup, vs, max_error: 1e-3 }, params, note })

const H0 = h('H0', 'main', 'Bigger tiles with 4 warps and 3 pipeline stages will recover most of the throughput the naive configuration leaves unused.', 2.0, 'naive', p(64, 64, 32, 4, 3, 'fp32', 'exact'), 'Seeded from Papiers note n1')
const H1 = h('H1', 'speed', 'Accumulating in FP16 with 128x128 tiles will cut latency further while staying under the error bound.', 1.3, 'parent_best', p(128, 128, 32, 4, 3, 'fp16', 'exact'), 'Seeded from Papiers note n2')
const H2 = h('H2', 'fast', 'The gain from 128x128 tiles survives when accumulation returns to FP32.', 1.25, 'parent_best', p(128, 128, 32, 4, 3, 'fp32', 'exact'))
const H3 = h('H3', 'prec', 'A deeper pipeline with a 128x64 tile gives a modest, safe speedup with exact GELU.', 1.05, 'parent_best', p(128, 64, 32, 4, 4, 'fp32', 'exact'))
const H4 = h('H4', 'fast', 'Approximate GELU on the FP32 accumulation config removes most of the epilogue cost and stays inside the adversarial bound.', 1.3, 'parent_best', p(128, 128, 32, 4, 3, 'fp32', 'approx'), 'Suggested by the Speed lens in answer to hq1')
const H5 = h('H5', 'merge', 'Approximate GELU with the precision branch pipeline depth gives a small further gain with no extra error.', 1.03, 'parent_best', p(128, 128, 32, 4, 4, 'fp32', 'approx'))

const run = (id: string, hyp: Hypothesis, metrics: Metrics): Run => ({ id, branch: hyp.branch, hypothesis: hyp.id, params: hyp.params, metrics })

const r1 = run('r1', H0, m(9.6, 14.3, 3.3e-4, 4.4e-4, 252))
const r2 = run('r2', H1, m(5.4, 25.5, 7.4e-4, 3.2e-2, 252))
const r3 = run('r3', H2, m(7.9, 17.4, 3.4e-4, 4.9e-4, 252))
const r4 = run('r4', H3, m(8.6, 16.0, 3.3e-4, 4.5e-4, 248))
const r5 = run('r5', H4, m(6.6, 20.8, 5.6e-4, 8.1e-4, 252))
const r6 = run('r6', H5, m(6.1, 22.5, 5.7e-4, 8.3e-4, 252))

const postMortem: PostMortem = {
  hypothesis: 'H1',
  expected: 'Speedup of at least 1.30x with max error under 1e-3 on every batch.',
  observed: 'Speedup of 1.78x and standard error 7.4e-4, but adversarial error 3.2e-2, 32x over the bound.',
  cause: 'FP16 accumulation error grows roughly with the square root of K. The sweep exceeds the bound at every K from 256 to 4096, and the standard batch hid it because its values are small.',
  pivot: 'Keep the 128x128 tiles and fusion, return to FP32 accumulation, and test whether the speedup survives.',
  sweep: [
    { k: 256, adv_error: 8.0e-3 },
    { k: 512, adv_error: 1.1e-2 },
    { k: 1024, adv_error: 1.6e-2 },
    { k: 2048, adv_error: 2.3e-2 },
    { k: 4096, adv_error: 3.2e-2 },
  ],
}

const summaryMain: Summary = {
  branch: 'main',
  tried: 'Tuned tile sizes, warps, and stages (64x64x32 tiles, 4 warps, 3 stages).',
  moved: [d('latency', 41.0, 9.6, 'ms'), d('throughput', 3.35, 14.3, 'TFLOPS'), d('max error', 3.1e-4, 3.3e-4), d('adversarial error', 4.2e-4, 4.4e-4)],
  verdict: 'Confirmed: 4.27x over naive, predicted at least 2.0x, error 3.3e-4.',
  confidence: 'high',
  confidence_reason: 'Run to run variance was under 1% across the timing repeats.',
  uncertainties: ['Adversarial inputs are measured but no lens gates them yet', 'Still 1.39x slower than PyTorch eager'],
  would_change_mind: 'A rerun showing more than 5% variance would reopen this result.',
  next_step: 'Wait for a lens directive, otherwise try larger tiles.',
}

const summarySpeedProvisional: Summary = {
  branch: 'speed',
  tried: '128x128x32 tiles with FP16 accumulation, seeded from Papiers note n2.',
  moved: [d('latency', 9.6, 5.4, 'ms'), d('throughput', 14.3, 25.5, 'TFLOPS'), d('max error', 3.3e-4, 7.4e-4), d('adversarial error', 4.4e-4, 3.2e-2)],
  verdict: 'Provisionally confirmed: 1.78x over the best so far, predicted at least 1.3x, standard error under the bound.',
  confidence: 'low',
  confidence_reason: 'Adversarial error rose by roughly 70x and no precision constraint is attached to gate it.',
  uncertainties: ['Why the adversarial error jumped', 'Whether the error grows with K'],
  would_change_mind: 'Any precision constraint on the adversarial batch would reject this result.',
  next_step: 'Hold and flag the adversarial error until a precision lens is attached.',
}

const summarySpeedFinal: Summary = {
  branch: 'speed',
  tried: '128x128x32 tiles with FP16 accumulation, seeded from Papiers note n2.',
  moved: summarySpeedProvisional.moved,
  verdict: 'Rejected: adversarial error 3.2e-2 against a 1e-3 bound.',
  confidence: 'high',
  confidence_reason: 'The K sweep reproduces the error at every size tested.',
  uncertainties: ['Whether a mixed accumulation scheme could keep part of the gain'],
  would_change_mind: 'An FP16 accumulation variant that holds adversarial error under 1e-3 across the K sweep.',
  next_step: 'Fork: keep the 128x128 tiles and return to FP32 accumulation.',
}

const summaryFastH2: Summary = {
  branch: 'fast',
  tried: '128x128x32 tiles with FP32 accumulation and exact GELU.',
  moved: [d('latency', 9.6, 7.9, 'ms'), d('throughput', 14.3, 17.4, 'TFLOPS'), d('max error', 3.3e-4, 3.4e-4), d('adversarial error', 4.4e-4, 4.9e-4)],
  verdict: 'Inconclusive: 1.22x against a predicted 1.25x, inside the noise band.',
  confidence: 'medium',
  confidence_reason: 'Run to run noise is about 4% on this configuration.',
  uncertainties: ['Whether more tile tuning would clear 1.25x', 'Whether the remaining cost sits in the matmul or the GELU epilogue'],
  would_change_mind: 'A profile showing the epilogue under 5% of kernel time would favor more tile search.',
  next_step: 'Ask the Speed lens whether to keep tuning or change approach.',
}

const summaryPrec: Summary = {
  branch: 'prec',
  tried: '128x64x32 tiles, 4 pipeline stages, FP32 accumulation, exact GELU.',
  moved: [d('latency', 9.6, 8.6, 'ms'), d('throughput', 14.3, 16.0, 'TFLOPS'), d('max error', 3.3e-4, 3.3e-4), d('adversarial error', 4.4e-4, 4.5e-4)],
  verdict: 'Confirmed: 1.12x over the best so far, predicted at least 1.05x.',
  confidence: 'high',
  confidence_reason: 'Latency and error were stable across the timing repeats.',
  uncertainties: ['The gain is modest because the strict bound limits the available options'],
  would_change_mind: 'A config in the same family that is faster with the same error would replace it.',
  next_step: 'Keep as the safe reference branch while the Speed lens answers.',
}

const summaryFastH4: Summary = {
  branch: 'fast',
  tried: 'Approximate GELU on the 128x128x32 FP32 accumulation config, suggested by the Speed lens.',
  moved: [d('latency', 9.6, 6.6, 'ms'), d('throughput', 14.3, 20.8, 'TFLOPS'), d('max error', 3.3e-4, 5.6e-4), d('adversarial error', 4.4e-4, 8.1e-4)],
  verdict: 'Confirmed: 1.45x over the best so far, predicted at least 1.3x, adversarial error 8.1e-4 inside the 1e-3 bound.',
  confidence: 'medium',
  confidence_reason: 'The adversarial margin is only 19% and rests on one adversarial set.',
  uncertainties: ['Thin margin to the bound', 'Not checked on other input distributions'],
  would_change_mind: 'A second adversarial set with error above 1e-3 would reject the approximate GELU.',
  next_step: 'Propose a hybrid with the precision branch pipeline depth.',
}

const summaryMerge: Summary = {
  branch: 'merge',
  tried: 'Approximate GELU with 4 pipeline stages on 128x128x32 tiles.',
  moved: [d('latency', 6.6, 6.1, 'ms'), d('throughput', 20.8, 22.5, 'TFLOPS'), d('max error', 5.6e-4, 5.7e-4), d('adversarial error', 8.1e-4, 8.3e-4)],
  verdict: 'Confirmed: 1.08x over the best so far, predicted at least 1.03x.',
  confidence: 'medium',
  confidence_reason: 'The gain is small and the adversarial margin is 17%.',
  uncertainties: ['Beats eager by 1.13x on this shape only', 'Untested on other matrix shapes'],
  would_change_mind: 'A rerun on other shapes that regresses against eager.',
  next_step: 'Export the results for Papiers import and stop, the remaining gain is within noise.',
}

const report: Report = {
  points: [
    { run: 'eager', label: 'PyTorch eager', branch: '-', latency_ms: 6.9, adv_error: 3.8e-4, reference: true },
    { run: 'naive', label: 'Naive Triton', branch: '-', latency_ms: 41.0, adv_error: 4.2e-4, reference: true },
    { run: 'r1', label: 'H0 tuned tiles', branch: 'main', latency_ms: 9.6, adv_error: 4.4e-4 },
    { run: 'r2', label: 'H1 FP16 accumulate', branch: 'speed', latency_ms: 5.4, adv_error: 3.2e-2 },
    { run: 'r3', label: 'H2 FP32 accumulate', branch: 'fast', latency_ms: 7.9, adv_error: 4.9e-4 },
    { run: 'r4', label: 'H3 deeper pipeline', branch: 'prec', latency_ms: 8.6, adv_error: 4.5e-4 },
    { run: 'r5', label: 'H4 approximate GELU', branch: 'fast', latency_ms: 6.6, adv_error: 8.1e-4 },
  ],
  pareto: ['eager', 'r5'],
  recommendation: 'Approximate GELU with FP32 accumulation is the only agent found configuration that beats PyTorch eager while staying inside the 1e-3 adversarial bound, with a 19% margin. The FP16 accumulation shortcut is faster but exceeds the bound by 32x and should be dropped. The next experiment merges the fast branch GELU choice with the precision branch pipeline depth.',
  best: { run: 'r5', vs_naive: 6.21, vs_eager: 1.05 },
}

const raw: DemoEvent[] = [
  { t: 0, type: 'session_started', baselines: { eager, naive }, branch: main },
  { t: 1500, type: 'agent_log', text: 'Baselines measured: PyTorch eager and the naive Triton kernel both pass the correctness gate. Starting from the naive configuration.' },
  { t: 4000, type: 'papiers_context', note: { id: 'n1', title: 'Triton matmul tiling notes', excerpt: 'Small output tiles waste tensor core throughput. Try 64 by 64 or larger with 4 warps and a few pipeline stages.' } },
  { t: 6500, type: 'hypothesis_proposed', hypothesis: H0 },
  { t: 7500, type: 'run_started', run: 'r1', branch: 'main', hypothesis: 'H0' },
  { t: 10500, type: 'run_finished', run: r1 },
  { t: 11200, type: 'verdict', hypothesis: 'H0', verdict: 'confirmed', provisional: false, reason: 'Prediction holds.', checks: [c('speedup vs naive', '>= 2.00x', '4.27x', true), c('standard max error', '< 1e-3', '3.3e-4', true)] },
  { t: 12000, type: 'summary', summary: summaryMain },
  { t: 12500, type: 'branch_status', branch: 'main', status: 'done' },
  { t: 13500, type: 'agent_log', text: 'H0 confirmed. Tuned tiles are 4.27x faster than naive but 1.39x slower than eager. Waiting for a lens directive.' },

  { t: 25000, type: 'directive_added', directive: { id: 'd1', lens: 'speed', text: 'Push throughput harder than tile tuning alone.', constraint: 'maximize_speed' } },
  { t: 26000, type: 'branch_created', branch: speed },
  { t: 27000, type: 'papiers_context', note: { id: 'n2', title: 'Mixed precision highlights', excerpt: 'On GeForce parts, FP16 accumulate can roughly double tensor core throughput compared with FP32 accumulate.' } },
  { t: 29000, type: 'hypothesis_proposed', hypothesis: H1 },
  { t: 30500, type: 'run_started', run: 'r2', branch: 'speed', hypothesis: 'H1' },
  { t: 36000, type: 'run_finished', run: r2 },
  { t: 37000, type: 'verdict', hypothesis: 'H1', verdict: 'confirmed', provisional: true, reason: 'Speed and standard error predictions hold. Adversarial error is measured but not gated by any lens yet.', checks: [c('speedup vs best', '>= 1.30x', '1.78x', true), c('standard max error', '< 1e-3', '7.4e-4', true)] },
  { t: 38500, type: 'summary', summary: summarySpeedProvisional },
  { t: 41000, type: 'agent_log', text: 'H1 is fast, but the adversarial error rose by roughly 70x. No precision constraint is attached, so the verdict is provisional.' },

  { t: 60000, type: 'directive_added', directive: { id: 'd2', lens: 'precision', text: 'Enforce max error under 1e-3 on the standard batch and the adversarial batch.', constraint: 'max_error', threshold: 1e-3 } },
  { t: 61000, type: 'agent_log', text: 'Precision constraint attached. Re-evaluating H1 against the logged run, no new run needed.' },
  { t: 63000, type: 'verdict', hypothesis: 'H1', verdict: 'rejected', provisional: false, reason: 'Adversarial error 3.2e-2 violates the 1e-3 bound added by the Precision lens.', checks: [c('speedup vs best', '>= 1.30x', '1.78x', true), c('standard max error', '< 1e-3', '7.4e-4', true), c('adversarial max error', '< 1e-3', '3.2e-2', false)] },
  { t: 64000, type: 'branch_status', branch: 'speed', status: 'rejected' },
  { t: 65500, type: 'conflict_detected', directives: ['d1', 'd2'], reason: 'The best candidate for the Speed directive (H1) violates the Precision constraint on the adversarial batch. Forking instead of overwriting either directive.' },
  { t: 67000, type: 'agent_log', text: 'H1 rejected. Running an error vs K sweep on the H1 kernel to find the cause.' },
  { t: 74000, type: 'post_mortem', post_mortem: postMortem },
  { t: 76000, type: 'summary', summary: summarySpeedFinal },
  { t: 78000, type: 'findings_exported', items: [{ kind: 'negative_result', title: 'H1 rejected: FP16 accumulation exceeds the adversarial bound', file: 'exports/H1-negative-result.md' }, { kind: 'summary', title: 'Branch summary: speed', file: 'exports/summary-speed.md' }] },
  { t: 82000, type: 'branch_created', branch: prec, reason: 'Precision lens: find the fastest configuration inside the strict bound.' },
  { t: 83500, type: 'branch_created', branch: fast, reason: 'Pivot from the H1 post-mortem: keep the larger tiles and return to FP32 accumulation.' },
  { t: 85500, type: 'hypothesis_proposed', hypothesis: H2 },
  { t: 87000, type: 'run_started', run: 'r3', branch: 'fast', hypothesis: 'H2' },
  { t: 91000, type: 'run_finished', run: r3 },
  { t: 92500, type: 'verdict', hypothesis: 'H2', verdict: 'inconclusive', provisional: false, reason: 'Speedup is inside the noise band around the predicted threshold.', checks: [c('speedup vs best', '>= 1.25x', '1.22x (noise +/- 0.05x)', null), c('standard max error', '< 1e-3', '3.4e-4', true), c('adversarial max error', '< 1e-3', '4.9e-4', true)] },

  { t: 95000, type: 'summary', summary: summaryFastH2 },
  {
    t: 97000,
    type: 'help_requested',
    help: {
      id: 'hq1',
      target: 'speed',
      question: 'The safe FP32 branch reaches 1.22x, just under my 1.25x prediction and inside the noise band. Should I keep tuning tiles or change approach?',
      evidence: [
        'r3: latency 7.9 ms, 1.22x over the best so far, noise about +/- 0.05x',
        'r2: FP16 accumulation reached 1.78x but failed the adversarial bound',
        'r1: tile tuning alone gave 4.27x over naive, then the gains flattened',
      ],
      options: [
        { id: 'tile_search', label: 'Keep tuning tiles and warps' },
        { id: 'accept', label: 'Accept 1.22x and move on' },
        { id: 'other', label: 'Suggest something else' },
      ],
      default: 'tile_search',
      blocks: 'fast',
      deadline_s: 45,
      what_changes: 'Tile search spends about 8 more iterations on this branch, accepting stops it here, and a different idea redirects it.',
    },
  },
  { t: 98000, type: 'branch_status', branch: 'fast', status: 'waiting_on_human' },
  { t: 100000, type: 'agent_log', text: 'Branch fast is blocked on the Speed lens. Continuing on the precision branch.' },
  { t: 101500, type: 'hypothesis_proposed', hypothesis: H3 },
  { t: 103000, type: 'run_started', run: 'r4', branch: 'prec', hypothesis: 'H3' },
  { t: 107000, type: 'run_finished', run: r4 },
  { t: 108500, type: 'verdict', hypothesis: 'H3', verdict: 'confirmed', provisional: false, reason: 'Prediction holds.', checks: [c('speedup vs best', '>= 1.05x', '1.12x', true), c('standard max error', '< 1e-3', '3.3e-4', true), c('adversarial max error', '< 1e-3', '4.5e-4', true)] },
  { t: 110000, type: 'summary', summary: summaryPrec },
  { t: 114000, type: 'help_answered', help: 'hq1', option: 'other', note: 'Try the tanh approximate GELU. The erf epilogue is probably the bottleneck here.', by: 'speed' },
  { t: 116000, type: 'plan_change', help: 'hq1', text: 'Because the Speed lens suspects the erf epilogue is the bottleneck, I am dropping tile search and testing approximate GELU on the fast branch. The approximation has to clear the Precision lens bound, so I will gate it on adversarial error.' },
  { t: 117000, type: 'branch_status', branch: 'fast', status: 'running' },
  { t: 118500, type: 'hypothesis_proposed', hypothesis: H4 },
  { t: 120000, type: 'run_started', run: 'r5', branch: 'fast', hypothesis: 'H4' },
  { t: 124000, type: 'run_finished', run: r5 },
  { t: 125500, type: 'verdict', hypothesis: 'H4', verdict: 'confirmed', provisional: false, reason: 'Prediction holds and adversarial error stays inside the bound.', checks: [c('speedup vs best', '>= 1.30x', '1.45x', true), c('standard max error', '< 1e-3', '5.6e-4', true), c('adversarial max error', '< 1e-3', '8.1e-4', true)] },
  { t: 127000, type: 'summary', summary: summaryFastH4 },
  { t: 129000, type: 'findings_exported', items: [{ kind: 'answer', title: 'Speed lens suggestion: approximate GELU', file: 'exports/hq1-answer.md' }, { kind: 'summary', title: 'Branch summary: fast', file: 'exports/summary-fast.md' }, { kind: 'summary', title: 'Branch summary: prec', file: 'exports/summary-prec.md' }] },

  { t: 131000, type: 'agent_log', text: 'Both live branches have enough evidence. Building the comparative report.' },
  { t: 136000, type: 'report_ready', report },
  { t: 152000, type: 'hybrid_proposed', branches: ['prec', 'fast'], text: 'Combine the fast branch approximate GELU with the precision branch pipeline depth. Expected gain is small, so the result will tell us whether the branches are additive.' },

  { t: 160500, type: 'branch_created', branch: merge, reason: 'Hybrid approved: approximate GELU with 4 pipeline stages.' },
  { t: 161500, type: 'hypothesis_proposed', hypothesis: H5 },
  { t: 163000, type: 'run_started', run: 'r6', branch: 'merge', hypothesis: 'H5' },
  { t: 167000, type: 'run_finished', run: r6 },
  { t: 168500, type: 'verdict', hypothesis: 'H5', verdict: 'confirmed', provisional: false, reason: 'Prediction holds and adversarial error stays inside the bound.', checks: [c('speedup vs best', '>= 1.03x', '1.08x', true), c('standard max error', '< 1e-3', '5.7e-4', true), c('adversarial max error', '< 1e-3', '8.3e-4', true)] },
  { t: 169500, type: 'branch_status', branch: 'prec', status: 'merged' },
  { t: 169800, type: 'branch_status', branch: 'fast', status: 'merged' },
  { t: 170200, type: 'branch_status', branch: 'merge', status: 'done' },
  { t: 171000, type: 'summary', summary: summaryMerge },
  { t: 173500, type: 'findings_exported', items: [{ kind: 'ledger', title: 'Hypothesis ledger H0 to H5', file: 'exports/ledger.md' }, { kind: 'summary', title: 'Branch summary: merge', file: 'exports/summary-merge.md' }] },
  {
    t: 176000,
    type: 'session_finished',
    ledger: [
      { hypothesis: 'H0', verdict: 'confirmed', history: ['confirmed'] },
      { hypothesis: 'H1', verdict: 'rejected', history: ['confirmed', 'rejected'] },
      { hypothesis: 'H2', verdict: 'inconclusive', history: ['inconclusive'] },
      { hypothesis: 'H3', verdict: 'confirmed', history: ['confirmed'] },
      { hypothesis: 'H4', verdict: 'confirmed', history: ['confirmed'] },
      { hypothesis: 'H5', verdict: 'confirmed', history: ['confirmed'] },
    ],
    best: { run: 'r6', vs_naive: 6.72, vs_eager: 1.13 },
  },
]

export const demoScript: DemoEvent[] = raw.map(e => ({ ...e, mock: true }))
export const DEMO_DURATION_MS = 180000
