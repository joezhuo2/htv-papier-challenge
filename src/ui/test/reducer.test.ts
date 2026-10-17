import { describe, expect, it } from 'vitest'
import { existsSync, readdirSync, readFileSync, statSync } from 'node:fs'
import { join } from 'node:path'
import { demoScript } from '@contract/demo-script'
import { fold, reduce, initialView } from '../src/state/reducer'
import { ledgerRows, predictedLatency, parentBest } from '../src/state/derive'
import { parseLog } from '../src/source'

describe('reducer on demoScript', () => {
  const v = fold(demoScript)

  it('yields five branches with the right statuses', () => {
    expect(Object.fromEntries(v.branches.map(b => [b.id, b.status]))).toEqual({
      main: 'done', speed: 'rejected', prec: 'merged', fast: 'merged', merge: 'done',
    })
  })

  it('yields six ledger entries and best run r6', () => {
    expect(v.ledger).toHaveLength(6)
    expect(ledgerRows(v).map(r => r.verdict)).toEqual(['confirmed', 'rejected', 'inconclusive', 'confirmed', 'confirmed', 'confirmed'])
    expect(v.best?.run).toBe('r6')
    expect(v.finished).toBe(true)
  })

  it('folds runs, verdict history, post-mortem and help', () => {
    expect(v.runs.map(r => r.id)).toEqual(['r1', 'r2', 'r3', 'r4', 'r5', 'r6'])
    expect(v.runs.every(r => r.metrics)).toBe(true)
    expect(v.history.H1).toEqual(['confirmed', 'rejected'])
    expect(v.postMortems[0].hypothesis).toBe('H1')
    expect(v.helps[0].answer?.option).toBe('other')
    expect(v.helps[0].expired).toBe(false)
    expect(v.mock).toBe(true)
  })

  it('links directives and the conflict to branches', () => {
    expect(v.directives.map(d => d.affected)).toEqual(['speed', 'speed'])
    expect(v.conflicts[0]).toMatchObject({ branch: 'speed', forked: 'prec' })
  })

  it('snapshots the provisional verdict on the first H1 summary', () => {
    const h1 = v.feed.filter(f => f.kind === 'summary' && f.hypothesis === 'H1')
    expect(h1.map(f => f.kind === 'summary' && f.verdict?.provisional)).toEqual([true, false])
  })

  it('predicts against the right parent best', () => {
    const pred = Object.fromEntries(v.hypotheses.map(h => [h.id, predictedLatency(v, h)!.toFixed(2)]))
    expect(pred).toEqual({ H0: '20.50', H1: '7.38', H2: '7.68', H3: '9.14', H4: '7.38', H5: '6.41' })
    expect(parentBest(v, 'merge', 1e9)).toBe(6.1)
  })

  it('a new session_started resets the state', () => {
    const again = reduce(v, demoScript[0])
    expect(again.branches).toHaveLength(1)
    expect(again.feed).toHaveLength(0)
  })

  it('expires an unanswered help request and tags later runs on the blocked branch', () => {
    const cut = demoScript.findIndex(e => e.type === 'help_answered')
    let s = fold(demoScript.slice(0, cut))
    s = reduce(s, { t: 200000, type: 'run_started', run: 'rx', branch: 'fast', hypothesis: 'H2' })
    s = reduce(s, { t: 201000, type: 'run_finished', run: { ...s.runs[2], id: 'rx', params: s.runs[2].params!, metrics: s.runs[2].metrics! } })
    expect(s.helps[0].expired).toBe(true)
    expect(s.runs.at(-1)?.assumption).toBe(true)
  })

  it('marks a run with null latency as failed instead of measured', () => {
    let s = fold(demoScript.slice(0, 5))
    s = reduce(s, { t: 8000, type: 'run_started', run: 'rf', branch: 'main', hypothesis: 'H0' })
    const metrics = { latency_ms: null, tflops: null, max_error: null, adv_error: null, vram_mb: null, compiled_ok: false, correct_ok: false }
    s = reduce(s, { t: 9000, type: 'run_finished', run: { id: 'rf', branch: 'main', hypothesis: 'H0', params: s.hypotheses[0].params, metrics: metrics as never } })
    expect(s.runs.at(-1)).toMatchObject({ id: 'rf', failed: true, metrics: undefined, finished: 9000 })
  })

  it('stores every verdict with its bound, in order, and the active bound', () => {
    expect(v.trail.H1.map(t => [t.verdict, t.provisional, t.bound.set_by, t.bound.standard, t.bound.adversarial])).toEqual([
      ['confirmed', true, 'speed', 2e-2, null],
      ['rejected', false, 'precision', 1e-3, 1e-3],
    ])
    expect(v.trail.H0[0].bound).toEqual({ standard: 1e-3, adversarial: null, set_by: 'default' })
    expect(Object.values(v.trail).flat()).toHaveLength(7)
    expect(v.bound).toEqual({ standard: 1e-3, adversarial: 1e-3, set_by: 'precision' })
    const before = fold(demoScript.filter(e => e.t < 60000))
    expect(before.bound?.adversarial).toBeNull()
  })

  it('handles ui actions', () => {
    let s = reduce(initialView(), { type: 'ui/lens', lens: 'precision' })
    s = reduce(s, { type: 'ui/select', selected: { run: 'r2', branch: 'speed' } })
    expect(s.lens).toBe('precision')
    expect(s.selected.run).toBe('r2')
  })
})

const runsDir = join(__dirname, '../../../runs')
const logs = existsSync(runsDir)
  ? readdirSync(runsDir, { recursive: true, encoding: 'utf8' }).filter(f => f.endsWith('.jsonl') && statSync(join(runsDir, f)).isFile())
  : []

describe('recorded log', () => {
  it.skipIf(!logs.length)(`reduces ${logs[0] ?? 'a recorded log (none in runs/ yet, skipped)'}`, () => {
    const events = parseLog(readFileSync(join(runsDir, logs[0]), 'utf8'))
    const v = fold(events)
    expect(v.started).toBe(true)
    expect(v.branches.length).toBeGreaterThan(0)
    expect(events.some(e => e.mock)).toBe(false)
  })
})
