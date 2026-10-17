import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { render, screen, within } from '@testing-library/react'
import type { DemoEvent, Metrics } from '@contract/types'
import { parseLog } from '../src/source'
import { fold } from '../src/state/reducer'
import { latencyCap, MetricChart } from '../src/components/MetricChart'
import { RunDetail } from '../src/components/RunDetail'

const golden = parseLog(readFileSync(join(__dirname, '../../../tests/fixtures/golden_session.jsonl'), 'utf8'))

describe('recording fixes', () => {
  it('clips the chart at 2x the median of non-outlier runs and marks the outlier off scale with its true value', () => {
    expect(latencyCap([0.7, 0.72, 0.71, 7.9])).toBeCloseTo(1.42)
    const last = golden.filter(e => e.type === 'run_finished').at(-1) as Extract<DemoEvent, { type: 'run_finished' }>
    const slow = { ...last, run: { ...last.run, metrics: { ...(last.run.metrics as Metrics), latency_ms: 40 } } }
    const view = fold(golden.map(e => (e === last ? slow : e)))
    render(<div style={{ width: 800, height: 300 }}><MetricChart view={view} onSelectRun={() => {}} /></div>)
    const mark = screen.getByTestId(`offscale-${last.run.id}`)
    expect(mark.getAttribute('aria-label')).toContain('off scale: 40.000 ms')
    expect(within(mark).getByText('off scale')).toBeTruthy()
    expect(screen.queryByTestId(`point-${last.run.id}`)).toBeNull()
  })

  it('hides provider failover and explanation fallback lines from older logs', () => {
    const t = golden[0].t
    const v = fold([...golden, { t, type: 'agent_log', text: 'gemini-3.8-flash failed (RateLimitError), answered by google/gemma-4-31b-it' },
      { t, type: 'agent_log', text: 'Explanation model unavailable (RateLimitError), using the factual summary.' }])
    expect(v.log.some(l => /answered by|Explanation model/.test(l.text))).toBe(false)
  })

  it('renders a raw Papiers id inside a hypothesis note as a note chip', () => {
    const run = golden.find(e => e.type === 'run_finished') as Extract<DemoEvent, { type: 'run_finished' }>
    const events = golden.map(e => (e.type === 'hypothesis_proposed' && e.hypothesis.id === run.run.hypothesis
      ? { ...e, hypothesis: { ...e.hypothesis, note: 'Context from Papiers note cnv_6630997c. Stages cut from 4 to 3' } } : e))
    const view = { ...fold(events), selected: { run: run.run.id } }
    render(<RunDetail view={view} onClose={() => {}} />)
    const note = screen.getByTestId('hypothesis-note')
    expect(note.textContent).not.toContain('cnv_6630997c')
    expect(within(note).getByTestId('note-chip').textContent).toContain('Papiers note')
  })
})

describe('summary attachment', () => {
  it('ties a summary to the hypothesis it names, not the newest run on its branch', () => {
    const runs = golden.filter(e => e.type === 'run_finished') as Extract<DemoEvent, { type: 'run_finished' }>[]
    const [first, last] = [runs[0].run, runs.at(-1)!.run]
    expect(first.hypothesis).not.toBe(last.hypothesis)
    const summary = (golden.find(e => e.type === 'summary') as Extract<DemoEvent, { type: 'summary' }>).summary
    const t = golden.at(-2)!.t
    const named = fold([...golden.slice(0, -1), { t, type: 'summary', summary: { ...summary, branch: last.branch, hypothesis: first.hypothesis } }]).feed.at(-1)
    const legacy = fold([...golden.slice(0, -1), { t, type: 'summary', summary: { ...summary, branch: last.branch, hypothesis: undefined } }]).feed.at(-1)
    expect(named?.kind === 'summary' && named.hypothesis).toBe(first.hypothesis)
    expect(legacy?.kind === 'summary' && legacy.hypothesis).toBe(last.hypothesis)
  })
})
