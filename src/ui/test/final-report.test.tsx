import { describe, expect, it } from 'vitest'
import { render, screen } from '@testing-library/react'
import type { DemoEvent, FinalReport } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { parseLog } from '../src/source'
import { fold } from '../src/state/reducer'
import { ResultCard } from '../src/components/Final'

const report: FinalReport = {
  best: { run: 'r1', hypothesis: 'H0', branch: 'main', params: { block_m: 128, block_n: 128, block_k: 32, warps: 4, stages: 3, acc: 'fp32', gelu: 'exact' }, latency_ms: 0.642, vs_naive: 2.95, vs_eager: 1.04 },
  ruled_out: ['H1 on speed, acc fp32 to fp16: rejected, 1.43x vs its parent'],
  next: ['Rerun H4 (gelu exact to approx) with more timing rounds: inconclusive, 1.03x vs its parent sits inside the noise band.'],
  human: [],
  hunches: [{ directive: 'd3', lens: 'speed', text: 'Try 128x128x32 tiles with 3 stages', outcome: 'not run: speed2\'s current best already has it' }],
}

const withReport = (): DemoEvent[] => {
  const events = demoScript.map(({ mock: _, ...e }) => e) as DemoEvent[]
  const end = events.findIndex(e => e.type === 'session_finished')
  return [...events.slice(0, end), { t: events[end].t, type: 'final_report', report }, ...events.slice(end)]
}

describe('final report', () => {
  it('passes log validation and renders every section with the best run labelled best confirmed', () => {
    const events = parseLog(withReport().map(e => JSON.stringify(e)).join('\n'))
    render(<ResultCard view={fold(events)} />)
    expect(screen.getByTestId('best-label').textContent).toBe('best confirmed')
    expect(screen.getByTestId('final-ruled').textContent).toContain('H1 on speed, acc fp32 to fp16: rejected')
    expect(screen.getByTestId('final-what').textContent).toContain('Rerun H4')
    expect(screen.getByTestId('final-needs').textContent).toContain('Nothing is waiting on a lens owner.')
    expect(screen.getByTestId('final-hunches').textContent).toContain('d3 Speed: "Try 128x128x32 tiles with 3 stages": not run')
  })

  it('older logs without a final report still show the result card', () => {
    render(<ResultCard view={fold(demoScript)} />)
    expect(screen.queryByTestId('final-hunches')).toBeNull()
    expect(screen.getByTestId('best-label')).toBeTruthy()
  })
})
