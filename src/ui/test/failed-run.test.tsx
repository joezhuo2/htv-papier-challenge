import { describe, expect, it } from 'vitest'
import { render, screen, within } from '@testing-library/react'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { RunDetail } from '../src/components/RunDetail'
import { fold, reduce } from '../src/state/reducer'

describe('a run that failed to compile', () => {
  it('arrives with null metrics, is marked failed, and renders as failed', () => {
    let v = fold(demoScript.slice(0, 5))
    const h = v.hypotheses[0]
    const nulls = { latency_ms: null, tflops: null, max_error: null, adv_error: null, vram_mb: null, compiled_ok: false, correct_ok: false }
    v = reduce(v, { t: 8000, type: 'run_started', run: 'rf', branch: 'main', hypothesis: h.id })
    v = reduce(v, { t: 9000, type: 'run_finished', run: { id: 'rf', branch: 'main', hypothesis: h.id, params: h.params, metrics: nulls } } as unknown as DemoEvent)
    v = reduce(v, { type: 'ui/select', selected: { run: 'rf', branch: 'main' } })
    expect(v.runs.at(-1)).toMatchObject({ failed: true, metrics: undefined })
    render(<RunDetail view={v} onClose={() => {}} />)
    expect(within(screen.getByTestId('run-detail')).getByText('failed to compile or timed out')).toBeTruthy()
  })
})
