import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, render, screen } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })
const recorded = demoScript.map(({ mock: _, ...e }) => e)

describe('working states', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('shows a pending card and tick between run_started and run_finished, then the summary', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(8000)
    const card = screen.getByTestId('pending-run')
    expect(card.textContent).toContain('Running r1 on main')
    expect(card.textContent).toContain('at least 2.00x vs naive (41.0 ms), error default 1e-3')
    expect(screen.getByTestId('pending-r1')).toBeTruthy()
    expect(screen.queryByTestId('choosing')).toBeNull()
    at(2600)
    expect(screen.queryByTestId('pending-run')).toBeNull()
    expect(screen.queryByTestId('pending-r1')).toBeNull()
    at(1500)
    expect(screen.getAllByTestId('summary-card')).toHaveLength(1)
  })

  it('never shows a pending or choosing state in recorded replay', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(recorded.map(e => JSON.stringify(e)).join('\n'))))
    render(<App source={{ mode: 'replay', file: '/runs/x.jsonl' }} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    for (let t = 0; t < 40; t++) {
      at(500)
      expect(screen.queryByTestId('pending-run')).toBeNull()
      expect(screen.queryByTestId('choosing')).toBeNull()
      expect(document.querySelector('[data-testid^="pending-r"]')).toBeNull()
    }
    expect(screen.getAllByTestId('summary-card').length).toBeGreaterThan(0)
  })

  it('in live mode shows choosing next hypothesis only between a verdict and the next proposal', () => {
    const socks: { onopen?: () => void; onmessage?: (e: { data: string }) => void; readyState: number }[] = []
    vi.stubGlobal('WebSocket', function (this: (typeof socks)[0]) {
      this.readyState = 0
      ;(this as unknown as { send: () => void; close: () => void }).send = () => {}
      ;(this as unknown as { close: () => void }).close = () => {}
      socks.push(this)
    })
    render(<App source={{ mode: 'live', ws: 'ws://test/ws' }} />)
    const push = (from: number, to: number) => act(() => recorded.slice(from, to).forEach(e => socks[0].onmessage!({ data: JSON.stringify(e) })))
    act(() => { socks[0].readyState = 1; socks[0].onopen!() })
    push(0, 5)
    expect(screen.getByTestId('pending-run')).toBeTruthy()
    expect(screen.queryByTestId('choosing')).toBeNull()
    push(5, 7)
    expect(screen.queryByTestId('pending-run')).toBeNull()
    expect(screen.getAllByTestId('choosing')).toHaveLength(1)
    push(7, 9)
    expect(screen.queryByTestId('choosing')).toBeNull()
  })

  it('mock mode never shows the choosing state', () => {
    render(<App source={{ mode: 'mock' }} />)
    for (let t = 0; t < 30; t++) {
      at(1000)
      expect(screen.queryByTestId('choosing')).toBeNull()
    }
  })
})
