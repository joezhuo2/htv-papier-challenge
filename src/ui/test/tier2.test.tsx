import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'
import { HybridCard } from '../src/components/Tier2'
import { fold } from '../src/state/reducer'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })

describe('tier 2', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('hybrid card shows both param sets and Approve emits approve_hybrid', () => {
    const view = fold(demoScript.slice(0, demoScript.findIndex(e => e.type === 'hybrid_proposed') + 1))
    const item = view.feed.find(f => f.kind === 'hybrid')!
    const onSend = vi.fn()
    const onDecide = vi.fn()
    render(<HybridCard view={view} item={item as Extract<typeof item, { kind: 'hybrid' }>} onSend={onSend} onDecide={onDecide} />)
    const card = screen.getByTestId('hybrid-card')
    expect(within(card).getByText('stages *')).toBeTruthy()
    expect(within(card).getByText('gelu *')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Approve' }))
    expect(onSend).toHaveBeenCalledWith({ type: 'approve_hybrid', branches: ['prec', 'fast'] })
    expect(onDecide).toHaveBeenCalledWith(true)
  })

  it('ledger ends with six rows and the H1 history, report view opens', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    const rows = within(screen.getByTestId('ledger')).getAllByRole('row')
    expect(rows).toHaveLength(7)
    expect(rows[2].textContent).toContain('was confirmed')
    expect(within(screen.getByTestId('hybrid-card')).getByText(/^Approved/).textContent).toBe('Approved, branch merge created')
    fireEvent.click(screen.getByRole('tab', { name: 'Report' }))
    expect(screen.getByTestId('report').textContent).toContain('Best at session end: r6')
    fireEvent.click(screen.getByRole('tab', { name: 'Research' }))
    expect(screen.getByTestId('branch-graph')).toBeTruthy()
  })

  it('Papiers findings are labeled exported, never written', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    const feed = screen.getByLabelText('summary feed').textContent!
    expect(feed).toContain('Exported for Papiers import')
    expect(feed).toContain('exports/H1-negative-result.md')
    expect(feed.match(/(write|written) (results )?to papiers/i)).toBeNull()
  })
})
