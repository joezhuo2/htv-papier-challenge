import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'
import { DirectivePanel } from '../src/components/DirectivePanel'
import { fold, reduce } from '../src/state/reducer'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })

describe('steering and feed', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('directive input emits add_directive for the active lens', () => {
    const onSend = vi.fn()
    const view = reduce(fold(demoScript.slice(0, 30)), { type: 'ui/lens', lens: 'precision' })
    render(<DirectivePanel view={view} sent={[]} onSend={onSend} onSelectBranch={() => {}} />)
    fireEvent.change(screen.getByLabelText(/new directive/i), { target: { value: 'Keep adversarial error under 1e-3' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(onSend).toHaveBeenCalledWith({ type: 'add_directive', lens: 'precision', text: 'Keep adversarial error under 1e-3' })
  })

  it('shows directives per lens, the conflict notice, and switches lens with keys', () => {
    render(<App source={{ mode: 'mock' }} />)
    expect(screen.getByText('mock')).toBeTruthy()
    at(90000)
    expect(screen.getByTestId('directive-d1')).toBeTruthy()
    expect(screen.queryByTestId('directive-d2')).toBeNull()
    fireEvent.keyDown(window, { key: '2' })
    const d2 = screen.getByTestId('directive-d2')
    expect(within(d2).getByRole('note').textContent).toBe('Conflicts with branch speed, forked prec')
    expect(document.querySelector('.app')?.getAttribute('data-lens')).toBe('precision')
  })

  it('feed shows summary cards with verdict words and the mock tag', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(13000)
    const card = screen.getByTestId('summary-card')
    expect(within(card).getByText('confirmed')).toBeTruthy()
    expect(within(card).getAllByText('mock').length).toBeGreaterThan(0)
    at(64000)
    const cards = screen.getAllByTestId('summary-card')
    expect(cards).toHaveLength(3)
    expect(within(cards[1]).getByText('provisionally confirmed')).toBeTruthy()
    expect(within(cards[1]).getByText(/superseded, now rejected/)).toBeTruthy()
    expect(within(cards[2]).getByText('rejected')).toBeTruthy()
  })

  it('records commands locally in mock mode', () => {
    render(<App source={{ mode: 'mock' }} />)
    fireEvent.change(screen.getByLabelText(/new directive/i), { target: { value: 'Try approximate GELU' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(screen.getByText(/Recorded locally, not delivered/)).toBeTruthy()
  })

  it('space pauses and right arrow steps', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(0)
    fireEvent.keyDown(window, { key: ' ' })
    at(60000)
    expect(screen.queryByTestId('summary-card')).toBeNull()
    for (let i = 0; i < 8; i++) fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(screen.getByTestId('summary-card')).toBeTruthy()
  })
})
