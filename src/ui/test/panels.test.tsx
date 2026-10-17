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
    expect(screen.getByRole('button', { name: 'mock' }).getAttribute('aria-pressed')).toBe('true')
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
    expect(cards.map(c => c.getAttribute('data-collapsed'))).toEqual(['true', 'true', null])
    expect(within(cards[1]).getByText('provisionally confirmed')).toBeTruthy()
    expect(within(cards[2]).getAllByText('rejected')).toHaveLength(2)
    fireEvent.click(cards[1])
    const opened = screen.getAllByTestId('summary-card')[1]
    expect(within(opened).getByText(/superseded, now rejected/)).toBeTruthy()
    expect(within(opened).getByTestId('judged-bound').textContent).toBe('standard under 2e-2 (set by Speed)')
    expect(within(opened).queryByTestId('verdict-trail')).toBeNull()
    const last = screen.getAllByTestId('summary-card')[2]
    expect(within(last).getByTestId('judged-bound').textContent).toBe('standard and adversarial under 1e-3 (set by Precision)')
    expect(within(last).getByTestId('verdict-trail').textContent).toBe(
      'provisionally confirmed standard under 2e-2 (set by Speed), 0:37rejected standard and adversarial under 1e-3 (set by Precision), 1:03')
    fireEvent.click(within(opened).getByRole('button', { name: 'Collapse' }))
    expect(screen.getAllByTestId('summary-card')[1].getAttribute('data-collapsed')).toBe('true')
  })

  it('records commands locally in mock mode', () => {
    render(<App source={{ mode: 'mock' }} />)
    fireEvent.change(screen.getByLabelText(/new directive/i), { target: { value: 'Try approximate GELU' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(screen.getByText(/Recorded locally, not delivered/)).toBeTruthy()
  })

  it('prev re-folds one event back and the mode toggle navigates', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(0)
    fireEvent.keyDown(window, { key: ' ' })
    for (let i = 0; i < 8; i++) fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(screen.getAllByTestId('summary-card')).toHaveLength(1)
    fireEvent.click(screen.getByRole('button', { name: 'prev' }))
    expect(screen.getAllByTestId('summary-card')).toHaveLength(1)
    fireEvent.keyDown(window, { key: 'ArrowLeft' })
    expect(screen.queryByTestId('summary-card')).toBeNull()
    fireEvent.keyDown(window, { key: 'ArrowRight' })
    expect(screen.getAllByTestId('summary-card')).toHaveLength(1)
    const loc = window.location
    const fake = { search: '' }
    Object.defineProperty(window, 'location', { value: fake, configurable: true })
    fireEvent.click(screen.getByRole('button', { name: 'mock' }))
    expect(fake.search).toBe('')
    fireEvent.click(screen.getByRole('button', { name: 'live' }))
    expect(fake.search).toBe('?mode=live')
    Object.defineProperty(window, 'location', { value: loc, configurable: true })
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
