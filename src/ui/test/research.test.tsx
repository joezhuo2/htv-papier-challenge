import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { App } from '../src/App'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })

describe('branch graph and metric chart', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('shows the rejected branch as dashed, struck through and labeled', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(70000)
    const node = screen.getByTestId('node-speed')
    expect(node.getAttribute('data-status')).toBe('rejected')
    expect(within(node).getByText('rejected')).toBeTruthy()
    expect(node.querySelector('rect.hit')?.getAttribute('stroke-dasharray')).toBe('5 3')
    expect(node.querySelector('[text-decoration="line-through"]')).toBeTruthy()
  })

  it('ghost prediction marker resolves to the measured point', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(7000)
    expect(screen.getByTestId('ghost-H0').getAttribute('data-resolved')).toBe('false')
    expect(screen.getByText(/H0 predicted at least 2.00x/)).toBeTruthy()
    expect(screen.queryByTestId('point-r1')).toBeNull()
    at(4000)
    expect(screen.getByTestId('ghost-H0').getAttribute('data-resolved')).toBe('true')
    expect(screen.getByTestId('point-r1')).toBeTruthy()
  })

  it('clicking a chart point opens the run in the right column, Esc clears it', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(70000)
    fireEvent.click(screen.getByTestId('point-r2'))
    const detail = screen.getByTestId('run-detail')
    const blocks = within(detail).getAllByTestId('verdict-block')
    expect(blocks.map(b => b.querySelector('.inline > span:last-child')!.textContent)).toEqual([
      'verdict 1 of 2, 0:37, standard under 2e-2 (set by Speed)',
      'verdict 2 of 2, 1:03, standard and adversarial under 1e-3 (set by Precision)',
    ])
    expect(within(blocks[0]).getByText('provisionally confirmed')).toBeTruthy()
    expect(within(blocks[1]).getByText('rejected')).toBeTruthy()
    expect(within(blocks[1]).getByText('fail')).toBeTruthy()
    expect(screen.getByTestId('adv-r2').getAttribute('data-over')).toBe('true')
    expect(screen.getByTestId('adv-panel').textContent).toContain('r2 3.2e-2 over bound')
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByTestId('run-detail')).toBeNull()
  })

  it('clicking a node selects its latest run', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(12000)
    fireEvent.click(screen.getByTestId('node-main').querySelector('[role="button"]')!)
    expect(within(screen.getByTestId('run-detail')).getByText('r1')).toBeTruthy()
  })

  it('keeps adversarial error ungated until the Precision directive, then reveals it', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(50000)
    expect(screen.getByTestId('adv-ungated')).toBeTruthy()
    expect(screen.getByTestId('adv-panel').textContent).not.toContain('3.2e-2')
    expect(screen.getByLabelText('legend').textContent).toContain('provisional')
    at(12000)
    expect(screen.getByTestId('adv-ungated')).toBeTruthy()
    at(2000)
    expect(screen.queryByTestId('adv-ungated')).toBeNull()
    expect(screen.getByTestId('adv-panel').textContent).toContain('bound 1e-3, set by Precision')
    fireEvent.click(screen.getByTestId('point-r1'))
    expect(screen.getByTestId('adv-panel').textContent).toContain('r2 3.2e-2 over bound')
    expect(screen.getByLabelText('legend').textContent).toContain('inconclusive')
  })

  it('reaches the final graph with five branches', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    fireEvent.click(screen.getByRole('tab', { name: 'Research' }))
    const statuses = ['main', 'speed', 'prec', 'fast', 'merge'].map(b => screen.getByTestId(`node-${b}`).getAttribute('data-status'))
    expect(statuses).toEqual(['done', 'rejected', 'merged', 'merged', 'done'])
  })
})
