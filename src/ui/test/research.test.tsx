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
    expect(within(detail).getByText('rejected')).toBeTruthy()
    expect(within(detail).getByText('fail')).toBeTruthy()
    expect(screen.getByTestId('adv-panel').textContent).toContain('3.2e-2 over')
    fireEvent.keyDown(window, { key: 'Escape' })
    expect(screen.queryByTestId('run-detail')).toBeNull()
  })

  it('clicking a node selects its latest run', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(12000)
    fireEvent.click(screen.getByTestId('node-main').querySelector('[role="button"]')!)
    expect(within(screen.getByTestId('run-detail')).getByText('r1')).toBeTruthy()
  })

  it('reaches the final graph with five branches', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    const statuses = ['main', 'speed', 'prec', 'fast', 'merge'].map(b => screen.getByTestId(`node-${b}`).getAttribute('data-status'))
    expect(statuses).toEqual(['done', 'rejected', 'merged', 'merged', 'done'])
  })
})
