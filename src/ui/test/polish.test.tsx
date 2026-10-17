import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'
import { chapters } from '../src/chapters'
import { parseLog, parseSource } from '../src/source'
import { fold } from '../src/state/reducer'
import { ResultCard } from '../src/components/Final'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })
const recorded = demoScript.map(({ mock: _, ...e }) => e)

describe('chapters', () => {
  it('finds the beats of the demo from the events', () => {
    expect(chapters(demoScript).map(c => [c.key, c.label, c.t])).toEqual([
      ['first', 'First prediction', 6500],
      ['provisional', 'H1 provisional', 37000],
      ['flip', 'Precision flip', 63000],
      ['post_mortem', 'Post-mortem H1', 74000],
      ['help', 'Help request', 97000],
      ['hybrid', 'Hybrid', 152000],
      ['end', 'Best result', 176000],
    ])
  })

  it('jumps to a chapter in mock mode and selects its run', () => {
    vi.useFakeTimers()
    render(<App source={{ mode: 'mock' }} />)
    at(0)
    fireEvent.change(screen.getByLabelText('jump to chapter'), { target: { value: 'flip' } })
    const detail = screen.getByTestId('run-detail')
    expect(within(detail).getByText('r2')).toBeTruthy()
    expect(within(detail).getAllByTestId('verdict-block')).toHaveLength(2)
    expect(screen.getByRole('button', { name: 'resume' })).toBeTruthy()
    vi.useRealTimers()
  })

  it('auto-pause stops at the next chapter', () => {
    vi.useFakeTimers()
    render(<App source={{ mode: 'mock' }} />)
    fireEvent.click(screen.getByRole('button', { name: 'auto-pause' }))
    at(20000)
    expect(screen.getByRole('button', { name: 'resume' })).toBeTruthy()
    expect(screen.queryByTestId('pending-run')).toBeNull()
    expect(screen.getByText(/H0 predicted at least/)).toBeTruthy()
    vi.useRealTimers()
  })
})

describe('labels, tooltips and the result screen', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('shows the mode pill and the compression note for a recorded run', async () => {
    const stretched = recorded.map((e, i) => ({ ...e, t: e.t + i * 10000 }))
    vi.stubGlobal('fetch', vi.fn(async () => new Response(stretched.map(e => JSON.stringify(e)).join('\n'))))
    render(<App source={{ mode: 'replay', file: '/runs/x.jsonl' }} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(screen.getByTestId('mode-pill').textContent).toBe('recorded runidle gaps compressed, 13:46 to 4:20')
  })

  it('verdict chips and terms carry plain-language definitions', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(40000)
    const chip = screen.getAllByText('provisionally confirmed')[0].closest('.verdict')!
    expect(chip.getAttribute('title')).toMatch(/^Provisional: passed under the looser Speed tolerance/)
    expect(screen.getByText('Adversarial error by run').getAttribute('title')).toMatch(/^Adversarial batch/)
  })

  it('opens the final result card at the end with the best run, rejections and decisions', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    const card = screen.getByTestId('result-card')
    expect(card.textContent).toContain('6.72x faster than naive, 1.13x vs PyTorch eager')
    expect(card.textContent).toContain('Adversarial error 3.2e-2 violates the 1e-3 bound')
    expect(card.textContent).toContain('Speed directive: Push throughput harder')
    expect(card.textContent).toContain('exports/ledger.md')
    fireEvent.click(screen.getByRole('button', { name: 'restart' }))
    expect(screen.queryByTestId('result-card')).toBeNull()
  })

  it('result card says so when no run stayed inside every bound', () => {
    render(<ResultCard view={fold(demoScript.slice(0, 3))} />)
    expect(screen.getByText('No run stayed inside every bound in this session.')).toBeTruthy()
  })

  it('first load is clean: titled, no selection, log hidden', () => {
    render(<App source={{ mode: 'mock' }} />)
    expect(document.querySelector('h1')!.textContent).toBe('Research War Room')
    expect(screen.queryByTestId('run-detail')).toBeNull()
    expect(screen.getByRole('button', { name: /Agent log/ }).getAttribute('aria-expanded')).toBe('false')
  })
})

describe('judging build and log validation', () => {
  it('the judging build never falls back to mock', () => {
    expect(parseSource('', true)).toMatchObject({ mode: 'replay', file: undefined })
    expect(parseSource('?mode=mock', true).mode).toBe('replay')
    expect(parseSource('?mode=live', true).mode).toBe('live')
    expect(parseSource('', false).mode).toBe('mock')
  })

  it('rejects malformed logs with a clear message', () => {
    expect(() => parseLog('not json')).toThrow('Line 1 is not valid JSON')
    expect(() => parseLog('{"t": 0}')).toThrow('Line 1 has no numeric t or no type')
    expect(() => parseLog('{"t": 0, "type": "explode"}')).toThrow('Line 1 has an unknown event type "explode"')
    const s = JSON.stringify(recorded[0])
    expect(() => parseLog(`${s}\n{"t": 1, "type": "verdict", "hypothesis": "H0"}`)).toThrow('Line 2: verdict is missing verdict, provisional, reason, checks, bound')
    expect(() => parseLog('')).toThrow('The log is empty')
    expect(parseLog(recorded.map(e => JSON.stringify(e)).join('\n'))).toHaveLength(recorded.length)
  })

  it('shows the validation error in the banner instead of a blank screen', async () => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => new Response('{"t": 0, "type": "explode"}')))
    render(<App source={{ mode: 'replay', file: '/runs/bad.jsonl' }} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(screen.getByRole('alert').textContent).toBe('Could not replay /runs/bad.jsonl: Line 1 has an unknown event type "explode"')
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })
})
