import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import type { DemoEvent, PriorVerdict } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'
import { DirectivePanel } from '../src/components/DirectivePanel'
import { PapiersPanel } from '../src/components/PapiersPanel'
import { SummaryFeed } from '../src/components/SummaryFeed'
import { parseLog } from '../src/source'
import { fold } from '../src/state/reducer'

const base = { block_m: 16, block_n: 16, block_k: 16, warps: 1, stages: 1, acc: 'fp32', gelu: 'exact' }
const prior: PriorVerdict = {
  id: 'hl_1', doc_id: 'doc_mp', run: '20261018-201500/H1', verdict: 'confirmed', measured: '1.54x', bound: '>= 1.20x', reason: 'Prediction holds.',
  fp: { params: { gelu: 'approx' }, base, kernel: 'matmul_bias_gelu', title: 'Approximate GELU', bounds: { std: '2.0e-2', adv: 'none' }, rounds: 7 },
}
const section = 'Prior verdicts from earlier sessions, read from Papiers (past results, not evidence, never cite them as support for a claim).\n| title |'
const speed: DemoEvent = { t: 10, type: 'directive_added', directive: { id: 'd1', lens: 'speed', text: 'try the tanh GELU approximation', constraint: 'maximize_speed', threshold: 2e-2 } }
const dup: DemoEvent = { t: 20, type: 'prior_duplicate', help: 'd1', branch: 'speed', prior, text: 'Because the Speed lens sent "try tanh", but an earlier session already ran this: gelu=approx. Confirm to run it anyway.' }
const log = (...e: DemoEvent[]) => fold([demoScript[0], ...e])

describe('prior verdicts in the Papiers panel', () => {
  it('lists each prior verdict with its change, numbers and bounds, and the exact section the agent saw', () => {
    render(<PapiersPanel view={log({ t: 1, type: 'prior_verdicts', count: 1, items: [prior], section })} />)
    expect(screen.getByTestId('prior-count').textContent).toBe('Prior verdicts read: 1')
    const row = screen.getByTestId('prior-row')
    expect(row.textContent).toContain('Approximate GELU')
    expect(row.textContent).toContain('gelu=approx')
    expect(row.textContent).toContain('1.54x vs >= 1.20x')
    expect(row.textContent).toContain('bounds std 2.0e-2 adv none')
    expect(row.textContent).toContain('20261018-201500/H1')
    expect(screen.getByTestId('prior-section').textContent).toBe(section)
  })

  it('shows N=0, an old note without a fingerprint, and an ignored session', () => {
    const { unmount } = render(<PapiersPanel view={log({ t: 1, type: 'prior_verdicts', count: 0, items: [], section: '' })} />)
    expect(screen.getByTestId('prior-count').textContent).toBe('Prior verdicts read: 0')
    expect(screen.queryByTestId('prior-table')).toBeNull()
    unmount()
    const old = { ...prior, fp: undefined }
    const second = render(<PapiersPanel view={log({ t: 1, type: 'prior_verdicts', count: 1, items: [old] })} />)
    expect(screen.getByTestId('prior-row').textContent).toContain('old note, no fingerprint')
    expect(screen.queryByTestId('prior-section')).toBeNull()
    second.unmount()
    render(<PapiersPanel view={log({ t: 1, type: 'prior_verdicts', count: 0, items: [], section: '', ignored: true })} />)
    expect(screen.getByTestId('prior-count').textContent).toBe('Prior verdicts read: 0 (ignored for this session)')
  })

  it('keeps the latest section when the branch bounds change which rows block', () => {
    const v = log({ t: 1, type: 'prior_verdicts', count: 1, items: [prior], section: 'first' }, { t: 2, type: 'prior_verdicts', count: 1, items: [prior], section: 'second', branch: 'speed' })
    render(<PapiersPanel view={v} />)
    expect(screen.getByTestId('prior-section').textContent).toBe('second')
  })
})

describe('a refused repeat in the event feed', () => {
  it('shows the refusal reason with the prior session, verdict and numbers', () => {
    const text = 'it repeats an experiment an earlier session already ran: gelu=approx from block_m=16, confirmed in 20261018-201500/H1 (1.54x vs >= 1.20x) under the same bounds; propose something different'
    const v = log({ t: 5, type: 'prior_refused', branch: 'prec', prior, text })
    render(<SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} />)
    const card = screen.getByTestId('prior-refused')
    expect(card.textContent).toContain('Refused a repeat')
    expect(card.textContent).toContain('confirmed in 20261018-201500/H1')
    expect(card.textContent).toContain('1.54x vs >= 1.20x')
    expect(card.textContent).toContain('On prec')
  })

  it('labels a skipped seed', () => {
    const v = log({ t: 5, type: 'prior_refused', branch: 'speed', prior, text: 'x', seed: true })
    render(<SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} />)
    expect(screen.getByTestId('prior-refused').textContent).toContain('Skipped a repeat')
  })
})

describe('the hunch confirm card', () => {
  it('warns in the Directive panel with the prior numbers and sends Run anyway or Skip', () => {
    const onSend = vi.fn()
    render(<DirectivePanel view={log(speed, dup)} sent={[]} onSend={onSend} onSelectBranch={() => {}} />)
    const card = screen.getByTestId('prior-hold-d1')
    expect(card.textContent).toContain('An earlier session already ran this')
    expect(card.textContent).toContain('Confirm to run it anyway')
    fireEvent.click(within(card).getByRole('button', { name: 'Run anyway' }))
    fireEvent.click(within(card).getByRole('button', { name: 'Skip' }))
    expect(onSend.mock.calls).toEqual([[{ type: 'confirm_prior', help: 'd1', run: true }], [{ type: 'confirm_prior', help: 'd1', run: false }]])
  })

  it('goes away when the agent answers the hunch, stays off the other lens and is disabled once the session finished', () => {
    const answered = log(speed, dup, { t: 30, type: 'plan_change', help: 'd1', text: 'Testing it next.' })
    expect(answered.holds).toEqual([])
    const open = { ...log(speed, dup), lens: 'precision' as const }
    const { unmount } = render(<DirectivePanel view={open} sent={[]} onSend={() => {}} onSelectBranch={() => {}} />)
    expect(screen.queryByTestId('prior-hold-d1')).toBeNull()
    unmount()
    const done = { ...log(speed, dup), finished: true }
    render(<DirectivePanel view={done} sent={[]} onSend={() => {}} onSelectBranch={() => {}} />)
    expect((screen.getByRole('button', { name: 'Run anyway' }) as HTMLButtonElement).disabled).toBe(true)
  })
})

describe('the log contract', () => {
  const line = (e: object) => JSON.stringify(e)
  it('accepts the new events and rejects them without their keys', () => {
    const start = line(demoScript[0])
    expect(parseLog([start, line({ t: 1, type: 'prior_refused', branch: 'b', prior, text: 'x' }), line(dup)].join('\n'))).toHaveLength(3)
    expect(() => parseLog([start, line({ t: 1, type: 'prior_refused', branch: 'b' })].join('\n'))).toThrow('prior_refused is missing prior, text')
    expect(() => parseLog([start, line({ t: 1, type: 'prior_duplicate', help: 'd1', branch: 'b', prior })].join('\n'))).toThrow('prior_duplicate is missing text')
  })
})

describe('the Ignore prior verdicts switch', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ started: false, checks: { gpu: { ok: true, detail: 'GPU' } } }))))
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  const open = async () => {
    const socks: { sent: string[]; readyState: number; onopen?: () => void }[] = []
    vi.stubGlobal('WebSocket', function (this: { sent: string[]; readyState: number }) {
      this.sent = []
      this.readyState = 0
      ;(this as unknown as { send: (d: string) => void }).send = d => this.sent.push(d)
      ;(this as unknown as { close: () => void }).close = () => {}
      socks.push(this)
    })
    render(<App source={{ mode: 'live', ws: 'ws://test/ws' }} />)
    act(() => { socks[0].readyState = 1; socks[0].onopen!() })
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    return () => socks[0].sent.map(x => JSON.parse(x))
  }

  it('starts with prior verdicts on by default and sends ignore_prior only when ticked', async () => {
    const sent = await open()
    expect((screen.getByTestId('ignore-prior') as HTMLInputElement).checked).toBe(false)
    fireEvent.click(screen.getByTestId('ignore-prior'))
    fireEvent.click(screen.getByRole('button', { name: 'Start session' }))
    expect(sent()).toEqual([{ type: 'start', ignore_prior: true }])
  })

  it('combines with recording', async () => {
    const sent = await open()
    fireEvent.click(screen.getByTestId('ignore-prior'))
    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }))
    expect(sent()).toEqual([{ type: 'start', record: true, ignore_prior: true }])
  })
})
