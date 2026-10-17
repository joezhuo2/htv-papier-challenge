import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'

const recorded = demoScript.map(({ mock: _, ...e }) => e)

type Sock = { sent: string[]; onopen?: () => void; onmessage?: (e: { data: string }) => void; readyState: number }

const setup = () => {
  const socks: Sock[] = []
  vi.stubGlobal('WebSocket', function (this: Sock) {
    this.sent = []
    this.readyState = 0
    ;(this as unknown as { send: (d: string) => void }).send = d => this.sent.push(d)
    ;(this as unknown as { close: () => void }).close = () => {}
    socks.push(this)
  })
  render(<App source={{ mode: 'live', ws: 'ws://test/ws' }} />)
  act(() => { socks[0].readyState = 1; socks[0].onopen!() })
  const push = (...events: object[]) => act(() => events.forEach(e => socks[0].onmessage!({ data: JSON.stringify(e) })))
  const sent = () => socks[0].sent.map(x => JSON.parse(x))
  return { push, sent }
}

describe('live controls', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ started: false, checks: {
      gpu: { ok: true, detail: 'NVIDIA GeForce RTX 5060 Laptop GPU' }, gemini: { ok: true, detail: 'models answered 200' }, papiers: { ok: false, detail: 'papiers CLI not found' },
    } }))))
  })
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('pre-flight shows real checks, starts the session, and offers the replay fallback', async () => {
    const { sent, push } = setup()
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    const pf = screen.getByTestId('preflight')
    expect(within(pf).getByTestId('check-gpu').textContent).toContain('ok')
    expect(within(pf).getByTestId('check-papiers').textContent).toContain('failed')
    expect(within(pf).getByText('Fall back to the recorded run').getAttribute('href')).toBe('?mode=replay&file=%2Fruns%2Fdemo.jsonl')
    fireEvent.click(screen.getByRole('button', { name: 'Start session' }))
    expect(sent()).toEqual([{ type: 'start' }])
    push(recorded[0])
    expect(screen.queryByTestId('preflight')).toBeNull()
  })

  it('recording is a separate start option and shows in the mode pill', async () => {
    const { sent, push } = setup()
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    fireEvent.click(screen.getByRole('button', { name: 'Start recording' }))
    expect(sent()).toEqual([{ type: 'start', record: true }])
    push({ t: 0, type: 'agent_log', text: 'Measuring baselines: PyTorch eager and the naive Triton kernel.' })
    expect((screen.getByRole('button', { name: 'Start session' }) as HTMLButtonElement).disabled).toBe(true)
    expect((screen.getByRole('button', { name: 'Start recording' }) as HTMLButtonElement).disabled).toBe(true)
    fireEvent.click(screen.getByRole('button', { name: 'Start session' }))
    expect(sent()).toHaveLength(1)
    push(recorded[0], { t: 100, type: 'agent_log', text: 'Recording mode: the full loop for up to 8 hypotheses or until Stop.' })
    expect(screen.getByTestId('mode-pill').textContent).toContain('live, recording')
  })

  it('stop shows stopping until the session finishes, and new session restarts the server session', async () => {
    const { sent, push } = setup()
    push(recorded[0])
    fireEvent.click(screen.getByRole('button', { name: 'stop' }))
    expect(sent()).toEqual([{ type: 'stop' }])
    expect((screen.getByRole('button', { name: 'stopping...' }) as HTMLButtonElement).disabled).toBe(true)
    push(recorded.at(-1)!)
    expect((screen.getByRole('button', { name: 'stop' }) as HTMLButtonElement).disabled).toBe(true)
    const fetch = vi.fn(async () => new Response('{}', { status: 500 }))
    vi.stubGlobal('fetch', fetch)
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'new session' })) })
    expect(fetch).toHaveBeenCalledWith('/restart', { method: 'POST' })
  })

  it('directives show whether the agent applied them and are disabled after the session finishes', () => {
    const { push } = setup()
    push(recorded[0])
    const box = screen.getByLabelText('New directive or hunch')
    const send = (text: string) => {
      fireEvent.change(box, { target: { value: text } })
      fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    }
    send('Go faster.')
    expect(screen.getByText('Sent, waiting for the agent: Go faster.')).toBeTruthy()
    push({ t: 5000, type: 'directive_added', directive: { id: 'd1', lens: 'speed', text: 'Go faster.', constraint: 'maximize_speed', threshold: 0.02 } })
    expect(screen.getByText('Applied as d1: Go faster.')).toBeTruthy()
    send('Even faster.')
    push(recorded.at(-1)!)
    expect(screen.getByText('Session finished, directive not applied: Even faster.')).toBeTruthy()
    expect(screen.queryByText(/^Sent/)).toBeNull()
    expect((screen.getByLabelText('New directive or hunch') as HTMLTextAreaElement).disabled).toBe(true)
    expect(screen.getByText('Session finished, new directives are not applied.')).toBeTruthy()
  })

  it('the clock starts at 0:00, ticks from the first event, and a baseline failure shows on the pre-flight card', async () => {
    const { push } = setup()
    await act(async () => { await vi.advanceTimersByTimeAsync(400) })
    expect(screen.getByLabelText('elapsed').textContent).toBe('0:00')
    push({ t: 0, type: 'agent_log', text: 'Measuring baselines: PyTorch eager and the naive Triton kernel.' })
    await act(async () => { await vi.advanceTimersByTimeAsync(3000) })
    expect(screen.getByLabelText('elapsed').textContent).toBe('0:03')
    push({ t: 4000, type: 'agent_log', text: 'Cannot start: the eager baseline did not run (RuntimeError: Found no NVIDIA driver on your system.). Check the GPU, then press New session.' })
    await act(async () => { await vi.advanceTimersByTimeAsync(5000) })
    expect(screen.getByLabelText('elapsed').textContent).toBe('0:04')
    expect(screen.getByTestId('starting').textContent).toContain('Found no NVIDIA driver')
    expect((screen.getByRole('button', { name: 'Start session' }) as HTMLButtonElement).disabled).toBe(false)
  })

  it('shows a timeout as a visible state with retry and stop, cleared by progress', () => {
    const { sent, push } = setup()
    push(recorded[0], { t: 3000, type: 'agent_error', kind: 'llm_timeout', branch: 'main', retry_in_s: 30, text: 'The hypothesis model failed (Timeout). Retrying in 30 s or on retry.' })
    const card = screen.getByTestId('stalled')
    expect(card.textContent).toContain('timed out')
    fireEvent.click(within(card).getByRole('button', { name: 'Retry now' }))
    fireEvent.click(screen.getByRole('button', { name: 'stop' }))
    expect(sent()).toEqual([{ type: 'retry' }, { type: 'stop' }])
    push(recorded[3])
    expect(screen.queryByTestId('stalled')).toBeNull()
  })

  it('a harness timeout shows on the run as timed out', () => {
    const { push } = setup()
    const h0 = recorded[3] as { hypothesis: { params: object } }
    const nulls = { latency_ms: null, tflops: null, max_error: null, adv_error: null, vram_mb: null, compiled_ok: false, correct_ok: false }
    push(recorded[0], recorded[3], { t: 7500, type: 'run_started', run: 'r1', branch: 'main', hypothesis: 'H0' },
      { t: 30000, type: 'run_finished', error: 'timed out', run: { id: 'r1', branch: 'main', hypothesis: 'H0', params: h0.hypothesis.params, metrics: nulls } })
    fireEvent.click(screen.getByTestId('node-main').querySelector('[role="button"]')!)
    expect(within(screen.getByTestId('run-detail')).getByText('timed out (15 s harness limit)')).toBeTruthy()
  })

  it('save is labeled partial until session_finished, then saves the whole replay', () => {
    const { push } = setup()
    push(...recorded.slice(0, 5))
    const urls: Blob[] = []
    vi.stubGlobal('URL', { createObjectURL: (b: Blob) => { urls.push(b); return 'blob:x' }, revokeObjectURL: () => {} })
    const names: string[] = []
    vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(function (this: HTMLAnchorElement) { names.push(this.download) })
    expect(screen.queryByRole('button', { name: 'save as replay' })).toBeNull()
    fireEvent.click(screen.getByRole('button', { name: 'save partial replay' }))
    expect(names[0]).toMatch(/-partial\.jsonl$/)
    push(...recorded.slice(5))
    fireEvent.click(screen.getByRole('button', { name: 'save as replay' }))
    expect(names[1]).not.toMatch(/partial/)
    expect(urls).toHaveLength(2)
  })

  it('a provisional result shows the cue to send the Precision directive', () => {
    const { push } = setup()
    const h1 = recorded.findIndex(e => e.type === 'verdict' && e.hypothesis === 'H1')
    push(...recorded.slice(0, h1 + 1))
    expect(screen.getByTestId('provisional-cue').textContent).toContain('H1 passed provisionally under the Speed tolerance')
    expect(within(screen.getByTestId('tab-precision')).getByText('check')).toBeTruthy()
    fireEvent.click(screen.getByRole('button', { name: 'Switch to the Precision lens' }))
    expect(screen.getByTestId('tab-precision').getAttribute('aria-selected')).toBe('true')
  })
})
