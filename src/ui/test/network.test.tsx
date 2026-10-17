import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen } from '@testing-library/react'
import { readFileSync, readdirSync } from 'node:fs'
import { join } from 'node:path'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })
const ui = join(__dirname, '..')

describe('network at load', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('mock mode makes no network request through the whole script', () => {
    const fetch = vi.fn()
    const ws = vi.fn()
    vi.stubGlobal('fetch', fetch)
    vi.stubGlobal('WebSocket', ws)
    const xhr = vi.spyOn(XMLHttpRequest.prototype, 'open')
    render(<App source={{ mode: 'mock' }} />)
    at(180000)
    expect(fetch).not.toHaveBeenCalled()
    expect(ws).not.toHaveBeenCalled()
    expect(xhr).not.toHaveBeenCalled()
  })

  it('index.html, styles and bundled font CSS reference no other origin', () => {
    const files = [join(ui, 'index.html'), join(ui, 'src/styles.css')]
    for (const pkg of ['inter', 'jetbrains-mono']) {
      for (const w of ['400', '600']) {
        const f = join(ui, `node_modules/@fontsource/${pkg}/${w}.css`)
        if (readdirSync(join(ui, `node_modules/@fontsource/${pkg}`)).includes(`${w}.css`)) files.push(f)
      }
    }
    for (const f of files) expect(readFileSync(f, 'utf8'), f).not.toMatch(/url\(\s*['"]?(https?:)?\/\//)
    expect(readFileSync(files[0], 'utf8')).not.toMatch(/(src|href)="(https?:)?\/\//)
  })
})

describe('replay mode', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('loads a recorded log, labels it recorded run, and shows no mock tag', async () => {
    const recorded = demoScript.map(({ mock: _, ...e }) => e)
    const fetch = vi.fn(async () => new Response(recorded.map(e => JSON.stringify(e)).join('\n')))
    vi.stubGlobal('fetch', fetch)
    render(<App source={{ mode: 'replay', file: '/runs/demo.jsonl' }} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(fetch).toHaveBeenCalledWith('/runs/demo.jsonl')
    expect(screen.getByText('recorded run')).toBeTruthy()
    at(200000)
    expect(screen.getAllByTestId('summary-card').length).toBe(7)
    expect(screen.queryByText('mock')).toBeNull()
  })

  it('shows a banner when the log cannot load', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response('', { status: 404 })))
    render(<App source={{ mode: 'replay', file: '/runs/missing.jsonl' }} />)
    await act(async () => { await vi.advanceTimersByTimeAsync(0) })
    expect(screen.getByRole('alert').textContent).toContain('Could not load /runs/missing.jsonl (404)')
  })
})

describe('live mode', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
  })

  it('folds pushed events, sends commands, and shows the reconnect banner', () => {
    const socks: { sent: string[]; onopen?: () => void; onclose?: () => void; onmessage?: (e: { data: string }) => void; readyState: number }[] = []
    vi.stubGlobal('WebSocket', function (this: (typeof socks)[0]) {
      this.sent = []
      this.readyState = 0
      ;(this as unknown as { send: (d: string) => void }).send = d => this.sent.push(d)
      ;(this as unknown as { close: () => void }).close = () => {}
      socks.push(this)
    })
    render(<App source={{ mode: 'live', ws: 'ws://test/ws' }} />)
    expect(screen.getByText('live')).toBeTruthy()
    expect(screen.queryByRole('group', { name: 'presenter controls' })).toBeNull()
    act(() => {
      socks[0].readyState = 1
      socks[0].onopen!()
      demoScript.slice(0, 7).forEach(({ mock: _, ...e }) => socks[0].onmessage!({ data: JSON.stringify(e) }))
    })
    expect(screen.getByTestId('point-r1')).toBeTruthy()
    fireEvent.change(screen.getByLabelText(/new directive/i), { target: { value: 'faster' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send' }))
    expect(JSON.parse(socks[0].sent[0])).toEqual({ type: 'add_directive', lens: 'speed', text: 'faster' })
    act(() => { socks[0].readyState = 3; socks[0].onclose!() })
    expect(screen.getByTestId('reconnect-banner').textContent).toContain('Reconnecting in 1s')
    at(1000)
    expect(socks).toHaveLength(2)
  })
})
