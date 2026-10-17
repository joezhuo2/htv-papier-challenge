import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DemoEvent } from '@contract/types'
import { backoff, connectLive, parseMessage, type LiveStatus } from '../src/live'

class FakeSocket {
  static all: FakeSocket[] = []
  readyState = 0
  out: string[] = []
  onopen: ((ev: unknown) => void) | null = null
  onclose: ((ev: unknown) => void) | null = null
  onmessage: ((ev: { data: unknown }) => void) | null = null
  constructor(public url: string) { FakeSocket.all.push(this) }
  send(d: string) { this.out.push(d) }
  close() { this.drop() }
  open() { this.readyState = 1; this.onopen?.({}) }
  drop() { this.readyState = 3; this.onclose?.({}) }
  push(data: unknown) { this.onmessage?.({ data: JSON.stringify(data) }) }
}

const setup = () => {
  FakeSocket.all = []
  const events: DemoEvent[] = []
  const statuses: LiveStatus[] = []
  const errors: string[] = []
  const c = connectLive({
    url: 'ws://test/ws', onEvent: e => events.push(e), onStatus: s => statuses.push(s), onError: m => errors.push(m),
    socket: url => new FakeSocket(url),
  })
  return { c, events, statuses, errors, sock: () => FakeSocket.all.at(-1)! }
}

describe('live socket', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('folds single events and arrays in order', () => {
    const { events, sock } = setup()
    sock().open()
    sock().push({ t: 0, type: 'agent_log', text: 'a' })
    sock().push([{ t: 1, type: 'agent_log', text: 'b' }, { t: 2, type: 'agent_log', text: 'c' }])
    expect(events.map(e => e.t)).toEqual([0, 1, 2])
  })

  it('ignores malformed messages and reports them', () => {
    const { events, errors, sock } = setup()
    sock().open()
    sock().onmessage!({ data: 'not json' })
    sock().push({ hello: 1 })
    expect(events).toHaveLength(0)
    expect(errors).toHaveLength(2)
    expect(() => parseMessage('{"t":1}')).toThrow()
  })

  it('sends Command objects as JSON when open', () => {
    const { c, sock } = setup()
    sock().open()
    c.send({ type: 'add_directive', lens: 'speed', text: 'go faster' })
    expect(JSON.parse(sock().out[0])).toEqual({ type: 'add_directive', lens: 'speed', text: 'go faster' })
  })

  it('reconnects with backoff and flushes queued commands', () => {
    const { c, statuses } = setup()
    const first = FakeSocket.all[0]
    first.open()
    first.drop()
    expect(statuses.at(-1)).toEqual({ state: 'closed', retryIn: 1000, attempt: 1 })
    c.send({ type: 'approve_hybrid', branches: ['prec', 'fast'] })
    vi.advanceTimersByTime(999)
    expect(FakeSocket.all).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(FakeSocket.all).toHaveLength(2)
    FakeSocket.all[1].drop()
    expect(statuses.at(-1)).toEqual({ state: 'closed', retryIn: 2000, attempt: 2 })
    vi.advanceTimersByTime(2000)
    const third = FakeSocket.all[2]
    third.open()
    expect(statuses.at(-1)).toEqual({ state: 'open' })
    expect(third.out.map(o => JSON.parse(o))).toEqual([{ type: 'approve_hybrid', branches: ['prec', 'fast'] }])
    expect(first.out).toHaveLength(0)
    expect(backoff(10)).toBe(8000)
  })

  it('close stops reconnecting', () => {
    const { c } = setup()
    FakeSocket.all[0].open()
    c.close()
    vi.advanceTimersByTime(60000)
    expect(FakeSocket.all).toHaveLength(1)
  })
})
