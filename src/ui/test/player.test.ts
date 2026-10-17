import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { Player } from '../src/player'

const ev = (t: number, text = String(t)): DemoEvent => ({ t, type: 'agent_log', text })
const events = [ev(0), ev(1000), ev(1000, 'same'), ev(2500), ev(4000)]
const texts = (got: DemoEvent[]) => got.map(e => (e as { text: string }).text)

describe('player', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('fires events at t / speed', () => {
    const got: DemoEvent[] = []
    const p = new Player(events, e => got.push(e), 2)
    p.play()
    vi.advanceTimersByTime(0)
    expect(texts(got)).toEqual(['0'])
    vi.advanceTimersByTime(499)
    expect(got).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(texts(got)).toEqual(['0', '1000', 'same'])
    vi.advanceTimersByTime(750)
    expect(got).toHaveLength(4)
    vi.advanceTimersByTime(750)
    expect(got).toHaveLength(5)
    expect(p.ended).toBe(true)
  })

  it('pause and resume keep the timeline position', () => {
    const got: DemoEvent[] = []
    const p = new Player(events, e => got.push(e))
    p.play()
    vi.advanceTimersByTime(900)
    p.pause()
    expect(p.position()).toBe(900)
    vi.advanceTimersByTime(10000)
    expect(got).toHaveLength(1)
    p.play()
    vi.advanceTimersByTime(99)
    expect(got).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(got).toHaveLength(3)
  })

  it('step emits exactly the next event and stays paused', () => {
    const got: DemoEvent[] = []
    const p = new Player(events, e => got.push(e))
    p.step()
    p.step()
    expect(texts(got)).toEqual(['0', '1000'])
    expect(p.paused).toBe(true)
    expect(p.position()).toBe(1000)
    vi.advanceTimersByTime(10000)
    expect(got).toHaveLength(2)
    p.play()
    vi.advanceTimersByTime(0)
    expect(texts(got)).toEqual(['0', '1000', 'same'])
  })

  it('never drops or reorders under mixed controls', () => {
    const got: DemoEvent[] = []
    const p = new Player(demoScript, e => got.push(e))
    const ops = [() => p.play(), () => p.pause(), () => p.step(), () => p.setSpeed(4), () => p.setSpeed(1), () => p.play()]
    let i = 0
    while (!p.ended) {
      ops[i++ % ops.length]()
      vi.advanceTimersByTime(1700)
    }
    expect(got).toEqual(demoScript)
  })

  it('speed changes rescale the remaining wait', () => {
    const got: DemoEvent[] = []
    const p = new Player([ev(0), ev(4000)], e => got.push(e))
    p.play()
    vi.advanceTimersByTime(1000)
    p.setSpeed(4)
    vi.advanceTimersByTime(749)
    expect(got).toHaveLength(1)
    vi.advanceTimersByTime(1)
    expect(got).toHaveLength(2)
  })

  it('cancel stops pending events', () => {
    const got: DemoEvent[] = []
    const p = new Player(events, e => got.push(e))
    p.play()
    vi.advanceTimersByTime(1000)
    p.cancel()
    vi.advanceTimersByTime(10000)
    p.play()
    p.step()
    vi.advanceTimersByTime(10000)
    expect(texts(got)).toEqual(['0', '1000', 'same'])
  })

  it('cancel from inside a handler stops the rest of a batch', () => {
    const got: DemoEvent[] = []
    const p: Player = new Player(events, e => {
      got.push(e)
      if (got.length === 2) p.cancel()
    })
    p.play()
    vi.advanceTimersByTime(5000)
    expect(texts(got)).toEqual(['0', '1000'])
  })
})
