import { describe, expect, it } from 'vitest'
import { demoScript } from '@contract/demo-script'
import { compress, parseLog, parseSource } from '../src/source'

describe('source', () => {
  it('parses the mode query', () => {
    expect(parseSource('')).toEqual({ mode: 'mock', file: undefined, ws: undefined })
    expect(parseSource('?mode=replay&file=/runs/a.jsonl')).toMatchObject({ mode: 'replay', file: '/runs/a.jsonl' })
    expect(parseSource('?mode=live&ws=ws://x/ws')).toMatchObject({ mode: 'live', ws: 'ws://x/ws' })
    expect(parseSource('?mode=nonsense').mode).toBe('mock')
  })

  it('parses JSONL and skips blank lines', () => {
    const text = demoScript.slice(0, 3).map(e => JSON.stringify(e)).join('\n') + '\n\n'
    expect(parseLog(text)).toEqual(demoScript.slice(0, 3))
  })

  it('compresses idle gaps without dropping or reordering', () => {
    const out = compress(demoScript, 4000)
    expect(out.map(e => e.type)).toEqual(demoScript.map(e => e.type))
    out.slice(1).forEach((e, i) => {
      expect(e.t - out[i].t).toBeGreaterThanOrEqual(0)
      expect(e.t - out[i].t).toBeLessThanOrEqual(4000)
    })
    expect(out.at(-1)!.t).toBeLessThan(demoScript.at(-1)!.t)
  })
})
