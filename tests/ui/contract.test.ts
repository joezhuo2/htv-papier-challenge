import { describe, expect, it } from 'vitest'
import { demoScript, DEMO_DURATION_MS } from '../../src/contract/demo-script'

describe('demo script', () => {
  it('is ordered and fits the timeline', () => {
    const ts = demoScript.map(e => e.t)
    expect(ts).toEqual([...ts].sort((a, b) => a - b))
    expect(ts.at(-1)).toBeLessThan(DEMO_DURATION_MS)
  })
})
