import { afterEach, describe, expect, it, vi } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { demoScript } from '@contract/demo-script'
import { compress, loadDefault, parseLog, parseSource } from '../src/source'

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

  describe('default recorded run', () => {
    afterEach(() => vi.unstubAllGlobals())
    const log = demoScript.slice(0, 3).map(e => JSON.stringify(e)).join('\n')
    const serve = (files: Record<string, string>) => {
      const fetch = vi.fn(async (f: string) => (f in files ? new Response(files[f]) : new Response('not found', { status: 404 })))
      vi.stubGlobal('fetch', fetch)
      return fetch
    }

    it('prefers demo.jsonl', async () => {
      serve({ '/demo/replay.jsonl': log, '/runs/fallback1.jsonl': log })
      expect((await loadDefault()).file).toBe('/demo/replay.jsonl')
    })

    it('falls back to fallback1, then fallback2 when one is invalid, and stops at the first missing number', async () => {
      const fetch = serve({ '/runs/fallback1.jsonl': '{"broken": true}', '/runs/fallback2.jsonl': log, '/runs/fallback4.jsonl': log })
      expect((await loadDefault()).file).toBe('/runs/fallback2.jsonl')
      serve({ '/runs/fallback2.jsonl': log })
      await expect(loadDefault()).rejects.toThrow('No recorded run could be loaded, tried /demo/replay.jsonl (404), /runs/fallback1.jsonl (404)')
      expect(fetch).not.toHaveBeenCalledWith('/runs/fallback3.jsonl')
    })
  })
  it('parses the frozen demo replays', () => {
    for (const f of ['replay.jsonl', 'fallback-1541.jsonl']) {
      const events = parseLog(readFileSync(join(__dirname, '../../../demo', f), 'utf8'))
      expect(events.at(-1)?.type).toBe('session_finished')
      expect(events.some(e => e.type === 'final_report')).toBe(true)
    }
  })
})
