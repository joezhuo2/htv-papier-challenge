import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { parseLog } from '../src/source'
import { fold } from '../src/state/reducer'

// the agent's golden session (tests/fixtures/golden_session.jsonl, kept current by tests/test_golden.py) against the mock contract
const golden = parseLog(readFileSync(join(__dirname, '../../../tests/fixtures/golden_session.jsonl'), 'utf8'))
const keys = (e: object) => Object.keys(e).filter(k => k !== 't' && k !== 'mock').sort()
const OPTIONAL: Partial<Record<DemoEvent['type'], string[]>> = { branch_created: ['reason'], papiers_context: ['query', 'note'], run_finished: ['error'], help_answered: ['note'] }

describe('golden session', () => {
  it('passes log validation and folds into a finished session', () => {
    const v = fold(golden)
    expect(v.started && v.finished).toBe(true)
    expect(v.trail.H1.map(t => t.bound.set_by)).toEqual(['speed', 'precision'])
    expect(v.reads[0].note?.id).toBe('cnv_1')
  })

  it('uses the same payload keys as the mock for every shared event type', () => {
    const strip = (type: DemoEvent['type'], k: string[]) => k.filter(x => !(OPTIONAL[type] ?? []).includes(x))
    const shared = golden.flatMap(e => demoScript.filter(m => m.type === e.type).slice(0, 1).map(twin => [e, twin] as const))
    expect(shared.length).toBeGreaterThan(10)
    expect(shared.map(([e]) => [e.type, strip(e.type, keys(e))])).toEqual(shared.map(([e, twin]) => [e.type, strip(e.type, keys(twin))]))
  })
})
