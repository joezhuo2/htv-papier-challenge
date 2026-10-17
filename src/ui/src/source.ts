import type { DemoEvent } from '@contract/types'

export type Mode = 'mock' | 'replay' | 'live'

export interface Source { mode: Mode; file?: string; ws?: string }

// the judging build never shows mock numbers: mock requests fall back to the recorded replay
export const JUDGING = import.meta.env.VITE_JUDGING === '1'
export const DEFAULT_REPLAY: string = import.meta.env.VITE_REPLAY ?? '/runs/demo.jsonl'

export const parseSource = (search: string, judging = JUDGING): Source => {
  const q = new URLSearchParams(search)
  const m = q.get('mode')
  const mode: Mode = m === 'replay' || m === 'live' ? m : judging ? 'replay' : 'mock'
  return { mode, file: q.get('file') ?? (mode === 'replay' ? DEFAULT_REPLAY : undefined), ws: q.get('ws') ?? undefined }
}

export const MAX_GAP_MS = 4000

// required payload keys per event type, checked on load so a bad log gives a clear error instead of a blank screen
export const REQUIRED: Record<DemoEvent['type'], string[]> = {
  session_started: ['baselines', 'branch'], papiers_context: [], directive_added: ['directive'], conflict_detected: ['directives', 'reason'],
  branch_created: ['branch'], branch_status: ['branch', 'status'], hypothesis_proposed: ['hypothesis'], run_started: ['run', 'branch', 'hypothesis'],
  run_finished: ['run'], verdict: ['hypothesis', 'verdict', 'provisional', 'reason', 'checks', 'bound'], post_mortem: ['post_mortem'],
  summary: ['summary'], help_requested: ['help'], help_answered: ['help', 'option', 'by'], plan_change: ['help', 'text'], agent_log: ['text'],
  findings_exported: ['items'], report_ready: ['report'], hybrid_proposed: ['branches', 'text'], session_finished: ['ledger', 'best'],
  agent_error: ['kind', 'text'],
}

export const parseLog = (text: string): DemoEvent[] => {
  const out: DemoEvent[] = []
  text.split('\n').forEach((line, i) => {
    if (!line.trim()) return
    let e: Record<string, unknown>
    try {
      e = JSON.parse(line)
    } catch {
      throw new Error(`Line ${i + 1} is not valid JSON`)
    }
    if (typeof e.t !== 'number' || typeof e.type !== 'string') throw new Error(`Line ${i + 1} has no numeric t or no type`)
    const need = REQUIRED[e.type as DemoEvent['type']]
    if (!need) throw new Error(`Line ${i + 1} has an unknown event type "${e.type}"`)
    const missing = need.filter(k => !(k in e))
    if (missing.length) throw new Error(`Line ${i + 1}: ${e.type} is missing ${missing.join(', ')}`)
    out.push(e as unknown as DemoEvent)
  })
  if (!out.length) throw new Error('The log is empty')
  if (out[0].type !== 'session_started' && !out.some(e => e.type === 'session_started')) throw new Error('The log has no session_started event')
  return out
}

// shortens idle gaps longer than maxGap so a recorded run replays at demo pace, order is untouched
export const compress = (events: DemoEvent[], maxGap = MAX_GAP_MS): DemoEvent[] => {
  let shift = 0
  return events.map((e, i) => {
    if (i > 0) shift += Math.max(0, e.t - events[i - 1].t - maxGap)
    return { ...e, t: e.t - shift }
  })
}

export const loadLog = async (file: string) => {
  const res = await fetch(file)
  if (!res.ok) throw new Error(`Could not load ${file} (${res.status})`)
  return parseLog(await res.text())
}
