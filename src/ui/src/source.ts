import type { DemoEvent } from '@contract/types'

export type Mode = 'mock' | 'replay' | 'live'

export interface Source { mode: Mode; file?: string; ws?: string }

export const parseSource = (search: string): Source => {
  const q = new URLSearchParams(search)
  const m = q.get('mode')
  const mode: Mode = m === 'replay' || m === 'live' ? m : 'mock'
  return { mode, file: q.get('file') ?? undefined, ws: q.get('ws') ?? undefined }
}

export const MAX_GAP_MS = 4000

export const parseLog = (text: string): DemoEvent[] =>
  text.split('\n').filter(l => l.trim()).map(l => JSON.parse(l) as DemoEvent)

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
