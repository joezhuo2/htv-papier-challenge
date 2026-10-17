import type { DemoEvent } from '@contract/types'

export interface Chapter { key: string; label: string; index: number; t: number; hypothesis?: string }

// the beats worth jumping to, found from the events themselves: index is how many events to fold to include the beat
export function chapters(events: DemoEvent[]): Chapter[] {
  const out: Chapter[] = []
  const provisional = new Set<string>()
  const add = (key: string, label: string, i: number, hypothesis?: string) => {
    if (!out.some(c => c.key === key)) out.push({ key, label, index: i + 1, t: events[i].t, hypothesis })
  }
  events.forEach((e, i) => {
    if (e.type === 'hypothesis_proposed') add('first', 'First prediction', i, e.hypothesis.id)
    if (e.type === 'verdict' && e.provisional) {
      provisional.add(e.hypothesis)
      add('provisional', `${e.hypothesis} provisional`, i, e.hypothesis)
    }
    if (e.type === 'verdict' && e.verdict === 'rejected' && !e.provisional && provisional.has(e.hypothesis)) add('flip', 'Precision flip', i, e.hypothesis)
    if (e.type === 'post_mortem') add('post_mortem', `Post-mortem ${e.post_mortem.hypothesis}`, i, e.post_mortem.hypothesis)
    if (e.type === 'help_requested') add('help', 'Help request', i)
    if (e.type === 'hybrid_proposed') add('hybrid', 'Hybrid', i)
    if (e.type === 'session_finished') add('end', 'Best confirmed result', i)
  })
  return out
}
