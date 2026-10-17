import { useState } from 'react'
import type { HypothesisView, View } from '../state/reducer'
import { clock, papiersNote } from '../state/derive'
import { MockTag } from './Glyph'

interface Item { id: string; kind: string; title: string; excerpt: string }

const anchor = (id: string) => `papiers-${id}`

// scrolls the left column to a note and replays its highlight
export function showNote(id: string) {
  const el = document.getElementById(anchor(id))
  const box = el?.closest('.col') as HTMLElement | null
  if (!el || !box) return
  box.scrollTop = el.offsetTop - 12
  el.classList.remove('flash')
  void el.offsetWidth
  el.classList.add('flash')
}

export function InformedBy({ view, h }: { view: View; h: HypothesisView }) {
  const id = papiersNote(h)
  if (!id) return null
  const note = view.notes.find(n => n.id === id)
  return (
    <button className="informed" onClick={() => showNote(id)} data-testid="informed-by" title={note?.excerpt}>
      <span className="strong">Informed by note <span className="mono">{id}</span></span>
      {note ? <span className="oneline muted">{note.excerpt}</span> : null}
    </button>
  )
}

export function PapiersPanel({ view }: { view: View }) {
  const [q, setQ] = useState('')
  const [items, setItems] = useState<Item[]>()
  const [state, setState] = useState<'idle' | 'loading' | string>('idle')

  const search = async () => {
    if (!q.trim()) return
    setState('loading')
    const res = await fetch(`/papiers/search?q=${encodeURIComponent(q.trim())}`)
    const body = await res.json()
    if (!res.ok) return setState(body.error ?? `search failed (${res.status})`)
    setItems(body.results)
    setState('idle')
  }

  return (
    <div className="section" data-testid="papiers-panel">
      <div className="section-head">
        <h2>Papiers</h2>
        <span className="tag">read-only</span>
      </div>
      <p className="small muted">Read by the agent this session</p>
      {view.reads.length === 0 ? <p className="empty">Nothing read yet.</p> : null}
      {view.reads.map((r, i) => {
        const first = r.note && view.reads.findIndex(x => x.note?.id === r.note!.id) === i
        return (
          <div key={i} className="card flash" id={first ? anchor(r.note!.id) : undefined} data-testid="papiers-read" title={r.note?.excerpt}>
            <p className="meta"><span>{clock(r.t)}</span>{r.query ? <span>query "{r.query}"</span> : null}<MockTag on={r.mock} /></p>
            {r.note ? (
              <>
                <div className="row"><span className="strong small">{r.note.title}</span><span className="mono small muted">{r.note.id}</span></div>
                <p className="small clamp">{r.note.excerpt}</p>
              </>
            ) : <p className="small muted">No matching notes.</p>}
          </div>
        )
      })}
      <form className="stack" onSubmit={e => { e.preventDefault(); search() }}>
        <label className="label" htmlFor="papiers-q">Search your library</label>
        <div className="inline">
          <input id="papiers-q" value={q} onChange={e => setQ(e.target.value)} placeholder="FP16 accumulation" />
          <button type="submit" disabled={!q.trim() || state === 'loading'}>Search</button>
        </div>
      </form>
      {state === 'loading' ? <p className="small muted">Searching Papiers...</p> : null}
      {state !== 'idle' && state !== 'loading' ? <p className="small" role="alert">{state}</p> : null}
      {items && items.length === 0 ? <p className="empty">No matching notes.</p> : null}
      {items?.map(r => (
        <div key={r.id} className="card">
          <div className="row"><span className="strong small">{r.title}</span><span className="tag">{r.kind}</span></div>
          <p className="small clamp">{r.excerpt}</p>
          <p className="meta"><span className="mono">{r.id}</span></p>
        </div>
      ))}
    </div>
  )
}
