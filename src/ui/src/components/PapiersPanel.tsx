import { useState } from 'react'
import type { HypothesisView, NoteView, View } from '../state/reducer'
import { chipLabel, cleanNote, clock, kindName, noteKind, papiersNote, untex } from '../state/derive'
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

const NOTE_ID = /\b((?:cnv|hl|nte|psg|pas)_[0-9a-zA-Z]{4,})\b/

export function NoteChip({ view, id }: { view: View; id: string }) {
  const found = view.notes.find(n => n.id === id)
  const kind = noteKind(id, found?.kind)
  // highlights, notes and conversations look different, so the user's own marks stand out from retrieved context
  return (
    <button type="button" className={`tag note-chip chip-${kind}`} onClick={() => showNote(id)} title={id} data-testid="note-chip" data-kind={kind}>
      {chipLabel(kind)}{found ? `: ${found.document_title || cleanNote(found).title}` : ''}
    </button>
  )
}

// a retrieved item: the document title, the highlighted text, and the user's annotation labelled as their own words
export function NoteBody({ note }: { note: NoteView }) {
  const text = untex(note.highlight_text ?? cleanNote(note).excerpt)
  return (
    <div className="stack note-body" style={{ gap: 2 }} data-testid="note-body">
      <span className="small"><span className="tag">{kindName(note.kind)}</span> <span className="strong">{note.document_title || cleanNote(note).title}</span></span>
      {text ? <blockquote className="small clamp" style={{ margin: 0 }}>{text}</blockquote> : null}
      {note.annotation ? <p className="small" data-testid="your-note"><span className="muted">your note:</span> {untex(note.annotation)}</p> : null}
    </div>
  )
}

// text with raw Papiers ids replaced by note chips, so no cnv_ id is shown as prose
export function NoteText({ view, text }: { view: View; text: string }) {
  const parts = text.split(NOTE_ID)
  return <>{parts.map((p, i) => (i % 2 ? <NoteChip key={i} view={view} id={p} /> : p.replace(/\s*Context from Papiers note\s*$/, '')))}</>
}

export function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false)
  const copy = () => navigator.clipboard?.writeText(id).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
  return (
    <button type="button" className="link mono small copy-id" onClick={copy} title="Copy the Papiers id" data-testid="copy-id">
      {id} <span className="muted">{copied ? 'copied' : 'copy'}</span>
    </button>
  )
}

export function InformedBy({ view, h }: { view: View; h: HypothesisView }) {
  const id = papiersNote(h)
  if (!id) return null
  const found = view.notes.find(n => n.id === id)
  const note = found && cleanNote(found)
  return (
    <button className="informed" onClick={() => showNote(id)} data-testid="informed-by" title={note?.excerpt}>
      <span className="strong">Informed by {found?.kind ? kindName(found.kind) : 'Papiers note'}</span>
      {found ? <span className="oneline muted">{found.document_title || note!.title}</span> : null}
      {found?.annotation ? <span className="oneline small">your note: {untex(found.annotation)}</span> : null}
    </button>
  )
}

const fields = (d?: Record<string, string | number>) => (d ? Object.entries(d).map(([k, v]) => `${k}=${v}`).join(' ') : 'old note, no fingerprint')

// the verdicts earlier sessions wrote, as the agent was shown them
function PriorTable({ prior }: { prior: NonNullable<View['prior']> }) {
  return (
    <div data-testid="prior-table">
      {prior.items.map(p => (
        <div key={p.id} className="card" data-testid="prior-row">
          <div className="row"><span className="strong small">{p.fp?.title || p.reason}</span><span className="tag">{p.verdict}</span></div>
          <p className="small">{fields(p.fp?.params)}</p>
          <p className="meta"><span>{p.measured} vs {p.bound}</span>{p.fp ? <span>bounds std {p.fp.bounds.std} adv {p.fp.bounds.adv}</span> : null}<span className="mono">{p.run}</span></p>
        </div>
      ))}
      {prior.section ? (
        <details>
          <summary className="small muted">Exactly what the agent was sent</summary>
          <pre className="small" data-testid="prior-section">{prior.section}</pre>
        </details>
      ) : null}
    </div>
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
        <span className="tag">read, write on approval</span>
      </div>
      <p className="small muted">Read by the agent this session</p>
      {view.prior ? (
        <>
          <p className="small muted" data-testid="prior-count" title={view.prior.error ?? view.prior.items.map(p => `${p.run}: ${p.verdict}`).join('\n')}>
            Prior verdicts read: {view.prior.count}{view.prior.error ? ` (${view.prior.error})` : ''}{view.prior.ignored ? ' (ignored for this session)' : ''}
          </p>
          {view.prior.items.length ? <PriorTable prior={view.prior} /> : null}
        </>
      ) : null}
      {view.reads.length === 0 ? <p className="empty">Nothing read yet.</p> : null}
      {view.reads.map((r, i) => {
        const first = r.note && view.reads.findIndex(x => x.note?.id === r.note!.id) === i
        const note = r.note && cleanNote(r.note)
        return (
          <div key={i} className="card flash" id={first ? anchor(r.note!.id) : undefined} data-testid="papiers-read" title={note?.excerpt}>
            <p className="meta"><span>{clock(r.t)}</span>{r.query ? <span>query "{r.query}"</span> : null}<MockTag on={r.mock} /></p>
            {r.note ? (
              <>
                <NoteBody note={r.note} />
                <CopyId id={r.note.id} />
              </>
            ) : <p className="small muted">No matching note.</p>}
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
      {items && items.length === 0 ? <p className="empty">No matching note.</p> : null}
      {items?.map(r => {
        const note = cleanNote(r)
        return (
          <div key={r.id} className="card">
            <div className="row"><span className="strong small">{note.title}</span><span className="tag">{r.kind}</span></div>
            <p className="small clamp">{note.excerpt}</p>
            <CopyId id={r.id} />
          </div>
        )
      })}
    </div>
  )
}
