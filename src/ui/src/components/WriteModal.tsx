import { useEffect, useState } from 'react'
import type { WriteResult } from '@contract/types'
import type { Mode } from '../source'

interface Item { id: string; verdict: string; doc: string; pages: string | null; passage: string; note: string; color: string; written: boolean }
interface Skip { id: string; verdict: string; reason: string }
interface Preview { session: string; pending: Item[]; skipped: Skip[]; apply: boolean; reason: string | null }

const STATUS: Record<WriteResult['status'], string> = { written: 'written', already_written: 'already written', failed: 'failed' }

const detail = (body: { detail?: unknown }, fallback: string) => (typeof body.detail === 'string' ? body.detail : fallback)

// the write-back preview of a finished run: nothing reaches Papiers until the checked items are applied
export function WriteModal({ mode, file, onClose }: { mode: Mode; file?: string; onClose: () => void }) {
  const [data, setData] = useState<Preview>()
  const [error, setError] = useState<string>()
  const [checked, setChecked] = useState<string[]>([])
  const [results, setResults] = useState<WriteResult[]>()
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    if (mode === 'mock') return
    const url = mode === 'replay' && file ? `/papiers/writes?file=${encodeURIComponent(file)}` : '/papiers/writes'
    fetch(url)
      .then(async r => {
        const body = await r.json()
        if (!r.ok) throw new Error(detail(body, `the preview failed (${r.status})`))
        setData(body)
      })
      .catch((e: Error) => setError(`Could not load the preview from the agent server: ${e.message}`))
  }, [mode, file])

  const apply = async () => {
    setBusy(true)
    setError(undefined)
    const r = await fetch('/papiers/writes', {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ session: data!.session, ids: checked }),
    })
    const body = await r.json()
    setBusy(false)
    if (!r.ok) return setError(detail(body, `Apply failed (${r.status})`))
    setResults(body.results)
  }

  const items = data?.pending ?? []
  const why = mode === 'mock' ? 'Mock data never writes to Papiers.' : data && !data.apply ? data.reason : undefined
  const result = (id: string) => results?.find(r => r.id === id)
  const toggle = (id: string) => setChecked(c => (c.includes(id) ? c.filter(x => x !== id) : [...c, id]))
  const all = items.length > 0 && checked.length === items.length

  return (
    <div className="modal-backdrop" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal card" role="dialog" aria-modal="true" aria-labelledby="write-title" data-testid="write-modal">
        <div className="row">
          <h2 id="write-title">Write to Papiers</h2>
          <span className="spacer" />
          <button onClick={onClose}>Close</button>
        </div>
        <p className="small muted">
          Each final verdict that cites a Papiers passage becomes a highlight on that passage, with an [RWR] note and its verdict colour.
          This is a dry run: nothing is sent until you check items and press Apply. The Markdown export is unchanged.
        </p>
        {mode === 'mock' ? <p className="empty">Mock data has no Papiers passages, so there is nothing to preview.</p> : null}
        {error ? <p className="small" role="alert">{error}</p> : null}
        {mode !== 'mock' && !data && !error ? <p className="small muted">Loading the preview...</p> : null}
        {data ? (
          <>
            {items.length === 0 ? <p className="empty">No verdict cites a passage that can be written.</p> : (
              <>
                <label className="inline small">
                  <input type="checkbox" checked={all} onChange={() => setChecked(all ? [] : items.map(x => x.id))} /> Select all
                </label>
                <ul className="plain write-list" data-testid="write-pending">
                  {items.map(it => {
                    const r = result(it.id)
                    return (
                      <li key={it.id} className="write-item" data-testid="write-item">
                        <label className="inline">
                          <input type="checkbox" checked={checked.includes(it.id)} onChange={() => toggle(it.id)} aria-label={`Write ${it.id}`} />
                          <span className="mono strong">{it.id}</span>
                          <span className="swatch-chip small"><span className="swatch" style={{ background: it.color }} />{it.verdict}</span>
                          <span className="small">{it.doc}{it.pages ? `, page ${it.pages}` : ', webpage'}</span>
                          {it.written && !r ? <span className="tag">already written</span> : null}
                        </label>
                        <blockquote className="small clamp" style={{ margin: '2px 0' }}>{it.passage}</blockquote>
                        <p className="mono small muted" style={{ margin: 0 }}>{it.note}</p>
                        {r ? (
                          <p className={`small${r.status === 'failed' ? ' strong' : ''}`} data-testid="write-result">
                            {STATUS[r.status]}{r.error ? `: ${r.error}` : ''}
                          </p>
                        ) : null}
                      </li>
                    )
                  })}
                </ul>
              </>
            )}
            <div className="inline">
              <button className="primary" onClick={apply} disabled={!!why || !checked.length || busy} data-testid="write-apply">
                {busy ? 'Applying...' : `Apply ${checked.length} checked`}
              </button>
              {why ? <span className="small muted" data-testid="write-disabled">{why}</span> : null}
            </div>
            <details data-testid="write-skipped">
              <summary className="small">Skipped ({data.skipped.length})</summary>
              {data.skipped.length ? (
                <ul className="plain small">
                  {data.skipped.map(x => <li key={x.id}><span className="mono">{x.id}</span> {x.verdict}: {x.reason}</li>)}
                </ul>
              ) : <p className="empty">Nothing was skipped.</p>}
            </details>
          </>
        ) : null}
      </div>
    </div>
  )
}
