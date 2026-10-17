import { useState } from 'react'
import type { Command } from '@contract/types'
import type { HelpView, View } from '../state/reducer'
import type { Sent } from '../useSession'
import { clock, lensName } from '../state/derive'
import { Lock, MockTag } from './Glyph'

interface Props {
  view: View
  hv: HelpView
  now: number
  sent: Sent[]
  onSend: (c: Command) => void
}

export function HelpCard({ view, hv, now, sent, onSend }: Props) {
  const { help } = hv
  const [option, setOption] = useState(help.default)
  const [note, setNote] = useState('')
  const label = (id: string) => help.options.find(o => o.id === id)?.label ?? id
  const left = hv.t + help.deadline_s * 1000 - now
  const expired = !hv.answer && (hv.expired || left <= 0)
  const meanwhile = view.branches.filter(b => b.status === 'running' && b.id !== help.blocks).map(b => b.id)
  const mine = sent.find(s => s.cmd.type === 'answer_help' && s.cmd.help === help.id)

  if (hv.answer) {
    return (
      <article className="card" data-testid={`help-${help.id}`}>
        <div className="row"><h3>Help request <span className="mono">{help.id}</span></h3><span className="tag">answered</span><MockTag on={hv.mock} /></div>
        <p className="small">{help.question}</p>
        <p className="small"><span className="muted">{lensName(hv.answer.by)} answered at {clock(hv.answer.t)}:</span> {label(hv.answer.option)}{hv.answer.note ? `. "${hv.answer.note}"` : ''}</p>
      </article>
    )
  }

  if (hv.cancelled) {
    return (
      <article className="card" data-testid={`help-${help.id}`}>
        <div className="row"><h3>Help request <span className="mono">{help.id}</span></h3><span className="tag">withdrawn</span><MockTag on={hv.mock} /></div>
        <p className="small">{help.question}</p>
        <p className="small muted">{hv.cancelled}</p>
      </article>
    )
  }

  if (view.finished) {
    return (
      <article className="card" data-testid={`help-${help.id}`}>
        <div className="row"><h3>Help request <span className="mono">{help.id}</span></h3><span className="tag">closed</span><MockTag on={hv.mock} /></div>
        <p className="small">{help.question}</p>
        <p className="small"><span className="muted">Session ended without an answer, proceeded on:</span> {label(help.default)}</p>
      </article>
    )
  }

  return (
    <article className="card focus" data-testid={`help-${help.id}`} aria-label={`help request for ${lensName(help.target)}`}>
      <div className="row">
        <span className="inline"><Lock /><h3>Help request <span className="mono">{help.id}</span></h3></span>
        <MockTag on={hv.mock} />
        <span style={{ flex: 1 }} />
        <span className="countdown mono" data-testid="countdown">{expired ? 'deadline passed' : `${Math.ceil(left / 1000)}s left`}</span>
      </div>
      <p>{help.question}</p>
      <p className="meta" data-testid="help-status">
        <span>{view.branches.find(b => b.id === help.blocks)?.status === 'running' ? 'Applies to' : 'Blocks'} <span className="mono">{help.blocks}</span></span>
        <span>meanwhile {meanwhile.length ? meanwhile.map(b => <span key={b} className="mono">{b} </span>) : 'nothing else runs'}</span>
        <span>default: {label(help.default)}</span>
      </p>
      {expired ? <p className="strong" role="status">Proceeding on assumption: {label(help.default)}</p> : null}
      <form
        className="stack"
        onSubmit={e => {
          e.preventDefault()
          onSend({ type: 'answer_help', help: help.id, option, by: view.lens, ...(note.trim() ? { note: note.trim() } : {}) })
        }}
      >
        <div className="options" role="radiogroup" aria-label="answer">
          {help.options.map(o => (
            <label key={o.id}>
              <input type="radio" name={`opt-${help.id}`} checked={option === o.id} onChange={() => setOption(o.id)} style={{ width: 'auto' }} />
              {o.label}{o.id === help.default ? <span className="muted small"> (default)</span> : null}
            </label>
          ))}
        </div>
        <input aria-label="answer note" value={note} onChange={e => setNote(e.target.value)} placeholder="Optional note or hunch" />
        <div className="inline">
          <button type="submit" className="primary">Send answer</button>
          {mine ? <span className="small muted">{mine.delivered ? 'Sent, waiting for the agent' : 'Recorded locally, not delivered in replay'}</span> : null}
        </div>
      </form>
      <details className="small">
        <summary className="muted">What changes and evidence ({help.evidence.length})</summary>
        <p style={{ marginTop: 4 }}>{help.what_changes}</p>
        <ul className="plain">{help.evidence.map((e, i) => <li key={i}>{e}</li>)}</ul>
      </details>
    </article>
  )
}
