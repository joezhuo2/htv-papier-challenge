import { useState } from 'react'
import type { Command, Lens } from '@contract/types'
import type { View } from '../state/reducer'
import type { Sent } from '../useSession'
import { clock, lensName } from '../state/derive'
import { MockTag } from './Glyph'

interface Props {
  view: View
  sent: Sent[]
  onSend: (c: Command) => void
  onSelectBranch: (b: string) => void
}

// what happened to a directive this client sent, from the session's own events
export const directiveStatus = (view: View, s: Sent) => {
  if (s.cmd.type !== 'add_directive') return ''
  const { lens, text } = s.cmd
  if (!s.delivered) return 'Recorded locally, not delivered in replay'
  const applied = view.directives.find(d => d.lens === lens && d.text === text)
  if (applied) return `Applied as ${applied.id}`
  const ignored = view.log.find(l => l.text.startsWith(`Ignored a ${lens} directive`) && l.text.includes(text.slice(0, 80)))
  if (ignored) return `Not applied: ${ignored.text.slice(ignored.text.indexOf('(') + 1, ignored.text.indexOf(')'))}`
  return view.finished ? 'Session finished, directive not applied' : 'Sent, waiting for the agent'
}

export function DirectivePanel({ view, sent, onSend, onSelectBranch }: Props) {
  const [text, setText] = useState('')
  const lens: Lens = view.lens
  const list = view.directives.filter(d => d.lens === lens)
  const mine = sent.filter(s => s.cmd.type === 'add_directive' && s.cmd.lens === lens)
  const submit = () => {
    const t = text.trim()
    if (!t || view.finished) return
    onSend({ type: 'add_directive', lens, text: t })
    setText('')
  }
  const holds = view.holds.filter(x => view.directives.find(d => d.id === x.help)?.lens === lens)
  const branchLink = (id: string) => <button className="link" onClick={() => onSelectBranch(id)}>{id}</button>

  return (
    <div className="section">
      <div className="section-head">
        <h2>{lensName(lens)} directives</h2>
        <span className="muted small">{list.length}</span>
      </div>
      {list.length === 0 ? <p className="empty">No directives for this lens yet.</p> : null}
      {list.map(d => {
        const conflict = view.conflicts.find(c => c.directives.includes(d.id))
        return (
          <div key={d.id} className="card" data-testid={`directive-${d.id}`}>
            <p>{d.text}</p>
            <p className="meta">
              <span>{d.id}, {lensName(d.lens)}, {clock(d.t)}</span>
              {d.threshold !== undefined ? <span>bound {d.threshold.toExponential(0)}</span> : null}
              {d.affected ? <span>affected {branchLink(d.affected)}</span> : null}
              <MockTag on={d.mock} />
            </p>
            {conflict?.branch ? (
              <p className="notice" role="note">
                Conflicts with branch {branchLink(conflict.branch)}
                {conflict.forked ? <>, forked {branchLink(conflict.forked)}</> : ', fork pending'}
              </p>
            ) : null}
          </div>
        )
      })}
      {holds.map(x => (
        <div key={x.help} className="card focus" role="alert" data-testid={`prior-hold-${x.help}`}>
          <p className="strong">An earlier session already ran this</p>
          <p className="small">{x.text}</p>
          <div className="inline">
            <button className="primary" disabled={view.finished} onClick={() => onSend({ type: 'confirm_prior', help: x.help, run: true })}>Run anyway</button>
            <button disabled={view.finished} onClick={() => onSend({ type: 'confirm_prior', help: x.help, run: false })}>Skip</button>
            <span className="muted small">{x.help} on {branchLink(x.branch)}</span>
          </div>
        </div>
      ))}
      <form className="stack" onSubmit={e => { e.preventDefault(); submit() }}>
        <label className="label" htmlFor="directive">New directive or hunch</label>
        <textarea
          id="directive" value={text} onChange={e => setText(e.target.value)} disabled={view.finished}
          placeholder={lens === 'speed' ? 'Push throughput harder' : 'Keep max error under 1e-3'}
          onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) submit() }}
        />
        <div className="inline">
          <button type="submit" className="primary" disabled={!text.trim() || view.finished}>Send</button>
          <span className="muted small">{view.finished ? 'Session finished, new directives are not applied.' : `as ${lensName(lens)}`}</span>
        </div>
      </form>
      {mine.map((s, i) => (
        <p key={i} className="small muted">
          {directiveStatus(view, s)}: {s.cmd.type === 'add_directive' ? s.cmd.text : ''}
        </p>
      ))}
    </div>
  )
}
