import type { PostMortem } from '@contract/types'
import type { View } from '../state/reducer'
import { ruledOut, sci } from '../state/derive'
import { NoteBody, NoteChip } from './PapiersPanel'

function Sweep({ sweep }: { sweep: PostMortem['sweep'] }) {
  const w = 320, h = 150, m = { l: 48, r: 16, t: 12, b: 28 }
  const bound = 1e-3
  const lk = sweep.map(s => Math.log2(s.k))
  const le = [...sweep.map(s => Math.log10(s.adv_error)), Math.log10(bound)]
  const [k0, k1] = [Math.min(...lk), Math.max(...lk)]
  const [e0, e1] = [Math.floor(Math.min(...le)), Math.ceil(Math.max(...le))]
  const x = (k: number) => m.l + ((Math.log2(k) - k0) / (k1 - k0 || 1)) * (w - m.l - m.r)
  const y = (e: number) => m.t + ((e1 - Math.log10(e)) / (e1 - e0 || 1)) * (h - m.t - m.b)
  const decades = Array.from({ length: e1 - e0 + 1 }, (_, i) => e0 + i)
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label="adversarial error versus K" data-testid="sweep">
      {decades.map(d => <text key={d} x={m.l - 6} y={y(10 ** d) + 4} fontSize="11" textAnchor="end" fill="var(--muted)">1e{d}</text>)}
      <line x1={m.l} x2={w - m.r} y1={y(bound)} y2={y(bound)} stroke="var(--ink)" strokeDasharray="4 4" />
      <text x={w - m.r} y={y(bound) - 4} fontSize="11" textAnchor="end">bound 1e-3</text>
      <polyline fill="none" stroke="var(--rejected)" strokeWidth="1.5" points={sweep.map(s => `${x(s.k)},${y(s.adv_error)}`).join(' ')} />
      {sweep.map(s => (
        <g key={s.k}>
          <circle cx={x(s.k)} cy={y(s.adv_error)} r="3" fill="var(--rejected)" />
          <text x={x(s.k)} y={h - 8} fontSize="11" textAnchor="middle" className="mono">{s.k}</text>
        </g>
      ))}
      <text x={m.l - 24} y={h - 8} fontSize="11" textAnchor="end" fill="var(--muted)">K</text>
    </svg>
  )
}

export function PostMortemCard({ view, pm, open, onToggle }: { view: View; pm: PostMortem; open: boolean; onToggle: (open: boolean) => void }) {
  const worst = pm.sweep.length ? Math.max(...pm.sweep.map(s => s.adv_error)) : undefined
  const verdict = view.verdicts[pm.hypothesis]?.verdict === 'inconclusive' ? 'inconclusive' : 'rejected'
  return (
    <details
      className="card focus" open={open} id={`pm-${pm.hypothesis}`} data-testid={`postmortem-${pm.hypothesis}`}
      onToggle={e => onToggle((e.target as HTMLDetailsElement).open)}
    >
      <summary className="row" style={{ cursor: 'pointer' }}>
        <h3>Post-mortem <span className="mono">{pm.hypothesis}</span></h3>
        <span className={`verdict ${verdict}`}>{verdict}</span>
        {worst ? <span className="small muted">worst adversarial {sci(worst)}</span> : null}
      </summary>
      <div className="stack" style={{ paddingTop: 6 }}>
        <dl>
          <dt>Predicted</dt><dd>{pm.expected}</dd>
          <dt>Happened</dt><dd>{pm.observed}</dd>
        </dl>
        {pm.sweep.length ? <Sweep sweep={pm.sweep} /> : null}
        <dl>
          <dt>Mechanism</dt><dd>{pm.cause}</dd>
          <dt>Will not retry</dt><dd>{ruledOut(view, pm.hypothesis)}</dd>
          <dt>Pivot</dt><dd>{pm.pivot}</dd>
        </dl>
        {pm.note ? (() => {
          const found = view.notes.find(n => n.id === pm.note)
          return (
            <div className="stack" style={{ gap: 4 }} data-testid="pm-note">
              <span className="small"><span className="muted">Read from Papiers for this cause:</span> <NoteChip view={view} id={pm.note} /></span>
              {found ? <NoteBody note={found} /> : null}
            </div>
          )
        })() : null}
      </div>
    </details>
  )
}
