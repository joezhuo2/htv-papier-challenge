import type { Command, Params } from '@contract/types'
import type { FeedItem, View } from '../state/reducer'
import { clock, ledgerRows, ms, predictionText, runFor, sci, times } from '../state/derive'
import { useSize } from '../useSize'
import { MockTag, VerdictChip } from './Glyph'

const bestParams = (v: View, branch: string) => {
  const runs = v.runs.filter(r => r.branch === branch && r.metrics && v.verdicts[r.hypothesis]?.verdict === 'confirmed')
  return runs.sort((a, b) => a.metrics!.latency_ms - b.metrics!.latency_ms)[0]
}

interface HybridProps {
  view: View
  item: Extract<FeedItem, { kind: 'hybrid' }>
  onSend: (c: Command) => void
  onDecide: (approved: boolean) => void
}

export function HybridCard({ view, item, onSend, onDecide }: HybridProps) {
  const runs = item.branches.map(b => ({ b, r: bestParams(view, b) }))
  const keys = Object.keys(runs.find(x => x.r?.params)?.r?.params ?? {}) as (keyof Params)[]
  const state = view.hybrid
  const merged = view.branches.find(b => item.branches.every(x => b.parents.includes(x)))
  const h = merged && view.hypotheses.find(x => x.branch === merged.id)
  const r = h && runFor(view, h.id)

  return (
    <article className="card" data-testid="hybrid-card">
      <div className="row">
        <h3>Hybrid proposal</h3>
        {item.branches.map(b => <span key={b} className="tag mono">{b}</span>)}
        <MockTag on={item.mock} />
        <span style={{ flex: 1 }} />
        <span className="mono small muted">{clock(item.t)}</span>
      </div>
      <p>{item.text}</p>
      {keys.length ? (
        <table className="table">
          <thead><tr><th>param</th>{runs.map(x => <th key={x.b}><span className="mono">{x.b}</span> {x.r ? `(${x.r.id})` : ''}</th>)}</tr></thead>
          <tbody>
            {keys.map(k => {
              const vals = runs.map(x => x.r?.params?.[k])
              const differs = new Set(vals).size > 1
              return (
                <tr key={k}>
                  <td className="mono">{k}{differs ? ' *' : ''}</td>
                  {vals.map((v, i) => <td key={i} className={`mono${differs ? ' strong' : ''}`}>{String(v ?? '')}</td>)}
                </tr>
              )
            })}
          </tbody>
        </table>
      ) : null}
      {keys.length ? <p className="small muted">* differs between the branches, the hybrid takes one value from each lens.</p> : null}
      {h ? <p className="small"><span className="muted">Predicted:</span> {predictionText(view, h)}</p> : null}
      {r?.metrics ? <p className="small"><span className="muted">Measured:</span> {ms(r.metrics.latency_ms)}, adversarial {sci(r.metrics.adv_error)} <VerdictChip verdict={view.verdicts[h!.id]?.verdict} provisional={view.verdicts[h!.id]?.provisional} /></p> : null}
      {state?.approved || merged ? (
        <p className="small strong">Approved{merged ? <>, branch <span className="mono">{merged.id}</span> created</> : ', waiting for the agent'}</p>
      ) : state?.dismissed ? (
        <p className="small muted">Dismissed locally. <button className="link" onClick={() => onDecide(true)}>Undo</button></p>
      ) : (
        <div className="inline">
          <button className="primary" onClick={() => { onSend({ type: 'approve_hybrid', branches: item.branches }); onDecide(true) }}>Approve</button>
          <button onClick={() => onDecide(false)}>Dismiss</button>
        </div>
      )}
    </article>
  )
}

export function Ledger({ view }: { view: View }) {
  const rows = ledgerRows(view)
  return (
    <details className="ledger" open data-testid="ledger">
      <summary className="section-head" style={{ cursor: 'pointer' }}>
        <h2>Hypothesis ledger</h2><span className="muted small">{rows.length}{view.ledger ? ', final' : ''}</span>
      </summary>
      {rows.length === 0 ? <p className="empty">No hypotheses yet.</p> : (
        <table className="table">
          <thead><tr><th>id</th><th>predicted, result</th><th>verdict</th><th>branch</th></tr></thead>
          <tbody>
            {rows.map(r => (
              <tr key={r.hypothesis}>
                <td><span className="mono">{r.hypothesis}</span><div className="muted">{r.lens}</div></td>
                <td>{r.prediction}<div className="muted">{r.result || 'not run'}</div></td>
                <td>
                  <VerdictChip verdict={r.verdict} provisional={r.provisional} />
                  {r.history.length > 1 ? <div className="muted">was {r.history.slice(0, -1).join(', ')}</div> : null}
                </td>
                <td><span className="mono">{r.branch}</span>{r.note ? <div className="tag">Papiers {r.note}</div> : null}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </details>
  )
}

export function ReportView({ view }: { view: View }) {
  const [ref, { width, height }] = useSize<HTMLDivElement>()
  const rep = view.report
  if (!rep) return <p className="empty">No report yet.</p>
  const pts = rep.points
  const m = { l: 56, r: 24, t: 24, b: 40 }
  const lat = pts.map(p => p.latency_ms)
  const x0 = 0, x1 = Math.max(...lat) * 1.08
  const e0 = -4, e1 = -1
  const iw = Math.max(width - m.l - m.r, 40), ih = Math.max(height - m.t - m.b, 40)
  const x = (v: number) => m.l + ((v - x0) / (x1 - x0)) * iw
  const y = (v: number) => m.t + ((e1 - Math.min(e1, Math.max(e0, Math.log10(v)))) / (e1 - e0)) * ih

  return (
    <div className="stack" style={{ minHeight: 0, flex: 1 }} data-testid="report">
      <div className="card">
        <div className="row"><h3>Comparative report</h3><MockTag on={view.mock} /></div>
        <p>{rep.recommendation}</p>
        <p className="small">
          <span className="muted">Best at report time:</span> <span className="mono">{rep.best.run}</span>, {times(rep.best.vs_naive)} vs naive, {times(rep.best.vs_eager)} vs eager
          {view.best ? <>. <span className="muted">Best at session end:</span> <span className="mono">{view.best.run}</span>, {times(view.best.vs_naive)} vs naive, {times(view.best.vs_eager)} vs eager</> : null}
        </p>
      </div>
      <div className="chart" ref={ref}>
        <span className="panel-title">Latency against adversarial error, lower left is better. Pareto points are filled.</span>
        <svg width={width} height={height} role="img" aria-label="pareto chart">
          {[-4, -3, -2, -1].map(e => <text key={e} x={m.l - 8} y={y(10 ** e) + 4} fontSize="11" textAnchor="end" fill="var(--muted)">1e{e}</text>)}
          {[0, 10, 20, 30, 40].filter(v => v <= x1).map(v => (
            <text key={v} x={x(v)} y={m.t + ih + 18} fontSize="11" textAnchor="middle" fill="var(--muted)">{v} ms</text>
          ))}
          <line x1={m.l} x2={m.l + iw} y1={y(1e-3)} y2={y(1e-3)} stroke="var(--ink)" strokeDasharray="4 4" />
          <text x={m.l + iw} y={y(1e-3) - 4} fontSize="11" textAnchor="end">bound 1e-3</text>
          {pts.map(p => {
            const pareto = rep.pareto.includes(p.run)
            return (
              <g key={p.run}>
                <circle cx={x(p.latency_ms)} cy={y(p.adv_error)} r="5" fill={pareto ? 'var(--ink)' : 'var(--surface)'} stroke="var(--ink)" strokeDasharray={p.reference ? '2 2' : undefined} />
                <text x={x(p.latency_ms) + 8} y={y(p.adv_error) - 6} fontSize="11">{p.label}{p.reference ? ' (reference)' : ''}</text>
              </g>
            )
          })}
        </svg>
      </div>
    </div>
  )
}
