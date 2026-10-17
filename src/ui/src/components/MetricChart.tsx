import type { KeyboardEvent } from 'react'
import type { View } from '../state/reducer'
import { predictedLatency, runOf, sci, times } from '../state/derive'
import { useSize } from '../useSize'

interface Props { view: View; onSelectRun: (r: string) => void }

const press = (fn: () => void) => ({
  role: 'button', tabIndex: 0, onClick: fn,
  onKeyDown: (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      e.stopPropagation()
      fn()
    }
  },
})

const niceStep = (max: number) => [0.5, 1, 2, 5, 10, 20].find(s => max / s <= 6) ?? 50

export function MetricChart({ view, onSelectRun }: Props) {
  const [ref, { width, height }] = useSize<HTMLDivElement>()
  const naive = view.baselines?.naive.latency_ms
  const eager = view.baselines?.eager.latency_ms
  const sp = (lat: number) => (naive ? naive / lat : 0)

  const ghosts = view.hypotheses.flatMap(h => {
    const lat = predictedLatency(view, h)
    if (lat === undefined) return []
    const run = [...view.runs].reverse().find(r => r.hypothesis === h.id)
    return [{ h: h.id, run, value: sp(lat) }]
  })
  const pending = ghosts.filter(g => !g.run).length
  const slots = Math.max(view.runs.length + pending, 6)
  const measured = view.runs.filter(r => r.metrics).map(r => sp(r.metrics!.latency_ms))
  const maxY = Math.max(eager ? sp(eager) : 1, ...measured, ...ghosts.map(g => g.value), 2) * 1.12
  const step = niceStep(maxY)

  const m = { l: 44, r: 72, t: width < 560 ? 58 : 40, b: 28 }
  const iw = Math.max(width - m.l - m.r, 40)
  const ih = Math.max(height - m.t - m.b, 40)
  const x = (i: number) => m.l + (slots === 1 ? iw / 2 : (i + 0.5) * (iw / slots))
  const y = (v: number) => m.t + ih - (v / maxY) * ih
  const selBranch = view.selected.branch
  const branches = [...new Set(view.runs.map(r => r.branch))]
  let next = view.runs.length
  const crowded = iw / slots < 56
  const lastRun = view.runs.filter(r => r.metrics).at(-1)?.id

  return (
    <div className="chart" ref={ref} data-testid="metric-chart">
      <div className="panel-head">
        <span>Speedup vs naive Triton, higher is better</span>
        <span className="legend" aria-label="legend">
          <svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" fill="var(--surface)" stroke="var(--muted)" strokeWidth="1.5" /></svg>predicted
          <svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" fill="var(--ink)" /></svg>measured
          <svg width="12" height="12" viewBox="0 0 12 12"><circle cx="6" cy="6" r="4.5" fill="var(--surface)" stroke="var(--ink)" /><path d="M3.5 3.5 l5 5 m0 -5 l-5 5" stroke="var(--rejected)" strokeWidth="1.4" /></svg>rejected
        </span>
      </div>
      {!naive ? <p className="empty" style={{ padding: '32px 12px' }}>No runs yet.</p> : (
        <svg width={width} height={height} role="img" aria-label="speedup by run">
          {Array.from({ length: Math.floor(maxY / step) + 1 }, (_, i) => i * step).map(v => (
            <g key={v}>
              <line x1={m.l} x2={m.l + iw} y1={y(v)} y2={y(v)} stroke="var(--border)" />
              <text x={m.l - 6} y={y(v) + 4} fontSize="11" textAnchor="end" className="muted" fill="var(--muted)">{v}x</text>
            </g>
          ))}
          {[{ v: 1, label: 'naive' }, ...(eager ? [{ v: sp(eager), label: 'eager' }] : [])].map(r => (
            <g key={r.label}>
              <line x1={m.l} x2={m.l + iw} y1={y(r.v)} y2={y(r.v)} stroke="var(--ink)" strokeDasharray="4 4" strokeWidth="1" />
              <text x={m.l + iw + 6} y={y(r.v) + 4} fontSize="11">{r.label} {times(r.v)}</text>
            </g>
          ))}
          {branches.map(b => {
            const pts = view.runs.filter(r => r.branch === b && r.metrics)
            if (pts.length < 2) return null
            return (
              <polyline
                key={b} fill="none" points={pts.map(r => `${x(r.index)},${y(sp(r.metrics!.latency_ms))}`).join(' ')}
                stroke={selBranch === b ? 'var(--ink)' : '#B9B2A4'} strokeWidth={selBranch === b ? 2 : 1}
              />
            )
          })}
          {ghosts.map(g => {
            const i = g.run ? g.run.index : next++
            const done = g.run?.metrics
            const gy = y(g.value)
            return (
              <g key={g.h} data-testid={`ghost-${g.h}`} data-resolved={done ? 'true' : 'false'}>
                {done ? <line x1={x(i)} x2={x(i)} y1={gy} y2={y(sp(done.latency_ms))} stroke="var(--muted)" strokeWidth="1" /> : null}
                <circle cx={x(i)} cy={gy} r="6" fill="var(--surface)" stroke="var(--muted)" strokeWidth="1.5" opacity={done ? 0.8 : 1} />
                {!done && !crowded ? (
                  <text x={x(i) + 10} y={gy + 4} fontSize="11">{g.h} predicted at least {times(g.value)}</text>
                ) : null}
              </g>
            )
          })}
          {view.runs.filter(r => r.metrics).map(r => {
            const v = sp(r.metrics!.latency_ms)
            const vd = view.verdicts[r.hypothesis]
            const final = vd && !vd.provisional ? vd.verdict : undefined
            const sel = view.selected.run === r.id
            const cx = x(r.index), cy = y(v)
            return (
              <g key={r.id} {...press(() => onSelectRun(r.id))} data-testid={`point-${r.id}`} aria-label={`${r.id} ${times(v)} ${final ?? 'pending verdict'}`}>
                <circle className="hit" cx={cx} cy={cy} r={sel ? 6 : 5} fill={final === 'confirmed' ? 'var(--ink)' : 'var(--surface)'} stroke="var(--ink)" strokeWidth={sel ? 2 : 1} strokeDasharray={vd?.provisional ? '1.5 1.5' : undefined} />
                {final === 'rejected' ? <path d={`M${cx - 3} ${cy - 3} l6 6 m0 -6 l-6 6`} stroke="var(--rejected)" strokeWidth="1.4" /> : null}
                {!crowded || sel || r.id === lastRun ? <text x={cx + 9} y={cy - 6} fontSize="11">{times(v)}</text> : null}
              </g>
            )
          })}
          {view.runs.map(r => (
            <text key={r.id} x={x(r.index)} y={m.t + ih + 18} fontSize="11" textAnchor="middle" className="mono">{r.id}</text>
          ))}
        </svg>
      )}
    </div>
  )
}

export function AdvPanel({ view }: { view: View }) {
  const [ref, { width, height }] = useSize<HTMLDivElement>({ width: 220, height: 200 })
  const done = view.runs.filter(r => r.metrics)
  const sel = runOf(view, view.selected.run) ?? done.at(-1)
  const lo = -4.5, hi = -1
  const m = { l: 40, r: 16, t: 44, b: 26 }
  const ih = Math.max(height - m.t - m.b, 40)
  const iw = Math.max(width - m.l - m.r, 40)
  const y = (v: number) => m.t + ((hi - Math.min(hi, Math.max(lo, Math.log10(v)))) / (hi - lo)) * ih
  const x = (i: number) => m.l + (done.length <= 1 ? iw / 2 : (i / (done.length - 1)) * iw)
  const bound = 1e-3

  return (
    <div className="chart adv" ref={ref} data-testid="adv-panel">
      <span className="panel-title">Adversarial error{sel ? `, ${sel.id}` : ''}</span>
      {!sel?.metrics ? <p className="empty" style={{ padding: '32px 12px' }}>No runs yet.</p> : (
        <svg width={width} height={height} role="img" aria-label={`adversarial error ${sel.id} ${sci(sel.metrics.adv_error)}`}>
          {[-4, -3, -2, -1].map(e => (
            <text key={e} x={m.l - 6} y={y(10 ** e) + 4} fontSize="11" textAnchor="end" fill="var(--muted)">1e{e}</text>
          ))}
          <line x1={m.l} x2={m.l + iw} y1={y(bound)} y2={y(bound)} stroke="var(--ink)" strokeDasharray="4 4" />
          <text x={m.l + 4} y={y(bound) - 4} fontSize="11">bound 1e-3</text>
          {done.map((r, i) => {
            const on = r.id === sel.id
            const v = r.metrics!.adv_error
            return (
              <g key={r.id}>
                <circle cx={x(i)} cy={y(v)} r={on ? 5 : 3} fill={on ? 'var(--ink)' : '#B9B2A4'} />
                {on ? (
                  <text x={x(i) + (x(i) > m.l + iw / 2 ? -9 : 9)} y={y(v) + 4} fontSize="11" textAnchor={x(i) > m.l + iw / 2 ? 'end' : 'start'}>
                    {sci(v)}{v > bound ? ' over' : ''}
                  </text>
                ) : null}
                {on || done.length < 2 || iw / (done.length - 1) >= 26 ? (
                  <text x={x(i)} y={m.t + ih + 16} fontSize="10" textAnchor="middle" className="mono" fill={on ? 'var(--ink)' : 'var(--muted)'}>{r.id}</text>
                ) : null}
              </g>
            )
          })}
        </svg>
      )}
    </div>
  )
}
