import type { KeyboardEvent } from 'react'
import type { BranchView, View } from '../state/reducer'
import { lensName } from '../state/derive'
import { useSize } from '../useSize'
import { Lock } from './Glyph'

interface Props {
  view: View
  onSelectBranch: (b: string) => void
  onSelectRun: (r: string) => void
  showPending?: boolean
}

const statusWord: Record<BranchView['status'], string> = {
  running: 'running', done: 'done', rejected: 'rejected', waiting_on_human: 'blocked', merged: 'merged',
}

const activate = (fn: () => void) => ({
  role: 'button',
  tabIndex: 0,
  onClick: fn,
  onKeyDown: (e: KeyboardEvent) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault()
      e.stopPropagation()
      fn()
    }
  },
})

const clip = (s: string, px: number) => {
  const n = Math.floor(px / 6.6)
  return s.length > n ? `${s.slice(0, n - 1)}...` : s
}

export function BranchGraph({ view, onSelectBranch, onSelectRun, showPending = true }: Props) {
  const [ref, { width, height }] = useSize<HTMLDivElement>()
  const pad = 16
  // five columns matches the usual session, so nodes do not move as branches appear
  // nodes and lanes scale to the panel: columns follow the branch count, rows only for lenses that have branches
  const cols = Math.max(view.branches.length, 3)
  const colW = (width - pad * 2) / cols
  const gap = colW < 130 ? 14 : 32
  const w = Math.min(280, colW - gap)
  const top = 34
  const rows = (['speed', 'both', 'precision'] as const).filter(l => view.branches.some(b => b.lens === l))
  const band = (height - top - 8) / Math.max(rows.length, 1)
  const NODE_H = Math.round(Math.min(64, Math.max(44, band * 0.36)))
  const fs = NODE_H >= 58 && w > 190 ? 14 : NODE_H >= 50 && w > 160 ? 13 : 12
  const laneY = Object.fromEntries(rows.map((l, i) => [l, top + band * i + Math.max(NODE_H / 2 + 6, band * 0.4)])) as Record<BranchView['lens'], number>
  const pos = new Map(view.branches.map(b => [b.id, { x: pad + b.index * colW + (colW - gap - w) / 2, y: laneY[b.lens] - NODE_H / 2 }]))

  return (
    <div className="graph" ref={ref} data-testid="branch-graph">
      <span className="panel-title">Branches</span>
      {view.branches.length === 0 ? <p className="empty" style={{ padding: '32px 12px' }}>No branches yet.</p> : null}
      <svg width={width} height={height} role="img" aria-label="branch graph">
        {view.branches.flatMap(b => b.parents.map(pid => {
          const a = pos.get(pid)
          const c = pos.get(b.id)!
          if (!a) return null
          const x1 = a.x + w, y1 = a.y + NODE_H / 2, x2 = c.x, y2 = c.y + NODE_H / 2
          const mx = (x1 + x2) / 2
          return <path key={`${pid}-${b.id}`} d={`M${x1} ${y1} C${mx} ${y1} ${mx} ${y2} ${x2} ${y2}`} fill="none" stroke="var(--muted)" strokeWidth="1" />
        }))}
        {view.branches.map(b => {
          const { x, y } = pos.get(b.id)!
          const sel = view.selected.branch === b.id
          const runs = view.runs.filter(r => r.branch === b.id && (showPending || r.finished !== undefined))
          const blocker = view.helps.find(h => h.help.blocks === b.id && !h.answer)
          const s = b.status
          const dash = s === 'rejected' ? '5 3' : undefined
          const roomy = (b.id.length + statusWord[s].length + 1) * fs * 0.62 + 40 <= w
          return (
            <g key={b.id} data-testid={`node-${b.id}`} data-status={s} aria-label={`branch ${b.id}, ${statusWord[s]}`}>
              <g {...activate(() => onSelectBranch(b.id))}>
                {s === 'running' ? <rect className="pulse" x={x - 3} y={y - 3} width={w + 6} height={NODE_H + 6} rx="8" fill="none" stroke="var(--ink)" strokeWidth="1" /> : null}
                {s === 'merged' ? <rect x={x - 3} y={y - 3} width={w + 6} height={NODE_H + 6} rx="8" fill="none" stroke="var(--ink)" strokeWidth="1" /> : null}
                <rect
                  className="hit" x={x} y={y} width={w} height={NODE_H} rx="6"
                  fill={s === 'waiting_on_human' ? 'none' : 'var(--surface)'}
                  stroke={sel ? 'var(--accent)' : s === 'rejected' ? 'var(--rejected)' : 'var(--ink)'} strokeDasharray={dash} strokeWidth={sel ? 1.5 : 1}
                />
                <title>{b.label}</title>
                <text x={x + 8} y={y + NODE_H / 2 - 5} fontSize={fs} className="mono">
                  <tspan fontWeight="600" textDecoration={s === 'rejected' ? 'line-through' : undefined}>{b.id}</tspan>
                  {roomy ? <tspan dx="6" fill="var(--muted)">{statusWord[s]}</tspan> : null}
                </text>
                <text x={x + 8} y={y + NODE_H / 2 + 12} fontSize={fs} textDecoration={s === 'rejected' && roomy ? 'line-through' : undefined} className={roomy ? undefined : 'mono'} fill={roomy ? undefined : 'var(--muted)'}>
                  {roomy ? clip(b.label, w - 16) : statusWord[s]}
                </text>
                {s === 'rejected' || s === 'waiting_on_human' ? (
                  <circle cx={x + w} cy={y} r="9" fill="var(--surface)" stroke={s === 'rejected' ? 'var(--rejected)' : 'var(--ink)'} />
                ) : null}
                {s === 'rejected' ? (
                  <path d={`M${x + w - 4} ${y - 4} l8 8 m0 -8 l-8 8`} stroke="var(--rejected)" strokeWidth="1.6" strokeLinecap="round" />
                ) : null}
                {s === 'waiting_on_human' ? <g color="var(--ink)"><Lock x={x + w - 6} y={y - 6} /></g> : null}
              </g>
              {s === 'waiting_on_human' ? (
                <text x={x + w > width - 160 ? x + w : x} y={y - 14} fontSize="12" textAnchor={x + w > width - 160 ? 'end' : 'start'} data-testid="blocked-label">
                  waiting for {blocker ? lensName(blocker.help.target) : 'lens'} owner
                </text>
              ) : null}
              <line x1={x} x2={x + w} y1={y + NODE_H + 12} y2={y + NODE_H + 12} stroke="var(--border)" />
              {runs.map((r, i) => {
                const cx = x + 10 + i * 30
                const cy = y + NODE_H + 12
                const vd = view.verdicts[r.hypothesis]
                const final = vd && !vd.provisional ? vd.verdict : undefined
                const rsel = view.selected.run === r.id
                return (
                  <g key={r.id} {...activate(() => onSelectRun(r.id))} aria-label={`run ${r.id}, ${final ?? (r.metrics ? 'measured' : r.failed ? 'failed' : 'running')}`}
                    data-testid={r.finished === undefined ? `pending-${r.id}` : undefined} className={r.finished === undefined ? 'working' : undefined}>
                    {rsel ? <circle cx={cx} cy={cy} r="8.5" fill="none" stroke="var(--ink)" strokeWidth="1" /> : null}
                    <circle className="hit" cx={cx} cy={cy} r="5" fill={final === 'confirmed' ? 'var(--ink)' : 'var(--surface)'} stroke="var(--ink)" strokeWidth="1" strokeDasharray={r.metrics || r.failed ? undefined : '2 2'} />
                    {final === 'rejected' ? <path d={`M${cx - 3} ${cy - 3} l6 6 m0 -6 l-6 6`} stroke="var(--rejected)" strokeWidth="1.4" /> : null}
                    {final === 'inconclusive' ? <path d={`M${cx - 3} ${cy} q1.5 -2 3 0 t3 0`} stroke="var(--ink)" fill="none" strokeWidth="1.2" /> : null}
                    <text x={cx} y={cy + 18} fontSize="11" textAnchor="middle" className="mono">{r.id}</text>
                  </g>
                )
              })}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
