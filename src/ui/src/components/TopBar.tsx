import type { Mode } from '../source'
import { MAX_GAP_MS } from '../source'
import { clock } from '../state/derive'

interface Props {
  mode: Mode
  mock: boolean
  now: number
  paused: boolean
  ended: boolean
  speed: number
  onPause: () => void
  onResume: () => void
  onStep: () => void
  onPrev: () => void
  canPrev: boolean
  onMode: (m: 'mock' | 'live') => void
  onSpeed: (s: number) => void
  onRestart: () => void
}

export function TopBar(p: Props) {
  const replay = p.mode !== 'live'
  return (
    <header className="topbar">
      <h1>Research War Room</h1>
      <span className="muted small">fused matmul + bias + GELU</span>
      {p.mode === 'live' ? <span className="pill ink">live</span> : null}
      {p.mode === 'replay' ? (
        <span className="pill ink" title={`Idle gaps longer than ${MAX_GAP_MS / 1000}s are compressed`}>
          recorded run <span className="muted">idle gaps compressed</span>
        </span>
      ) : null}
      <div className="toggle" role="group" aria-label="data source">
        {(['mock', 'live'] as const).map(m => (
          <button key={m} aria-pressed={p.mode === m} onClick={() => p.mode !== m && p.onMode(m)}>{m}</button>
        ))}
      </div>
      {p.mode === 'replay' && p.mock ? <span className="tag mock">mock data</span> : null}
      <span className="spacer" />
      <span className="mono" aria-label="elapsed">{clock(p.now)}</span>
      {replay ? (
        <div className="controls" role="group" aria-label="presenter controls">
          {p.paused ? (
            <button onClick={p.onResume} disabled={p.ended}>resume</button>
          ) : (
            <button onClick={p.onPause}>pause</button>
          )}
          <button onClick={p.onPrev} disabled={!p.canPrev} title="Left arrow">prev</button>
          <button onClick={p.onStep} disabled={p.ended} title="Right arrow">step</button>
          {[1, 2, 4].map(s => (
            <button key={s} aria-pressed={p.speed === s} onClick={() => p.onSpeed(s)}>{s}x</button>
          ))}
          <button onClick={p.onRestart}>restart</button>
        </div>
      ) : null}
    </header>
  )
}
