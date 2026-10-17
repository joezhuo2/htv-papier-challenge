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
      {p.mode === 'mock' || p.mock ? <span className="tag mock">mock</span> : null}
      <span className="spacer" />
      <span className="mono" aria-label="elapsed">{clock(p.now)}</span>
      {replay ? (
        <div className="controls" role="group" aria-label="presenter controls">
          {p.paused ? (
            <button onClick={p.onResume} disabled={p.ended}>resume</button>
          ) : (
            <button onClick={p.onPause}>pause</button>
          )}
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
