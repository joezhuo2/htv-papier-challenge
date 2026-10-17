import type { Chapter } from '../chapters'
import type { Mode } from '../source'
import { clock } from '../state/derive'

interface Props {
  mode: Mode
  file?: string
  mock: boolean
  recording?: boolean
  judging: boolean
  now: number
  paused: boolean
  ended: boolean
  speed: number
  compression?: { from: number; to: number }
  chapters: Chapter[]
  autoPause: boolean
  canPrev: boolean
  onPause: () => void
  onResume: () => void
  onStep: () => void
  onPrev: () => void
  onSpeed: (s: number) => void
  onRestart: () => void
  onMode: (m: Mode) => void
  onChapter: (c: Chapter) => void
  onAutoPause: (on: boolean) => void
  onStop: () => void
  onSave: () => void
  onNewSession: () => void
  live?: { started: boolean; finished: boolean; stopping: boolean }
}

export function TopBar(p: Props) {
  const replay = p.mode !== 'live'
  const modes: Mode[] = p.judging ? ['replay', 'live'] : ['mock', 'live']
  const pill = p.mode === 'live' ? (p.recording ? 'live, recording' : 'live') : p.mode === 'replay' ? 'recorded run' : 'mock'
  return (
    <header className="topbar">
      <h1>Research War Room</h1>
      <span className="muted small">fused matmul + bias + GELU</span>
      <span className={`pill ink${p.mode === 'mock' ? ' mock' : ''}`} data-testid="mode-pill">
        {pill}
        {p.mode === 'replay' && p.compression && p.compression.from !== p.compression.to ? (
          <span className="muted" title="Idle gaps longer than 4 s between events are shortened, the order of events is unchanged">
            idle gaps compressed, {clock(p.compression.from)} to {clock(p.compression.to)}
          </span>
        ) : null}
      </span>
      {p.mode === 'replay' && p.file ? <span className="mono small muted" data-testid="replay-file">{p.file.split('/').pop()}</span> : null}
      {p.mode === 'replay' && p.mock ? <span className="tag mock">mock data</span> : null}
      <div className="toggle" role="group" aria-label="data source">
        {modes.map(m => (
          <button key={m} aria-pressed={p.mode === m} onClick={() => p.mode !== m && p.onMode(m)}>{m}</button>
        ))}
      </div>
      <span className="spacer" />
      {p.chapters.length ? (
        <select
          aria-label="jump to chapter" className="chapters" value=""
          onChange={e => { const c = p.chapters.find(x => x.key === e.target.value); if (c) p.onChapter(c) }}
        >
          <option value="">chapters</option>
          {p.chapters.map(c => <option key={c.key} value={c.key}>{clock(c.t)} {c.label}</option>)}
        </select>
      ) : null}
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
          <button aria-pressed={p.autoPause} onClick={() => p.onAutoPause(!p.autoPause)} title="Pause at each chapter">auto-pause</button>
          <button onClick={p.onRestart}>restart</button>
        </div>
      ) : (
        <div className="controls" role="group" aria-label="live controls">
          <button onClick={p.onSave} title={p.live?.finished ? 'Download the whole session as a JSONL replay' : 'The session has not finished: this saves only the events so far, without session_finished'}>
            {p.live?.finished ? 'save as replay' : 'save partial replay'}
          </button>
          <button onClick={p.onStop} disabled={!p.live?.started || p.live.finished || p.live.stopping} title="Finish the session with what is logged so far">
            {p.live?.stopping && !p.live.finished ? 'stopping...' : 'stop'}
          </button>
          <button onClick={p.onNewSession} title="Stop this session if it is running and start a new one">new session</button>
        </div>
      )}
    </header>
  )
}
