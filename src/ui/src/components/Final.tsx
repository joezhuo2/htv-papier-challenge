import { useEffect, useState } from 'react'
import type { Command } from '@contract/types'
import type { View } from '../state/reducer'
import { boundText, clock, lensName, ms, paramText, runOf, windowText } from '../state/derive'
import { MockTag, VerdictChip } from './Glyph'

// one closing screen: best result, what was rejected and why, and what the humans decided
export function ResultCard({ view }: { view: View }) {
  const best = view.best
  const run = best && runOf(view, best.run)
  const rejected = view.hypotheses.filter(h => view.verdicts[h.id]?.verdict === 'rejected')
  const answers = view.feed.filter(f => f.kind === 'answer' || f.kind === 'plan_change')
  const exported = view.feed.flatMap(f => (f.kind === 'export' ? f.items : []))
  return (
    <div className="stack result" data-testid="result-card" style={{ overflow: 'auto', minHeight: 0 }}>
      <article className="card focus">
        <div className="row"><h2>Final result</h2><MockTag on={view.mock} /></div>
        {best && run?.metrics ? (
          <>
            <p className="strong" style={{ fontSize: 20 }}>{best.vs_naive.toFixed(2)}x faster than naive, {best.vs_eager.toFixed(2)}x vs PyTorch eager</p>
            <p className="small">
              <span className="mono">{run.id}</span> on <span className="mono">{run.branch}</span>, {ms(run.metrics.latency_ms)}, {run.metrics.tflops.toFixed(1)} TFLOPS,
              {' '}max error {run.metrics.max_error.toExponential(1)}, adversarial {run.metrics.adv_error.toExponential(1)}
            </p>
            {run.params ? <p className="mono small">{paramText(run.params)}</p> : null}
            {run.metrics.speedups ? <p className="small muted">{windowText(run.metrics)}, same timing window</p> : null}
            {view.verdicts[run.hypothesis] ? <p className="small">Judged against {boundText(view.verdicts[run.hypothesis].bound)}.</p> : null}
          </>
        ) : <p className="empty">No confirmed result in this session.</p>}
      </article>
      <article className="card">
        <h3>What the agent rejected, and why</h3>
        {rejected.length === 0 ? <p className="empty">Nothing was rejected.</p> : rejected.map(h => (
          <div key={h.id} className="small" style={{ paddingTop: 4 }}>
            <div className="inline"><span className="mono">{h.id}</span><VerdictChip verdict="rejected" /><span className="muted">{h.claim}</span></div>
            <p>{view.verdicts[h.id].reason}</p>
          </div>
        ))}
      </article>
      <article className="card">
        <h3>What the lens owners decided</h3>
        {view.directives.length + answers.length === 0 ? <p className="empty">No human input in this session.</p> : null}
        <ul className="plain small">
          {view.directives.map(d => <li key={d.id}>{clock(d.t)} {lensName(d.lens)} directive: {d.text}</li>)}
          {answers.map((f, i) => <li key={i}>{clock(f.t)} {f.kind === 'answer' ? `${lensName(f.by)} answered ${f.help}` : `Plan change: ${f.text}`}</li>)}
        </ul>
        {exported.length ? <p className="small muted">Exported for Papiers import: {exported.map(x => x.file).join(', ')}</p> : null}
      </article>
    </div>
  )
}

interface Check { ok: boolean; detail: string }

export function Preflight({ onStart, fallback }: { onStart: () => void; fallback: string }) {
  const [checks, setChecks] = useState<Record<string, Check>>()
  const [error, setError] = useState<string>()
  useEffect(() => {
    fetch('/preflight').then(r => r.json()).then(b => setChecks(b.checks), (e: Error) => setError(`Could not reach the agent server: ${e.message}`))
  }, [])
  const failed = checks && Object.values(checks).some(c => !c.ok)
  return (
    <article className="card focus" data-testid="preflight" style={{ maxWidth: 560 }}>
      <h2>Pre-flight</h2>
      {error ? <p className="small" role="alert">{error}</p> : !checks ? <p className="small muted">Checking GPU, model keys and Papiers...</p> : (
        <ul className="plain small" style={{ listStyle: 'none', paddingLeft: 0 }}>
          {Object.entries(checks).map(([name, c]) => (
            <li key={name} className="inline" data-testid={`check-${name}`}>
              <span className={`verdict ${c.ok ? 'confirmed' : 'rejected'}`}>{c.ok ? 'ok' : 'failed'}</span>
              <span className="strong">{name}</span><span className="muted">{c.detail}</span>
            </li>
          ))}
        </ul>
      )}
      <div className="inline">
        <button className="primary" onClick={onStart} disabled={!checks}>Start session</button>
        <a href={fallback} className={failed || error ? 'strong' : 'muted'}>Fall back to the recorded run</a>
      </div>
    </article>
  )
}

export function Stalled({ view, live, onSend }: { view: View; live: boolean; onSend: (c: Command) => void }) {
  const s = view.stalled
  if (!s) return null
  return (
    <article className="card focus" role="status" data-testid="stalled">
      <div className="row"><span className="verdict rejected">{s.kind === 'llm_timeout' ? 'timed out' : 'model error'}</span><span className="mono small muted">{clock(s.t)}</span></div>
      <p className="small">{s.text}</p>
      {live ? (
        <div className="inline">
          <button onClick={() => onSend({ type: 'retry' })}>Retry now</button>
          <button onClick={() => onSend({ type: 'stop' })}>Stop session</button>
        </div>
      ) : null}
    </article>
  )
}
