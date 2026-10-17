import { useEffect, useState } from 'react'
import type { Command, FinalReport } from '@contract/types'
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
        <div className="row"><h2>Best confirmed result</h2><MockTag on={view.mock} /></div>
        {best && run?.metrics ? (
          <>
            <p className="strong" style={{ fontSize: 20 }}>{best.vs_naive.toFixed(2)}x faster than naive, {best.vs_eager.toFixed(2)}x vs PyTorch eager</p>
            <p className="small">
              <span className="tag" data-testid="best-label">best confirmed</span> <span className="mono">{run.id}</span>
              {view.final?.best ? <> (<span className="mono">{view.final.best.hypothesis}</span>)</> : null} on <span className="mono">{run.branch}</span>, {ms(run.metrics.latency_ms)}, {run.metrics.tflops.toFixed(1)} TFLOPS,
              {' '}max error {run.metrics.max_error.toExponential(1)}, adversarial {run.metrics.adv_error.toExponential(1)}
            </p>
            {run.params ? <p className="mono small">{paramText(run.params)}</p> : null}
            {run.metrics.speedups ? <p className="small muted">{windowText(run.metrics)}, same timing window</p> : null}
            {view.verdicts[run.hypothesis] ? <p className="small">Judged against {boundText(view.verdicts[run.hypothesis].bound)}.</p> : null}
          </>
        ) : <p className="empty">No confirmed result in this session.</p>}
      </article>
      {view.final ? <FinalSections report={view.final} /> : null}
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

function FinalSections({ report }: { report: FinalReport }) {
  const sections: [string, string, string[]][] = [
    ['Ruled out', 'Nothing was ruled out.', report.ruled_out],
    ['What to try next', 'No follow-up suggested.', report.next],
    ['Needs a human decision', 'Nothing is waiting on a lens owner.', report.human],
  ]
  return (
    <>
      {sections.map(([title, empty, items]) => (
        <article key={title} className="card" data-testid={`final-${title.split(' ')[0].toLowerCase()}`}>
          <h3>{title}</h3>
          {items.length ? <ul className="plain small">{items.map((x, i) => <li key={i}>{x}</li>)}</ul> : <p className="empty">{empty}</p>}
        </article>
      ))}
      <article className="card" data-testid="final-hunches">
        <h3>Your hunches</h3>
        {report.hunches.length ? (
          <ul className="plain small">
            {report.hunches.map(x => <li key={x.directive}><span className="mono">{x.directive}</span> {lensName(x.lens)}: "{x.text}": {x.outcome}</li>)}
          </ul>
        ) : <p className="empty">No hunch was typed in this session.</p>}
      </article>
    </>
  )
}

interface Check { ok: boolean; detail: string; warn?: boolean }

export function Preflight({ onStart, fallback, log }: { onStart: (record: boolean) => void; fallback: string; log: View['log'] }) {
  const [checks, setChecks] = useState<Record<string, Check>>()
  const [caps, setCaps] = useState<{ cap: number | null; record_cap: number }>()
  const [error, setError] = useState<string>()
  useEffect(() => {
    fetch('/preflight').then(r => r.json()).then(b => { setChecks(b.checks); setCaps(b) }, (e: Error) => setError(`Could not reach the agent server: ${e.message}`))
  }, [])
  const plural = (n: number) => `${n} hypothes${n === 1 ? 'is' : 'es'}`
  const failed = checks && Object.values(checks).some(c => !c.ok)
  const latest = log.at(-1)
  const blocked = latest?.text.startsWith('Cannot start')
  // one start per session: locked from the click (or the first agent event) until the session starts, unless it cannot start
  const [clicked, setClicked] = useState(false)
  const locked = (clicked || log.length > 0) && !blocked
  const start = (record: boolean) => { setClicked(true); onStart(record) }
  return (
    <article className="card focus" data-testid="preflight" style={{ maxWidth: 560 }}>
      <h2>Pre-flight</h2>
      {error ? <p className="small" role="alert">{error}</p> : !checks ? <p className="small muted">Checking GPU, model keys and Papiers...</p> : (
        <ul className="plain small" style={{ listStyle: 'none', paddingLeft: 0 }}>
          {Object.entries(checks).map(([name, c]) => (
            <li key={name} className="inline" data-testid={`check-${name}`}>
              <span className={`verdict ${!c.ok ? 'rejected' : c.warn ? 'inconclusive' : 'confirmed'}`}>{!c.ok ? 'failed' : c.warn ? 'warning' : 'ok'}</span>
              <span className="strong">{name}</span><span className="muted">{c.detail}</span>
            </li>
          ))}
        </ul>
      )}
      {latest ? <p className={`small${blocked ? ' strong' : ' muted'}`} role={blocked ? 'alert' : 'status'} data-testid="starting">{latest.text}</p> : null}
      <div className="inline">
        <button className="primary" onClick={() => start(false)} disabled={!checks || locked} data-testid="start">Start session</button>
        {caps?.cap ? <span className="small muted">table demo, capped at {plural(caps.cap)}</span> : null}
        <a href={fallback} className={failed || error ? 'strong' : 'muted'}>Fall back to the recorded run</a>
      </div>
      <hr className="divider" />
      <h3>Record a replay</h3>
      <p className="small muted">
        Not the table demo. Runs H0, then waits for your lens directive (continuing on main as a flagged default if none arrives in time),
        and keeps going until {plural(caps?.record_cap ?? 8)} or Stop. Directives and help answers are accepted at any point, and the whole
        event log is saved as a replay under runs/sessions.
      </p>
      <div className="inline">
        <button onClick={() => start(true)} disabled={!checks || locked} data-testid="record">Start recording</button>
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
