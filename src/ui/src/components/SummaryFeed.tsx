import { useEffect, useRef } from 'react'
import type { FeedItem, View } from '../state/reducer'
import { clock, hypothesisOf, lensName, ms, papiersNote, parentBest, predictionText, runOf, sci, times } from '../state/derive'
import { MockTag, VerdictChip } from './Glyph'
import type { ReactNode } from 'react'

interface Props {
  view: View
  onSelectRun: (run: string) => void
  onPostMortem: (h: string) => void
  onReport: () => void
  hybrid: (item: Extract<FeedItem, { kind: 'hybrid' }>) => ReactNode
}

type SummaryItem = Extract<FeedItem, { kind: 'summary' }>

function SummaryCard({ view, item, onSelectRun, onPostMortem }: { view: View; item: SummaryItem } & Pick<Props, 'onSelectRun' | 'onPostMortem'>) {
  const { summary } = item
  const h = hypothesisOf(view, item.hypothesis)
  const run = runOf(view, item.run)
  const m = run?.metrics
  const vd = item.verdict
  const current = item.hypothesis ? view.verdicts[item.hypothesis] : undefined
  const superseded = vd && current && (current.verdict !== vd.verdict || current.provisional !== vd.provisional)
  const base = h && (h.prediction.vs === 'naive' ? undefined : parentBest(view, h.branch, h.t))
  const eager = view.baselines?.eager.latency_ms
  const naive = view.baselines?.naive.latency_ms
  const note = papiersNote(h)
  const rejected = vd?.verdict === 'rejected' && !vd.provisional

  return (
    <article className="card" data-testid="summary-card" aria-label={`${item.hypothesis ?? ''} summary`}>
      <div className="row">
        <h3>{item.hypothesis ? <span className="mono">{item.hypothesis}</span> : null} on <span className="mono">{summary.branch}</span></h3>
        <VerdictChip verdict={vd?.verdict} provisional={vd?.provisional} />
        <span className="spacer" style={{ flex: 1 }} />
        <span className="mono small muted">{clock(item.t)}</span>
      </div>
      <div className="row">
        {superseded ? <span className="tag">superseded, now {current.provisional ? 'provisional' : current.verdict}</span> : null}
        {run?.assumption ? <span className="tag">assumption pending</span> : null}
        {note ? <span className="tag" title={h?.note}>Papiers note {note}</span> : null}
        <MockTag on={item.mock} />
      </div>
      <p>{summary.tried}</p>
      <dl>
        {h ? <><dt>Predicted</dt><dd>{predictionText(view, h)}</dd></> : null}
        {m && run ? (
          <>
            <dt>Measured</dt>
            <dd>
              <button className="link mono" onClick={() => onSelectRun(run.id)}>{run.id}</button>{' '}
              {ms(m.latency_ms)}, {m.tflops.toFixed(1)} TFLOPS, max error {sci(m.max_error)}, adversarial {sci(m.adv_error)}
            </dd>
            <dt>Delta</dt>
            <dd>
              {base ? <>{times(base / m.latency_ms)} vs previous best ({ms(base)})</> : naive ? <>{times(naive / m.latency_ms)} vs naive</> : null}
              {eager ? <>, {times(eager / m.latency_ms)} vs eager</> : null}
            </dd>
          </>
        ) : null}
        <dt>Why</dt><dd>{summary.verdict}</dd>
        <dt>Next</dt><dd>{summary.next_step}</dd>
      </dl>
      <details className="small">
        <summary className="muted">Confidence {summary.confidence}, uncertainties</summary>
        <div className="stack" style={{ paddingTop: 6 }}>
          <p>{summary.confidence_reason}</p>
          <ul className="plain">{summary.uncertainties.map((u, i) => <li key={i}>{u}</li>)}</ul>
          <p><span className="muted">Would change my mind:</span> {summary.would_change_mind}</p>
        </div>
      </details>
      {rejected && item.hypothesis ? (
        <button className="link small" style={{ alignSelf: 'flex-start' }} onClick={() => onPostMortem(item.hypothesis!)}>
          {view.postMortems.some(p => p.hypothesis === item.hypothesis) ? 'Open post-mortem' : 'Post-mortem pending'}
        </button>
      ) : null}
    </article>
  )
}

export function SummaryFeed(p: Props) {
  const { view } = p
  const end = useRef<HTMLDivElement>(null)
  const touched = useRef(0)
  useEffect(() => {
    const box = end.current?.closest('.col')
    const mark = () => { touched.current = Date.now() }
    box?.addEventListener('wheel', mark)
    box?.addEventListener('touchmove', mark)
    return () => {
      box?.removeEventListener('wheel', mark)
      box?.removeEventListener('touchmove', mark)
    }
  }, [])
  // follow the newest card unless the presenter scrolled the feed in the last few seconds
  useEffect(() => {
    const box = end.current?.closest('.col') as HTMLElement | null
    if (box && end.current && Date.now() - touched.current > 8000) box.scrollTop = end.current.offsetTop - box.clientHeight + 12
  }, [view.feed.length])

  const optionLabel = (help: string, option: string) =>
    view.helps.find(h => h.help.id === help)?.help.options.find(o => o.id === option)?.label ?? option

  return (
    <div className="section" aria-label="summary feed">
      <div className="section-head"><h2>Summaries</h2><span className="muted small">{view.feed.filter(f => f.kind === 'summary').length}</span></div>
      {view.feed.length === 0 ? <p className="empty">No runs yet.</p> : null}
      {view.feed.map((f, i) => {
        switch (f.kind) {
          case 'summary':
            return <SummaryCard key={i} view={view} item={f} onSelectRun={p.onSelectRun} onPostMortem={p.onPostMortem} />
          case 'plan_change':
            return (
              <article key={i} className="card" data-testid="plan-change">
                <div className="row"><h3>Plan change</h3><span className="tag">after <span className="mono">{f.help}</span></span><MockTag on={f.mock} /><span style={{ flex: 1 }} /><span className="mono small muted">{clock(f.t)}</span></div>
                <p>{f.text}</p>
              </article>
            )
          case 'answer':
            return (
              <p key={i} className="line">
                <span className="mono">{clock(f.t)}</span> <span className="strong">{lensName(f.by)} lens answered {f.help}:</span> {optionLabel(f.help, f.option)}{f.note ? `. "${f.note}"` : ''}
              </p>
            )
          case 'export':
            return (
              <p key={i} className="line">
                <span className="mono">{clock(f.t)}</span> <span className="strong">Exported for Papiers import:</span> {f.items.map(x => x.title).join('; ')} <MockTag on={f.mock} />
              </p>
            )
          case 'note':
            return (
              <p key={i} className="line" title={f.note.excerpt}>
                <span className="mono">{clock(f.t)}</span> <span className="strong">Read Papiers note {f.note.id}:</span> {f.note.title}
              </p>
            )
          case 'branch':
            return (
              <p key={i} className="line">
                <span className="mono">{clock(f.t)}</span> <span className="strong">Branch <span className="mono">{f.branch}</span> created:</span> {f.reason}
              </p>
            )
          case 'report':
            return (
              <p key={i} className="line">
                <span className="mono">{clock(f.t)}</span> <span className="strong">Comparative report ready.</span> <button className="link" onClick={p.onReport}>Open report</button>
              </p>
            )
          case 'hybrid':
            return <div key={i}>{p.hybrid(f)}</div>
        }
      })}
      <div ref={end} />
    </div>
  )
}
