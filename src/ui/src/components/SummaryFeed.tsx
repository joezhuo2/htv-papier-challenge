import { useEffect, useRef, useState } from 'react'
import type { FeedItem, View } from '../state/reducer'
import { boundText, clock, hypothesisOf, lensName, ms, parentBest, predictionText, runOf, sci, times, windowText } from '../state/derive'
import { Def, MockTag, VerdictChip } from './Glyph'
import { InformedBy, NoteChip, NoteText } from './PapiersPanel'
import type { Command } from '@contract/types'
import { HybridCard } from './Tier2'

interface Props {
  view: View
  onSelectRun: (run: string) => void
  onPostMortem: (h: string) => void
  onReport: () => void
  onSend: (c: Command) => void
  onHybrid: (approved: boolean) => void
  // working states come only from real events: running cards outside recorded replay, choosing only in live mode
  working?: { runs: boolean; choosing: boolean }
}

type SummaryItem = Extract<FeedItem, { kind: 'summary' }>

function SummaryCard({ view, item, onSelectRun, onPostMortem, collapsed, onToggle }: { view: View; item: SummaryItem; collapsed: boolean; onToggle?: () => void } & Pick<Props, 'onSelectRun' | 'onPostMortem'>) {
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
  const rejected = vd?.verdict === 'rejected' && !vd.provisional
  // verdicts up to this card, so a later flip shows on the later card and the earlier card keeps its snapshot
  const trail = (item.hypothesis ? view.trail[item.hypothesis] ?? [] : []).filter(t => t.t <= item.t)
  // written while another branch waits on a help request
  const waiting = view.helps.find(x => x.t < item.t && (!x.answer || x.answer.t > item.t) && !x.cancelled && x.help.blocks !== summary.branch)
  const head = (
    <div className="row">
      <h3>{item.hypothesis ? <span className="mono">{item.hypothesis}</span> : null} on <span className="mono">{summary.branch}</span></h3>
      <VerdictChip verdict={vd?.verdict} provisional={vd?.provisional} />
      {waiting ? <span className="tag" data-testid="meanwhile">meanwhile, <span className="mono">{waiting.help.blocks}</span> waits on {lensName(waiting.help.target)}</span> : null}
      <span style={{ flex: 1 }} />
      <span className="mono small muted">{clock(item.t)}</span>
    </div>
  )

  if (collapsed) {
    return (
      <article
        className="card collapsed" data-testid="summary-card" data-collapsed="true" aria-label={`${item.hypothesis ?? ''} summary, collapsed`}
        role="button" tabIndex={0} onClick={onToggle}
        onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onToggle?.() } }}
      >
        {head}
        <p className="small oneline">{summary.verdict}</p>
      </article>
    )
  }

  return (
    <article className="card" data-testid="summary-card" aria-label={`${item.hypothesis ?? ''} summary`}>
      {head}
      <div className="row">
        {superseded ? <span className="tag">superseded, now {current.provisional ? 'provisional' : current.verdict}</span> : null}
        {run?.assumption ? <span className="tag">assumption pending</span> : null}

        <MockTag on={item.mock} />
      </div>
      <p><NoteText view={view} text={summary.tried} /></p>
      {h ? <InformedBy view={view} h={h} /> : null}
      <dl>
        {h ? <><dt>Predicted</dt><dd>{predictionText(view, h)}</dd></> : null}
        {m && run ? (
          <>
            <dt>Measured</dt>
            <dd>
              <button className="link mono" onClick={() => onSelectRun(run.id)}>{run.id}</button>{' '}
              {ms(m.latency_ms)}, {m.tflops.toFixed(1)} TFLOPS, max error {sci(m.max_error)}, adversarial {sci(m.adv_error)}
            </dd>
            <dt><Def term="speedup">Delta</Def></dt>
            {m.speedups ? <dd data-testid="window-speedups"><Def term="noise">{windowText(m)}</Def>, same timing window</dd> : <dd>
              {base ? <>{times(base / m.latency_ms)} vs previous best ({ms(base)})</> : naive ? <>{times(naive / m.latency_ms)} vs naive</> : null}
              {eager ? <>, {times(eager / m.latency_ms)} vs eager</> : null}
            </dd>}
          </>
        ) : null}
        {vd ? <><dt><Def term="bound">Judged against</Def></dt><dd data-testid="judged-bound">{boundText(vd.bound)}</dd></> : null}
        {trail.length > 1 ? (
          <>
            <dt>Verdict history</dt>
            <dd data-testid="verdict-trail">
              {trail.map((t, i) => (
                <div key={i} className="inline small"><VerdictChip verdict={t.verdict} provisional={t.provisional} /> {boundText(t.bound)}, {clock(t.t)}</div>
              ))}
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
      {onToggle ? <button className="link small" style={{ alignSelf: 'flex-start' }} onClick={onToggle}>Collapse</button> : null}
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
  const [opened, setOpened] = useState<number[]>([])
  const latest = view.feed.map(f => f.kind).lastIndexOf('summary')
  const toggle = (i: number) => setOpened(o => (o.includes(i) ? o.filter(x => x !== i) : [...o, i]))
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
  const items = view.feed.length
  useEffect(() => {
    if (!items) return
    const box = end.current?.closest('.col') as HTMLElement | null
    const cards = box?.querySelectorAll<HTMLElement>('[data-testid="summary-card"]')
    const last = cards?.[cards.length - 1] ?? null
    // follow the newest item, but never scroll the newest summary card's top out of view
    if (box && end.current && last && Date.now() - touched.current > 8000)
      box.scrollTop = Math.min(end.current.offsetTop - box.clientHeight + 12, last.offsetTop - 12)
  }, [items])

  const pending = view.runs.filter(r => r.finished === undefined)
  const lastVerdict = Math.max(-1, ...Object.values(view.trail).flat().map(t => t.t))
  const lastProposal = Math.max(-1, ...view.hypotheses.map(h => h.t))
  const choosing = !!p.working?.choosing && view.started && !view.finished && pending.length === 0
    && lastVerdict > lastProposal && view.branches.some(b => b.status === 'running')

  const optionLabel = (help: string, option: string) =>
    view.helps.find(h => h.help.id === help)?.help.options.find(o => o.id === option)?.label ?? option

  return (
    <div className="section" aria-label="summary feed">
      <div className="section-head"><h2>Summaries</h2><span className="muted small">{view.feed.filter(f => f.kind === 'summary').length}</span></div>
      {view.feed.length === 0 ? <p className="empty">No runs yet.</p> : null}
      {view.feed.map((f, i) => {
        switch (f.kind) {
          case 'summary':
            return (
              <SummaryCard
                key={i} view={view} item={f} onSelectRun={p.onSelectRun} onPostMortem={p.onPostMortem}
                collapsed={i !== latest && !opened.includes(i)} onToggle={i === latest ? undefined : () => toggle(i)}
              />
            )
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
                <span className="mono">{clock(f.t)}</span> <span className="strong">Exported for Papiers import:</span> {f.items.map((x, j) => <span key={j}>{j ? '; ' : ''}{x.title} <span className="mono">{x.file}</span></span>)} <MockTag on={f.mock} />
              </p>
            )
          case 'note':
            return (
              <p key={i} className="line" title={f.note?.excerpt}>
                <span className="mono">{clock(f.t)}</span>{' '}
                {f.note ? <><span className="strong">Read Papiers:</span> <NoteChip view={view} id={f.note.id} /></> : <><span className="strong">Searched Papiers:</span> no matching notes for "{f.query ?? 'this topic'}"</>}
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
            return <HybridCard key={i} view={view} item={f} onSend={p.onSend} onDecide={p.onHybrid} />
        }
      })}
      {p.working?.runs ? pending.map(r => {
        const h = hypothesisOf(view, r.hypothesis)
        return (
          <article key={r.id} className="card working" data-testid="pending-run" aria-live="polite">
            <h3>Running <span className="mono">{r.id}</span> on <span className="mono">{r.branch}</span></h3>
            {h ? <p className="small"><span className="muted">Predicted:</span> {predictionText(view, h)}</p> : null}
            <p className="meta"><span className="mono">{h?.id}</span><span>started {clock(r.started)}</span></p>
          </article>
        )
      }) : null}
      {choosing ? <p className="line working-line" data-testid="choosing" aria-live="polite">Choosing next hypothesis</p> : null}
      <div ref={end} />
    </div>
  )
}
