import { useEffect, useState } from 'react'
import type { Lens } from '@contract/types'
import { DEFAULT_REPLAY, JUDGING, type Source } from './source'
import { chapters, type Chapter } from './chapters'
import { Preflight, ResultCard, Stalled } from './components/Final'
import { useSession } from './useSession'
import { TopBar } from './components/TopBar'
import { LensTabs } from './components/LensTabs'
import { DirectivePanel } from './components/DirectivePanel'
import { SummaryFeed } from './components/SummaryFeed'
import { BranchGraph } from './components/BranchGraph'
import { AdvPanel, MetricChart } from './components/MetricChart'
import { RunDetail } from './components/RunDetail'
import { HelpCard } from './components/HelpCard'
import { PostMortemCard } from './components/PostMortem'
import { Banner } from './components/Banner'
import { PapiersPanel } from './components/PapiersPanel'
import { Ledger, ReportView } from './components/Tier2'
import { clock, latestRun } from './state/derive'

const typing = (el: EventTarget | null) => el instanceof HTMLElement && el.closest('input, textarea, select, [contenteditable]') !== null

export function App({ source }: { source: Source }) {
  const s = useSession(source)
  const { view, dispatch, controls } = s
  const [logOpen, setLogOpen] = useState(false)
  const [stopping, setStopping] = useState(false)
  const newSession = async () => {
    if (view.started && !view.finished && !window.confirm('Stop this session and start a new one?')) return
    const res = await fetch('/restart', { method: 'POST' })
    if (res.ok) window.location.reload()
  }
  const [closedPm, setClosedPm] = useState<string[]>([])
  const [tab, setTab] = useState<'research' | 'report' | 'result'>('research')
  // the result screen opens once when the session finishes, and a restart returns to the research view
  const [wasFinished, setWasFinished] = useState(view.finished)
  if (wasFinished !== view.finished) {
    setWasFinished(view.finished)
    setTab(view.finished ? 'result' : 'research')
  }
  const openPm = (h: string) => {
    setClosedPm(c => c.filter(x => x !== h))
    requestAnimationFrame(() => {
      const el = document.getElementById(`pm-${h}`)
      const box = el?.closest('.col')
      if (el && box) box.scrollTop = el.offsetTop - 12
    })
  }
  const setLens = (lens: Lens) => dispatch({ type: 'ui/lens', lens })
  const selectRun = (run: string) => dispatch({ type: 'ui/select', selected: { run, branch: view.runs.find(r => r.id === run)?.branch } })
  const selectBranch = (branch: string) => dispatch({ type: 'ui/select', selected: { branch, run: latestRun(view, branch)?.id } })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e.target) || e.metaKey || e.ctrlKey || e.altKey) return
      if (e.key === '1') setLens('speed')
      else if (e.key === '2') setLens('precision')
      else if (e.key === 'Escape') dispatch({ type: 'ui/select', selected: {} })
      else if (source.mode === 'live') return
      else if (e.key === ' ') (s.paused ? controls.resume : controls.pause)()
      else if (e.key === 'ArrowRight') controls.step()
      else if (e.key === 'ArrowLeft') controls.prev()
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const marks = chapters(s.events)
  const jump = (c: Chapter) => {
    if (source.mode !== 'live') controls.seek(c.index)
    const run = s.events.find(e => e.type === 'run_started' && e.hypothesis === c.hypothesis)
    if (run?.type === 'run_started') dispatch({ type: 'ui/select', selected: { run: run.run, branch: run.branch } })
    setTab(c.key === 'end' ? 'result' : 'research')
  }
  const save = () => {
    const blob = new Blob([s.events.map(e => JSON.stringify(e)).join('\n') + '\n'], { type: 'application/x-ndjson' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `war-room-${new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')}.jsonl`
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const waiting = view.finished ? [] : [...new Set(view.helps.filter(h => !h.answer).map(h => h.help.target))]
  const mine = view.helps.filter(h => h.help.target === view.lens)
  const open = mine.filter(h => !h.answer && !view.finished)
  const answered = mine.filter(h => h.answer || view.finished)

  return (
    <div className="app" data-lens={view.lens}>
      <TopBar
        mode={source.mode} mock={view.mock} recording={view.log.some(l => l.text.startsWith('Recording mode'))} judging={JUDGING} now={s.now} paused={s.paused} ended={s.ended} speed={s.speed}
        compression={s.compression} chapters={marks} autoPause={s.autoPause}
        onPause={controls.pause} onResume={controls.resume} onStep={controls.step}
        onPrev={controls.prev} canPrev={s.index > 0} onMode={m => { window.location.search = `?mode=${m}` }}
        onSpeed={controls.setSpeed} onRestart={controls.restart} onChapter={jump} onAutoPause={controls.setAutoPause}
        onStop={() => { setStopping(true); s.send({ type: 'stop' }) }} onSave={save} onNewSession={newSession}
        live={{ started: view.started, finished: view.finished, stopping }}
      />
      <div className="accent-line" />
      <Banner error={s.error} status={s.status} onRetry={s.retry} />
      <main className="main">
        <aside className="col left" aria-label="steering">
          <LensTabs lens={view.lens} waiting={waiting} onChange={setLens} />
          {open.map(h => <HelpCard key={h.help.id} view={view} hv={h} now={s.now} sent={s.sent} onSend={s.send} />)}
          <DirectivePanel view={view} sent={s.sent} onSend={s.send} onSelectBranch={selectBranch} />
          {answered.map(h => <HelpCard key={h.help.id} view={view} hv={h} now={s.now} sent={s.sent} onSend={s.send} />)}
          <hr className="divider" />
          <PapiersPanel view={view} />
        </aside>
        <section className="col center" aria-label="research">
          {view.report || view.finished ? (
            <div className="inline" role="tablist" aria-label="center view">
              <button role="tab" aria-selected={tab === 'research'} aria-pressed={tab === 'research'} onClick={() => setTab('research')}>Research</button>
              {view.report ? <button role="tab" aria-selected={tab === 'report'} aria-pressed={tab === 'report'} onClick={() => setTab('report')}>Report</button> : null}
              {view.finished ? <button role="tab" aria-selected={tab === 'result'} aria-pressed={tab === 'result'} onClick={() => setTab('result')}>Result</button> : null}
            </div>
          ) : null}
          {source.mode === 'live' && !view.started ? (
            <Preflight onStart={record => s.send(record ? { type: 'start', record } : { type: 'start' })} fallback={`?mode=replay&file=${encodeURIComponent(DEFAULT_REPLAY)}`} log={view.log} />
          ) : tab === 'result' && view.finished ? <ResultCard view={view} /> : tab === 'report' && view.report ? <ReportView view={view} /> : (
            <>
              <BranchGraph view={view} onSelectBranch={selectBranch} onSelectRun={selectRun} showPending={source.mode !== 'replay'} />
              <div className="chart-row">
                <MetricChart view={view} onSelectRun={selectRun} />
                <AdvPanel view={view} />
              </div>
            </>
          )}
        </section>
        <aside className="col right" aria-label="record">
          <Stalled view={view} live={source.mode === 'live'} onSend={s.send} />
          {view.postMortems.map(pm => (
            <PostMortemCard
              key={pm.hypothesis} view={view} pm={pm} open={!closedPm.includes(pm.hypothesis)}
              onToggle={o => setClosedPm(c => (o ? c.filter(x => x !== pm.hypothesis) : [...c, pm.hypothesis]))}
            />
          ))}
          <RunDetail view={view} onClose={() => dispatch({ type: 'ui/select', selected: {} })} />
          <SummaryFeed
            working={{ runs: source.mode !== 'replay', choosing: source.mode === 'live' }}
            view={view} onSelectRun={selectRun} onPostMortem={openPm} onReport={() => setTab('report')}
            onSend={s.send} onHybrid={approved => dispatch({ type: 'ui/hybrid', approved })}
          />
          <hr className="divider" />
          <Ledger view={view} />
        </aside>
      </main>
      <div className="drawer">
        <button aria-expanded={logOpen} onClick={() => setLogOpen(o => !o)}>
          Agent log ({view.log.length}) {logOpen ? 'hide' : 'show'}
        </button>
        {logOpen ? (
          <div className="log">
            {view.log.map((l, i) => <div key={i}><span className="mono muted">{clock(l.t)}</span> {l.text}</div>)}
            {s.notices.map((n, i) => <div key={`n${i}`} className="muted">client: {n}</div>)}
          </div>
        ) : null}
      </div>
    </div>
  )
}
