import { useEffect, useState } from 'react'
import type { Lens } from '@contract/types'
import type { Source } from './source'
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
import { HybridCard, Ledger, ReportView } from './components/Tier2'
import { clock, latestRun } from './state/derive'

const typing = (el: EventTarget | null) => el instanceof HTMLElement && el.closest('input, textarea, select, [contenteditable]') !== null

export function App({ source }: { source: Source }) {
  const s = useSession(source)
  const { view, dispatch, controls } = s
  const [logOpen, setLogOpen] = useState(false)
  const [closedPm, setClosedPm] = useState<string[]>([])
  const [tab, setTab] = useState<'research' | 'report'>('research')
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
      else return
      e.preventDefault()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  const waiting = [...new Set(view.helps.filter(h => !h.answer).map(h => h.help.target))]
  const mine = view.helps.filter(h => h.help.target === view.lens)
  const open = mine.filter(h => !h.answer)
  const answered = mine.filter(h => h.answer)

  return (
    <div className="app" data-lens={view.lens}>
      <TopBar
        mode={source.mode} mock={view.mock} now={s.now} paused={s.paused} ended={s.ended} speed={s.speed}
        onPause={controls.pause} onResume={controls.resume} onStep={controls.step}
        onSpeed={controls.setSpeed} onRestart={controls.restart}
      />
      <div className="accent-line" />
      <Banner error={s.error} status={s.status} onRetry={s.retry} />
      <main className="main">
        <aside className="col left" aria-label="steering">
          <LensTabs lens={view.lens} waiting={waiting} onChange={setLens} />
          {open.map(h => <HelpCard key={h.help.id} view={view} hv={h} now={s.now} sent={s.sent} onSend={s.send} />)}
          <DirectivePanel view={view} sent={s.sent} onSend={s.send} onSelectBranch={selectBranch} />
          {answered.map(h => <HelpCard key={h.help.id} view={view} hv={h} now={s.now} sent={s.sent} onSend={s.send} />)}
        </aside>
        <section className="col center" aria-label="research">
          {view.report ? (
            <div className="inline" role="tablist" aria-label="center view">
              <button role="tab" aria-selected={tab === 'research'} aria-pressed={tab === 'research'} onClick={() => setTab('research')}>Research</button>
              <button role="tab" aria-selected={tab === 'report'} aria-pressed={tab === 'report'} onClick={() => setTab('report')}>Report</button>
            </div>
          ) : null}
          {tab === 'report' && view.report ? <ReportView view={view} /> : (
            <>
              <BranchGraph view={view} onSelectBranch={selectBranch} onSelectRun={selectRun} />
              <div className="chart-row">
                <MetricChart view={view} onSelectRun={selectRun} />
                <AdvPanel view={view} />
              </div>
            </>
          )}
        </section>
        <aside className="col right" aria-label="record">
          {view.postMortems.map(pm => (
            <PostMortemCard
              key={pm.hypothesis} view={view} pm={pm} open={!closedPm.includes(pm.hypothesis)}
              onToggle={o => setClosedPm(c => (o ? c.filter(x => x !== pm.hypothesis) : [...c, pm.hypothesis]))}
            />
          ))}
          <RunDetail view={view} onClose={() => dispatch({ type: 'ui/select', selected: {} })} />
          <SummaryFeed
            view={view} onSelectRun={selectRun} onPostMortem={openPm} onReport={() => setTab('report')}
            hybrid={f => <HybridCard view={view} item={f} onSend={s.send} onDecide={approved => dispatch({ type: 'ui/hybrid', approved })} />}
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
