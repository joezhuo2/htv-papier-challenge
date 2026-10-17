import type {
  Bound, Branch, Check, DemoEvent, Directive, HelpRequest, Hypothesis, LedgerEntry, Lens, Metrics, PapiersNote, Params, PostMortem, Report, Summary, Verdict,
} from '@contract/types'

export interface VerdictView { verdict: Verdict; provisional: boolean; reason: string; checks: Check[]; t: number; bound: Bound }

export interface BranchView extends Branch { t: number; reason?: string; index: number }

export interface HypothesisView extends Hypothesis { t: number; mock: boolean }

export interface RunView {
  id: string
  branch: string
  hypothesis: string
  index: number
  started: number
  finished?: number
  params?: Params
  metrics?: Metrics
  assumption?: boolean
  failed?: boolean
  error?: string
}

export interface DirectiveView extends Directive { t: number; affected?: string; mock: boolean }

export interface ConflictView { directives: string[]; reason: string; t: number; branch?: string; forked?: string }

export interface HelpView {
  help: HelpRequest
  t: number
  mock: boolean
  answer?: { option: string; note?: string; by: Lens; t: number }
  expired: boolean
  cancelled?: string
}

export interface NoteView extends PapiersNote { t: number; query?: string }

// every Papiers read in the log, with or without a returned note
export interface ReadView { t: number; query?: string; note?: NoteView; mock: boolean }

type Base = { t: number; mock: boolean }
export type FeedItem = Base & (
  | { kind: 'summary'; summary: Summary; run?: string; hypothesis?: string; verdict?: VerdictView }
  | { kind: 'plan_change'; help: string; text: string }
  | { kind: 'answer'; help: string; option: string; note?: string; by: Lens }
  | { kind: 'export'; items: { kind: string; title: string; file: string }[] }
  | { kind: 'note'; note?: NoteView; query?: string }
  | { kind: 'branch'; branch: string; reason: string }
  | { kind: 'hybrid'; branches: string[]; text: string }
  | { kind: 'report' }
)

export interface Selection { branch?: string; run?: string }

export interface View {
  t: number
  mock: boolean
  started: boolean
  finished: boolean
  baselines?: { eager: Metrics; naive: Metrics }
  branches: BranchView[]
  hypotheses: HypothesisView[]
  runs: RunView[]
  verdicts: Record<string, VerdictView>
  history: Record<string, Verdict[]>
  trail: Record<string, VerdictView[]> // every verdict per hypothesis in order, each with the bound it was judged against
  bound?: Bound // bound of the latest verdict, the active one
  errors: { t: number; kind: string; text: string; branch?: string; retry_in_s?: number }[]
  stalled?: { t: number; kind: string; text: string; branch?: string; retry_in_s?: number } // an agent_error not yet followed by progress
  directives: DirectiveView[]
  conflicts: ConflictView[]
  helps: HelpView[]
  postMortems: PostMortem[]
  notes: NoteView[]
  reads: ReadView[]
  feed: FeedItem[]
  log: { t: number; text: string }[]
  report?: Report
  hybrid?: { branches: string[]; text: string; t: number; approved?: boolean; dismissed?: boolean }
  ledger?: LedgerEntry[]
  best?: Report['best']
  lens: Lens
  selected: Selection
}

export type UiAction =
  | { type: 'ui/lens'; lens: Lens }
  | { type: 'ui/select'; selected: Selection }
  | { type: 'ui/reset' }
  | { type: 'ui/hybrid'; approved: boolean }

export type Action = DemoEvent | UiAction

export const initialView = (lens: Lens = 'speed'): View => ({
  t: 0, mock: false, started: false, finished: false,
  branches: [], hypotheses: [], runs: [], verdicts: {}, history: {}, trail: {}, errors: [], directives: [], conflicts: [], helps: [],
  postMortems: [], notes: [], reads: [], feed: [], log: [], lens, selected: {},
})

const setBranch = (v: View, id: string, patch: Partial<BranchView>) =>
  v.branches.map(b => (b.id === id ? { ...b, ...patch } : b))

// a directive or conflict is linked to the first branch that changes after it
const linkBranch = (v: View, id: string, created: boolean): View => ({
  ...v,
  directives: v.directives.map(d => (d.affected ? d : { ...d, affected: id })),
  conflicts: created ? v.conflicts.map(c => (c.forked ? c : { ...c, forked: id })) : v.conflicts,
})

const expire = (v: View, t: number): View => {
  if (!v.helps.some(h => !h.answer && !h.expired && t >= h.t + h.help.deadline_s * 1000)) return v
  return { ...v, helps: v.helps.map(h => (!h.answer && !h.expired && t >= h.t + h.help.deadline_s * 1000 ? { ...h, expired: true } : h)) }
}

export function reduce(prev: View, a: Action): View {
  if (a.type.startsWith('ui/')) return reduceUi(prev, a as UiAction)
  const e = a as DemoEvent
  if (e.type === 'session_started') {
    const v = initialView(prev.lens)
    return {
      ...v, t: e.t, mock: !!e.mock, started: true, baselines: e.baselines,
      branches: [{ ...e.branch, t: e.t, index: 0 }],
    }
  }
  const mock = !!e.mock
  let v: View = expire({ ...prev, t: Math.max(prev.t, e.t), mock: prev.mock || mock }, e.t)
  if (v.stalled && e.type !== 'agent_error' && e.type !== 'agent_log') v = { ...v, stalled: undefined }
  switch (e.type) {
    case 'papiers_context': {
      const note = e.note && { ...e.note, t: e.t, query: e.query }
      return {
        ...v,
        notes: note && !v.notes.some(n => n.id === note.id) ? [...v.notes, note] : v.notes,
        reads: [...v.reads, { t: e.t, query: e.query, note, mock }],
        feed: [...v.feed, { kind: 'note', note, query: e.query, t: e.t, mock }],
      }
    }
    case 'directive_added':
      return { ...v, directives: [...v.directives, { ...e.directive, t: e.t, mock }] }
    case 'conflict_detected': {
      const lenses = v.directives.filter(d => e.directives.includes(d.id))
      const first = lenses[0]
      const branch = first && [...v.branches].reverse().find(b => b.lens === first.lens)?.id
      return { ...v, conflicts: [...v.conflicts, { directives: e.directives, reason: e.reason, t: e.t, branch }] }
    }
    case 'branch_created': {
      v = linkBranch(v, e.branch.id, true)
      const branches = [...v.branches.filter(b => b.id !== e.branch.id), { ...e.branch, t: e.t, reason: e.reason, index: v.branches.length }]
      const feed: FeedItem[] = e.reason ? [...v.feed, { kind: 'branch', branch: e.branch.id, reason: e.reason, t: e.t, mock }] : v.feed
      const hybrid = v.hybrid && e.branch.parents.length > 1 && v.hybrid.branches.every(b => e.branch.parents.includes(b))
        ? { ...v.hybrid, approved: true } : v.hybrid
      return { ...v, branches, feed, hybrid }
    }
    case 'branch_status':
      v = linkBranch(v, e.branch, false)
      return { ...v, branches: setBranch(v, e.branch, { status: e.status }) }
    case 'hypothesis_proposed':
      return { ...v, hypotheses: [...v.hypotheses, { ...e.hypothesis, t: e.t, mock }] }
    case 'run_started':
      return { ...v, runs: [...v.runs, { id: e.run, branch: e.branch, hypothesis: e.hypothesis, index: v.runs.length, started: e.t }] }
    case 'run_finished': {
      const assumption = v.helps.some(h => h.expired && !h.answer && h.help.blocks === e.run.branch)
      // the agent sends null latency for a run that failed to compile or timed out
      const m = e.run.metrics
      const failed = m.latency_ms === null
      const done = { params: e.run.params, metrics: m.latency_ms === null ? undefined : m, finished: e.t, assumption, failed, error: e.error }
      const known = v.runs.some(r => r.id === e.run.id)
      const runs = known
        ? v.runs.map(r => (r.id === e.run.id ? { ...r, ...done } : r))
        : [...v.runs, { id: e.run.id, branch: e.run.branch, hypothesis: e.run.hypothesis, index: v.runs.length, started: e.t, ...done }]
      return { ...v, runs }
    }
    case 'verdict': {
      const vd = { verdict: e.verdict, provisional: e.provisional, reason: e.reason, checks: e.checks, t: e.t, bound: e.bound }
      return {
        ...v,
        verdicts: { ...v.verdicts, [e.hypothesis]: vd },
        history: { ...v.history, [e.hypothesis]: [...(v.history[e.hypothesis] ?? []), e.verdict] },
        trail: { ...v.trail, [e.hypothesis]: [...(v.trail[e.hypothesis] ?? []), vd] },
        bound: e.bound,
      }
    }
    case 'post_mortem':
      return { ...v, postMortems: [...v.postMortems, e.post_mortem] }
    case 'summary': {
      // older logs carry no hypothesis id, so they fall back to the newest finished run on the branch
      const run = e.summary.hypothesis
        ? v.runs.find(r => r.hypothesis === e.summary.hypothesis)
        : [...v.runs].reverse().find(r => r.branch === e.summary.branch && r.finished !== undefined)
      const verdict = run && v.verdicts[run.hypothesis]
      return { ...v, feed: [...v.feed, { kind: 'summary', summary: e.summary, run: run?.id, hypothesis: run?.hypothesis, verdict, t: e.t, mock }] }
    }
    case 'help_requested': {
      // a reissued request (fresh evidence after a verdict flip, or moved to a successor branch) replaces the card in place
      const hv = { help: e.help, t: e.t, mock, expired: false }
      return v.helps.some(h => h.help.id === e.help.id)
        ? { ...v, helps: v.helps.map(h => (h.help.id === e.help.id ? hv : h)) }
        : { ...v, helps: [...v.helps, hv] }
    }
    case 'help_cancelled':
      return { ...v, helps: v.helps.map(h => (h.help.id === e.help ? { ...h, cancelled: e.reason } : h)) }
    case 'help_answered': {
      const answer = { option: e.option, note: e.note, by: e.by, t: e.t }
      return {
        ...v,
        helps: v.helps.map(h => (h.help.id === e.help ? { ...h, answer } : h)),
        feed: [...v.feed, { kind: 'answer', help: e.help, option: e.option, note: e.note, by: e.by, t: e.t, mock }],
      }
    }
    case 'plan_change':
      return { ...v, feed: [...v.feed, { kind: 'plan_change', help: e.help, text: e.text, t: e.t, mock }] }
    case 'agent_log':
      // provider failover and explanation fallbacks are diagnostics, older logs still carry them
      return DIAGNOSTIC.test(e.text) ? v : { ...v, log: [...v.log, { t: e.t, text: e.text }] }
    case 'findings_exported':
      return { ...v, feed: [...v.feed, { kind: 'export', items: e.items, t: e.t, mock }] }
    case 'report_ready':
      return { ...v, report: e.report, feed: [...v.feed, { kind: 'report', t: e.t, mock }] }
    case 'hybrid_proposed':
      return {
        ...v,
        hybrid: { branches: e.branches, text: e.text, t: e.t },
        feed: [...v.feed, { kind: 'hybrid', branches: e.branches, text: e.text, t: e.t, mock }],
      }
    case 'agent_error': {
      const err = { t: e.t, kind: e.kind, text: e.text, branch: e.branch, retry_in_s: e.retry_in_s }
      return { ...v, errors: [...v.errors, err], stalled: e.kind === 'harness_timeout' ? v.stalled : err }
    }
    case 'session_finished':
      return { ...v, finished: true, ledger: e.ledger, best: e.best, helps: v.helps.map(h => (h.answer ? h : { ...h, expired: true })) }
  }
}

function reduceUi(v: View, a: UiAction): View {
  switch (a.type) {
    case 'ui/lens': return { ...v, lens: a.lens }
    case 'ui/select': return { ...v, selected: a.selected }
    case 'ui/reset': return initialView(v.lens)
    case 'ui/hybrid': return v.hybrid ? { ...v, hybrid: { ...v.hybrid, approved: a.approved || undefined, dismissed: !a.approved || undefined } } : v
  }
}

const DIAGNOSTIC = /^Explanation model unavailable|failed \(\w+\), answered by /

export const fold = (events: DemoEvent[], v = initialView()) => events.reduce(reduce, v)
