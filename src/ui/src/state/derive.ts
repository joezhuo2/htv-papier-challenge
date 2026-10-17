import type { Bound, Hypothesis, Metrics, Params, Verdict } from '@contract/types'
import type { BranchView, HypothesisView, RunView, View } from './reducer'

export const clock = (ms: number) => {
  const s = Math.max(0, Math.floor(ms / 1000))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

export const sci = (x: number) => x.toExponential(1).replace('.0e', 'e').replace('e+', 'e')
export const ms = (x: number) => `${x.toPrecision(3)} ms`
export const times = (x: number) => `${x.toFixed(2)}x`

export const paramText = (p: Params) =>
  `${p.block_m}x${p.block_n}x${p.block_k}, ${p.warps} warps, ${p.stages} stages, ${p.acc} acc, ${p.gelu} GELU`

export const ancestors = (branches: BranchView[], id: string): Set<string> => {
  const seen = new Set<string>()
  const walk = (b: string) => {
    if (seen.has(b)) return
    seen.add(b)
    branches.find(x => x.id === b)?.parents.forEach(walk)
  }
  walk(id)
  return seen
}

// best latency among confirmed, non provisional runs on the branch and its ancestors, finished before time t
export const parentBest = (v: View, branch: string, t: number): number | undefined => {
  const lineage = ancestors(v.branches, branch)
  const lat = v.runs
    .filter(r => r.metrics && r.finished! < t && lineage.has(r.branch))
    .filter(r => {
      const vd = v.verdicts[r.hypothesis]
      return vd && vd.verdict === 'confirmed' && !vd.provisional && vd.t < t
    })
    .map(r => r.metrics!.latency_ms)
  return lat.length ? Math.min(...lat) : undefined
}

export const baseLatency = (v: View, h: Hypothesis & { t: number }) =>
  h.prediction.vs === 'naive' ? v.baselines?.naive.latency_ms : parentBest(v, h.branch, h.t)

export const predictedLatency = (v: View, h: HypothesisView) => {
  const base = baseLatency(v, h)
  return base === undefined ? undefined : base / h.prediction.min_speedup
}

export const boundText = (b: Bound) =>
  b.set_by === 'default' ? `default ${sci(b.standard)}`
    : b.adversarial === null ? `standard under ${sci(b.standard)} (set by ${lensName(b.set_by)})`
      : b.adversarial === b.standard ? `standard and adversarial under ${sci(b.standard)} (set by ${lensName(b.set_by)})`
        : `standard under ${sci(b.standard)}, adversarial under ${sci(b.adversarial)} (set by ${lensName(b.set_by)})`

export const predictionText = (v: View, h: HypothesisView) => {
  const base = baseLatency(v, h)
  const vs = h.prediction.vs === 'naive' ? 'naive' : 'best so far'
  const at = base === undefined ? '' : ` (${ms(base)})`
  return `at least ${times(h.prediction.min_speedup)} vs ${vs}${at}, error ${boundText(h.prediction.bound)}`
}

export const papiersNote = (h?: Hypothesis) => h?.note?.match(/papiers note (\S+)/i)?.[1]

export const hypothesisOf = (v: View, id?: string) => v.hypotheses.find(h => h.id === id)
export const runOf = (v: View, id?: string) => v.runs.find(r => r.id === id)
export const latestRun = (v: View, branch: string) => [...v.runs].reverse().find(r => r.branch === branch)
export const runFor = (v: View, hypothesis: string) => [...v.runs].reverse().find(r => r.hypothesis === hypothesis && r.metrics)

export interface LedgerRow {
  hypothesis: string
  lens: string
  prediction: string
  result: string
  verdict?: Verdict
  provisional: boolean
  history: Verdict[]
  branch: string
  note?: string
}

export const ledgerRows = (v: View): LedgerRow[] =>
  v.hypotheses.map(h => {
    const final = v.ledger?.find(l => l.hypothesis === h.id)
    const vd = v.verdicts[h.id]
    const run = runFor(v, h.id)
    const lens = v.branches.find(b => b.id === h.branch)?.lens ?? ''
    return {
      hypothesis: h.id,
      lens,
      prediction: `${times(h.prediction.min_speedup)} vs ${h.prediction.vs === 'naive' ? 'naive' : 'best'}`,
      result: run?.metrics ? `${ms(run.metrics.latency_ms)}, adv ${sci(run.metrics.adv_error)}` : '',
      verdict: final?.verdict ?? vd?.verdict,
      provisional: !final && !!vd?.provisional,
      history: final?.history ?? v.history[h.id] ?? [],
      branch: h.branch,
      note: papiersNote(h),
    }
  })

export const speedup = (v: View, r: RunView) =>
  v.baselines && r.metrics ? v.baselines.naive.latency_ms / r.metrics.latency_ms : undefined

export const lensName = (l: string) => (l === 'speed' ? 'Speed' : l === 'precision' ? 'Precision' : 'Both')

// what the pivot dropped: params that changed between the rejected hypothesis and the first one tried on a branch forked from it
export const ruledOut = (v: View, hyp: string) => {
  const h = hypothesisOf(v, hyp)
  if (!h) return undefined
  const forks = v.branches.filter(b => b.parents.includes(h.branch)).map(b => b.id)
  const next = v.hypotheses.find(x => forks.includes(x.branch))
  if (!next) return `the ${h.id} configuration (${paramText(h.params)})`
  const keys = (Object.keys(h.params) as (keyof Params)[]).filter(k => h.params[k] !== next.params[k])
  return keys.map(k => `${k} ${h.params[k]}`).join(', ') + ` (the ${next.id} pivot keeps everything else)`
}

// same-window speedups with their measured spread, e.g. "1.45x +/- 0.02x vs best, 3.10x vs naive, 1.05x vs eager"
export const windowText = (m: Metrics) => {
  const names = { parent: 'best', naive: 'naive', eager: 'eager' } as const
  return (['parent', 'naive', 'eager'] as const)
    .filter(k => m.speedups?.[k] !== undefined)
    .map(k => `${times(m.speedups![k]!)} +/- ${(m.speedups![k]! * (m.spreads?.[k] ?? 0)).toFixed(2)}x vs ${names[k]}`)
    .join(', ')
}
