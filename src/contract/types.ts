export type Lens = 'speed' | 'precision'
export type Verdict = 'confirmed' | 'rejected' | 'inconclusive'
export type BranchStatus = 'running' | 'waiting_on_human' | 'rejected' | 'merged' | 'done'

export interface Params {
  block_m: number
  block_n: number
  block_k: number
  warps: number
  stages: number
  acc: 'fp16' | 'fp32'
  gelu: 'exact' | 'approx'
}

export interface Metrics {
  latency_ms: number
  tflops: number
  max_error: number
  adv_error: number
  vram_mb: number
  compiled_ok: boolean
  correct_ok: boolean // default gate on the standard batch, adversarial gating comes from the precision lens
}

export interface Branch {
  id: string
  parents: string[]
  label: string
  lens: Lens | 'both'
  status: BranchStatus
}

export interface Directive {
  id: string
  lens: Lens
  text: string
  constraint: 'maximize_speed' | 'max_error'
  threshold?: number
}

export interface Prediction {
  min_speedup: number
  vs: 'naive' | 'parent_best'
  max_error: number
}

export interface Hypothesis {
  id: string
  branch: string
  claim: string
  prediction: Prediction
  params: Params
  note?: string
}

export interface Run {
  id: string
  branch: string
  hypothesis: string
  params: Params
  metrics: Metrics
}

// pass is null when the observed value sits inside the noise band
export interface Check {
  name: string
  predicted: string
  observed: string
  pass: boolean | null
}

export interface PostMortem {
  hypothesis: string
  expected: string
  observed: string
  cause: string
  pivot: string
  sweep: { k: number; adv_error: number }[]
}

export interface Delta {
  metric: string
  before: number
  after: number
  unit: string
}

export interface Summary {
  branch: string
  tried: string
  moved: Delta[]
  verdict: string
  confidence: 'high' | 'medium' | 'low'
  confidence_reason: string
  uncertainties: string[]
  would_change_mind: string
  next_step: string
}

export interface HelpRequest {
  id: string
  target: Lens
  question: string
  evidence: string[]
  options: { id: string; label: string }[]
  default: string
  blocks: string
  deadline_s: number
  what_changes: string
}

export interface Point {
  run: string
  label: string
  branch: string
  latency_ms: number
  adv_error: number
  reference?: boolean
}

export interface Report {
  points: Point[]
  pareto: string[]
  recommendation: string
  best: { run: string; vs_naive: number; vs_eager: number }
}

export interface LedgerEntry {
  hypothesis: string
  verdict: Verdict
  history: Verdict[]
}

// t is ms on the 3 minute demo timeline, mock is set on every scripted event and absent on real ones
type Ev<T extends string, P> = { t: number; mock?: true; type: T } & P

export type DemoEvent =
  | Ev<'session_started', { baselines: { eager: Metrics; naive: Metrics }; branch: Branch }>
  | Ev<'papiers_context', { note: { id: string; title: string; excerpt: string } }>
  | Ev<'directive_added', { directive: Directive }>
  | Ev<'conflict_detected', { directives: string[]; reason: string }>
  | Ev<'branch_created', { branch: Branch; reason?: string }>
  | Ev<'branch_status', { branch: string; status: BranchStatus }>
  | Ev<'hypothesis_proposed', { hypothesis: Hypothesis }>
  | Ev<'run_started', { run: string; branch: string; hypothesis: string }>
  | Ev<'run_finished', { run: Run }>
  | Ev<'verdict', { hypothesis: string; verdict: Verdict; provisional: boolean; reason: string; checks: Check[] }>
  | Ev<'post_mortem', { post_mortem: PostMortem }>
  | Ev<'summary', { summary: Summary }>
  | Ev<'help_requested', { help: HelpRequest }>
  | Ev<'help_answered', { help: string; option: string; note?: string; by: Lens }>
  | Ev<'plan_change', { help: string; text: string }>
  | Ev<'agent_log', { text: string }>
  | Ev<'papiers_write', { items: { kind: 'summary' | 'negative_result' | 'ledger' | 'answer'; title: string }[] }>
  | Ev<'report_ready', { report: Report }>
  | Ev<'hybrid_proposed', { branches: string[]; text: string }>
  | Ev<'session_finished', { ledger: LedgerEntry[]; best: Report['best'] }>

// client to server, not replayed by the mock because the scripted events already include their effects
export type Command =
  | { type: 'add_directive'; lens: Lens; text: string }
  | { type: 'answer_help'; help: string; option: string; note?: string; by: Lens }
  | { type: 'approve_hybrid'; branches: string[] }
