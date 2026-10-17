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
  // speedups against references timed in the same window as this run, and the relative half-range of the per-round ratios
  speedups?: Partial<Record<'naive' | 'eager' | 'parent', number>>
  spreads?: Partial<Record<'naive' | 'eager' | 'parent', number>>
  // from the compiled Triton kernel: registers per thread and 32-bit words spilled to local memory per thread
  n_regs?: number
  n_spills?: number
}

// a run that failed to compile, crashed or timed out sends null for every measured field
export type FailedMetrics = { [K in 'latency_ms' | 'tflops' | 'max_error' | 'adv_error' | 'vram_mb']: null } &
  Pick<Metrics, 'compiled_ok' | 'correct_ok' | 'n_regs' | 'n_spills'>

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

// The error bound a run is judged against. Speed tolerates more error on the standard batch, Precision gates both batches at 1e-3.
export interface Bound {
  standard: number
  adversarial: number | null // null while no lens gates the adversarial batch, which is still measured on every run
  set_by: Lens | 'default'
}

export interface Prediction {
  min_speedup: number
  vs: 'naive' | 'parent_best'
  max_error: number // equals bound.standard
  bound: Bound
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

// one retrieved Papiers item. kind and the fields after it are absent in logs recorded before highlights were read
export interface PapiersNote {
  id: string
  title: string
  excerpt: string
  kind?: 'highlight' | 'note' | 'passage' | 'conversation'
  document_title?: string
  highlight_text?: string
  annotation?: string // the user's own words on a highlight
}

export interface PostMortem {
  hypothesis: string
  expected: string
  observed: string
  cause: string
  pivot: string
  sweep: { k: number; adv_error: number }[]
  note?: string // the Papiers id retrieved for the measured cause, when one matched
}

export interface Delta {
  metric: string
  before: number
  after: number
  unit: string
}

export interface Summary {
  branch: string
  hypothesis?: string // the hypothesis this summary is about, absent in logs from before 16:30
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

// computed by the agent from the log at the end of a session, sent just before session_finished
export interface FinalReport {
  best: { run: string; hypothesis: string; branch: string; params: Params; latency_ms: number; vs_naive: number; vs_eager: number } | null
  ruled_out: string[]
  next: string[]
  human: string[] // decisions only a lens owner can make: unanswered requests, an unapproved hybrid, a lens conflict
  hunches: { directive: string; lens: Lens; text: string; outcome: string }[] // one per typed hunch, with its measured outcome or why it did not run
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
  // one event per read: the search query and one returned note, highlight or conversation, or no note when the search returned nothing
  | Ev<'papiers_context', { query?: string; note?: PapiersNote }>
  | Ev<'directive_added', { directive: Directive }>
  | Ev<'conflict_detected', { directives: string[]; reason: string }>
  | Ev<'branch_created', { branch: Branch; reason?: string }>
  | Ev<'branch_status', { branch: string; status: BranchStatus }>
  | Ev<'hypothesis_proposed', { hypothesis: Hypothesis }>
  | Ev<'run_started', { run: string; branch: string; hypothesis: string }>
  | Ev<'run_finished', { run: Omit<Run, 'metrics'> & { metrics: Metrics | FailedMetrics }; error?: string }> // error is set when the run failed to compile or hit the harness timeout
  | Ev<'verdict', { hypothesis: string; verdict: Verdict; provisional: boolean; reason: string; checks: Check[]; bound: Bound }>
  | Ev<'post_mortem', { post_mortem: PostMortem }>
  | Ev<'summary', { summary: Summary }>
  | Ev<'help_requested', { help: HelpRequest }>
  | Ev<'help_answered', { help: string; option: string; note?: string; by: Lens }>
  // a pending request withdrawn because a verdict flip rejected its branch and nothing continues it. A reissued request reuses its id.
  | Ev<'help_cancelled', { help: string; reason: string }>
  | Ev<'plan_change', { help: string; text: string }>
  | Ev<'agent_log', { text: string }>
  // Papiers is read-only, findings are exported as local files for manual import
  | Ev<'findings_exported', { items: { kind: 'summary' | 'negative_result' | 'ledger' | 'answer'; title: string; file: string }[] }>
  | Ev<'report_ready', { report: Report }>
  | Ev<'hybrid_proposed', { branches: string[]; text: string }>
  | Ev<'final_report', { report: FinalReport }>
  | Ev<'session_finished', { ledger: LedgerEntry[]; best: Report['best'] }>
  // a visible failure state: the loop waits for a retry command (or retries by itself after retry_in_s) instead of looking frozen
  | Ev<'agent_error', { kind: 'llm_timeout' | 'llm_error' | 'harness_timeout'; text: string; branch?: string; retry_in_s?: number }>

// client to server, not replayed by the mock because the scripted events already include their effects
export type Command =
  | { type: 'add_directive'; lens: Lens; text: string }
  | { type: 'answer_help'; help: string; option: string; note?: string; by: Lens }
  | { type: 'approve_hybrid'; branches: string[] }
  // record runs the full loop for up to RECORD_HYPS hypotheses and saves the log as a replay, the default start keeps the live demo cap
  | { type: 'start'; record?: boolean }
  | { type: 'stop' }
  | { type: 'retry' }
