import type { View } from '../state/reducer'
import { boundText, clock, hypothesisOf, lensName, ms, paramText, predictionText, runOf, sci, windowText } from '../state/derive'
import { InformedBy } from './PapiersPanel'
import { Glyph, VerdictChip } from './Glyph'

export function RunDetail({ view, onClose }: { view: View; onClose: () => void }) {
  const { run: rid, branch: bid } = view.selected
  const run = runOf(view, rid)
  const branch = view.branches.find(b => b.id === (run?.branch ?? bid))
  if (!run && !branch) return null
  const h = hypothesisOf(view, run?.hypothesis)
  const vd = h && view.verdicts[h.id]
  const m = run?.metrics

  return (
    <article className="card focus" aria-label="selection" data-testid="run-detail">
      <div className="row">
        <h3>{run ? <span className="mono">{run.id}</span> : null} {branch ? <>branch <span className="mono">{branch.id}</span></> : null}</h3>
        {h ? <VerdictChip verdict={vd?.verdict} provisional={vd?.provisional} /> : null}
        <span style={{ flex: 1 }} />
        <button className="small" onClick={onClose} title="Esc">close</button>
      </div>
      {branch ? <p className="small muted">{branch.label}, {lensName(branch.lens)} lens{branch.reason ? `. ${branch.reason}` : ''}</p> : null}
      {h ? (
        <>
          <p><span className="mono">{h.id}</span> {h.claim}</p>
          <InformedBy view={view} h={h} />
          <dl>
            <dt>Predicted</dt><dd>{predictionText(view, h)}</dd>
            {run?.params ? <><dt>Params</dt><dd className="mono">{paramText(run.params)}</dd></> : null}
            {m ? (
              <>
                <dt>Latency</dt><dd>{ms(m.latency_ms)}, {m.tflops.toFixed(1)} TFLOPS</dd>
                {m.speedups ? <><dt>Speedup</dt><dd>{windowText(m)}, same timing window</dd></> : null}
                <dt>Error</dt><dd>max {sci(m.max_error)}, adversarial {sci(m.adv_error)}</dd>
                <dt>VRAM</dt><dd>{m.vram_mb} MB</dd>
              </>
            ) : <><dt>Status</dt><dd>{run?.failed ? (run.error === 'timed out' ? 'timed out (15 s harness limit)' : run.error ? `failed: ${run.error}` : 'failed to compile or timed out') : 'running'}</dd></>}
          </dl>
          {(h ? view.trail[h.id] ?? [] : []).map((t, i, all) => (
            <div key={i} className="stack" data-testid="verdict-block" style={{ gap: 4 }}>
              <div className="inline small">
                <VerdictChip verdict={t.verdict} provisional={t.provisional} />
                <span>{all.length > 1 ? `verdict ${i + 1} of ${all.length}, ` : ''}{clock(t.t)}, {boundText(t.bound)}</span>
              </div>
              <table className="table">
                <thead><tr><th>check</th><th>predicted</th><th>observed</th><th /></tr></thead>
                <tbody>
                  {t.checks.map(c => {
                    const mark = c.pass === null ? 'inconclusive' : c.pass ? 'confirmed' : 'rejected'
                    return (
                      <tr key={c.name}>
                        <td>{c.name}</td><td className="mono">{c.predicted}</td><td className="mono">{c.observed}</td>
                        <td><span className={`verdict ${mark}`} style={{ border: 0, padding: 0 }}><Glyph mark={mark} />{c.pass === null ? 'noise' : c.pass ? 'pass' : 'fail'}</span></td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
              <p className="small">{t.reason}</p>
            </div>
          ))}
        </>
      ) : <p className="empty">No runs on this branch yet.</p>}
    </article>
  )
}
