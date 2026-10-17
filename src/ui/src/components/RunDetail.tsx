import type { View } from '../state/reducer'
import { hypothesisOf, lensName, ms, papiersNote, paramText, predictionText, runOf, sci } from '../state/derive'
import { Glyph, VerdictChip } from './Glyph'

export function RunDetail({ view, onClose }: { view: View; onClose: () => void }) {
  const { run: rid, branch: bid } = view.selected
  const run = runOf(view, rid)
  const branch = view.branches.find(b => b.id === (run?.branch ?? bid))
  if (!run && !branch) return null
  const h = hypothesisOf(view, run?.hypothesis)
  const vd = h && view.verdicts[h.id]
  const m = run?.metrics
  const note = papiersNote(h)

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
          {note ? <span className="tag" style={{ alignSelf: 'flex-start' }}>Papiers note {note}</span> : null}
          <dl>
            <dt>Predicted</dt><dd>{predictionText(view, h)}</dd>
            {run?.params ? <><dt>Params</dt><dd className="mono">{paramText(run.params)}</dd></> : null}
            {m ? (
              <>
                <dt>Latency</dt><dd>{ms(m.latency_ms)}, {m.tflops.toFixed(1)} TFLOPS</dd>
                <dt>Error</dt><dd>max {sci(m.max_error)}, adversarial {sci(m.adv_error)}</dd>
                <dt>VRAM</dt><dd>{m.vram_mb} MB</dd>
              </>
            ) : <><dt>Status</dt><dd>running</dd></>}
          </dl>
          {vd ? (
            <table className="table">
              <thead><tr><th>check</th><th>predicted</th><th>observed</th><th /></tr></thead>
              <tbody>
                {vd.checks.map(c => {
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
          ) : null}
          {vd ? <p className="small">{vd.reason}</p> : null}
        </>
      ) : <p className="empty">No runs on this branch yet.</p>}
    </article>
  )
}
