import type { Lens } from '@contract/types'
import { lensName } from '../state/derive'

export function LensTabs({ lens, waiting, onChange, check }: { lens: Lens; waiting: Lens[]; onChange: (l: Lens) => void; check?: boolean }) {
  return (
    <div className="tabs" role="tablist" aria-label="lens">
      {(['speed', 'precision'] as Lens[]).map((l, i) => (
        <button
          key={l} role="tab" aria-selected={lens === l} onClick={() => onChange(l)}
          title={`Press ${i + 1}`} data-testid={`tab-${l}`}
        >
          {lensName(l)}
          {waiting.includes(l) ? <span className="badge" aria-label="help request waiting">help</span> : null}
          {check && l === 'precision' ? <span className="badge" aria-label="provisional result waiting for a Precision check">check</span> : null}
        </button>
      ))}
    </div>
  )
}
