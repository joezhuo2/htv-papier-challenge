import type { LiveStatus } from '../live'

export function Banner({ error, status, onRetry }: { error?: string; status?: LiveStatus; onRetry: () => void }) {
  if (error) return <div className="banner" role="alert">{error}</div>
  if (status?.state === 'closed') {
    return (
      <div className="banner" role="alert" data-testid="reconnect-banner">
        {status.attempt > 1 ? 'No agent server is reachable at /ws.' : 'Connection to the agent dropped.'} Reconnecting in {Math.round(status.retryIn / 1000)}s (attempt {status.attempt}).
        {status.attempt > 1 ? <span className="mono">Start it with: uv run uvicorn server.app:app --port 8000</span> : null}
        <button className="link" onClick={onRetry}>Retry now</button>
      </div>
    )
  }
  if (status?.state === 'connecting') return <div className="banner muted" style={{ borderColor: 'var(--border)', color: 'var(--muted)' }}>Connecting to the agent...</div>
  return <div />
}
