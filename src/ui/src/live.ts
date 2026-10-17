import type { Command, DemoEvent } from '@contract/types'

export type LiveStatus = { state: 'connecting' | 'open' } | { state: 'closed'; retryIn: number; attempt: number }

interface Socket {
  readyState: number
  send(data: string): void
  close(): void
  onopen: ((ev: unknown) => void) | null
  onclose: ((ev: unknown) => void) | null
  onmessage: ((ev: { data: unknown }) => void) | null
}

export interface LiveOptions {
  url: string
  onEvent: (e: DemoEvent) => void
  onStatus: (s: LiveStatus) => void
  onError?: (msg: string) => void
  socket?: (url: string) => Socket
}

export const defaultUrl = (loc: Location) => `${loc.protocol === 'https:' ? 'wss' : 'ws'}://${loc.host}/ws`

export const backoff = (attempt: number) => Math.min(8000, 1000 * 2 ** attempt)

export const parseMessage = (data: unknown): DemoEvent[] => {
  const msg = JSON.parse(String(data))
  const list = Array.isArray(msg) ? msg : [msg]
  for (const e of list) if (typeof e?.type !== 'string' || typeof e?.t !== 'number') throw new Error('not a DemoEvent')
  return list
}

// keeps one WebSocket open, reconnects with backoff, and queues commands sent while disconnected
export function connectLive(o: LiveOptions) {
  const make = o.socket ?? (url => new WebSocket(url) as unknown as Socket)
  const queue: Command[] = []
  let ws: Socket
  let attempt = 0
  let timer: ReturnType<typeof setTimeout> | undefined
  let closed = false

  const open = () => {
    clearTimeout(timer)
    o.onStatus({ state: 'connecting' })
    ws = make(o.url)
    ws.onopen = () => {
      attempt = 0
      o.onStatus({ state: 'open' })
      while (queue.length) ws.send(JSON.stringify(queue.shift()))
    }
    ws.onmessage = ev => {
      try {
        parseMessage(ev.data).forEach(o.onEvent)
      } catch {
        o.onError?.(`Ignored a malformed message: ${String(ev.data).slice(0, 80)}`)
      }
    }
    ws.onclose = () => {
      if (closed) return
      const retryIn = backoff(attempt)
      o.onStatus({ state: 'closed', retryIn, attempt: ++attempt })
      timer = setTimeout(open, retryIn)
    }
  }
  open()

  return {
    send(cmd: Command) {
      if (ws.readyState === 1) ws.send(JSON.stringify(cmd))
      else queue.push(cmd)
    },
    retry: open,
    close() {
      closed = true
      clearTimeout(timer)
      ws.close()
    },
  }
}
