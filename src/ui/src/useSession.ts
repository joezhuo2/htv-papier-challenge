import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import type { Command, DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { Player } from './player'
import { initialView, reduce } from './state/reducer'
import { compress, loadLog, type Source } from './source'
import { chapters } from './chapters'
import { connectLive, defaultUrl, type LiveStatus } from './live'

export interface Sent { cmd: Command; t: number; delivered: boolean }

export function useSession(src: Source) {
  const [view, dispatch] = useReducer(reduce, undefined, () => initialView())
  const [events, setEvents] = useState<DemoEvent[]>()
  const [speed, setSpeedState] = useState(1)
  const [paused, setPaused] = useState(false)
  const [ended, setEnded] = useState(false)
  const [index, setIndex] = useState(0)
  const [now, setNow] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [status, setStatus] = useState<LiveStatus>()
  const [error, setError] = useState<string>()
  const [sent, setSent] = useState<Sent[]>([])
  const [notices, setNotices] = useState<string[]>([])
  const [compression, setCompression] = useState<{ from: number; to: number }>()
  const [autoPause, setAutoPauseState] = useState(false)
  const autoPauseRef = useRef(false)
  const liveEvents = useRef<DemoEvent[]>([])
  const [liveCount, setLiveCount] = useState(0)
  const player = useRef<Player | undefined>(undefined)
  const live = useRef<ReturnType<typeof connectLive> | undefined>(undefined)
  const last = useRef({ t: 0, wall: Date.now() })
  const speedRef = useRef(speed)

  useEffect(() => {
    if (src.mode === 'mock') setEvents(demoScript)
    else if (src.mode === 'replay') {
      if (!src.file) setError('Replay mode needs a log, for example ?mode=replay&file=/runs/session.jsonl')
      else loadLog(src.file).then(raw => {
        const c = compress(raw)
        setCompression({ from: raw.at(-1)!.t - raw[0].t, to: c.at(-1)!.t - c[0].t })
        setEvents(c)
      }, (err: Error) => setError(`Could not replay ${src.file}: ${err.message}`))
    }
  }, [src.mode, src.file])

  useEffect(() => {
    if (!events) return
    dispatch({ type: 'ui/reset' })
    const stops = new Set(chapters(events).map(c => c.index))
    const p: Player = new Player(events, e => {
      dispatch(e)
      if (autoPauseRef.current && stops.has(p.index)) p.pause()
    }, speedRef.current)
    player.current = p
    p.play()
    setPaused(false)
    setEnded(false)
    return () => p.cancel()
  }, [events, epoch])

  useEffect(() => {
    if (src.mode !== 'live') return
    const c = connectLive({
      url: src.ws ?? defaultUrl(window.location),
      onEvent: e => {
        last.current = { t: e.t, wall: Date.now() }
        if (e.type === 'session_started') liveEvents.current = []
        liveEvents.current.push(e)
        setLiveCount(liveEvents.current.length)
        dispatch(e)
      },
      onStatus: setStatus,
      onError: msg => setNotices(n => [...n, msg]),
    })
    live.current = c
    return () => c.close()
  }, [src.mode, src.ws])

  useEffect(() => {
    const id = setInterval(() => {
      const p = player.current
      if (p) {
        setNow(p.position())
        setEnded(p.ended)
        setPaused(p.paused)
        setIndex(p.index)
      } else if (src.mode === 'live') setNow(last.current.t + Date.now() - last.current.wall)
    }, 200)
    return () => clearInterval(id)
  }, [src.mode])

  const sync = () => {
    const p = player.current!
    setPaused(p.paused)
    setNow(p.position())
    setEnded(p.ended)
    setIndex(p.index)
  }

  const refold = (move: (p: Player) => void) => {
    const p = player.current
    if (!p) return
    move(p)
    dispatch({ type: 'ui/reset' })
    p.emitted().forEach(dispatch)
    sync()
  }

  const controls = {
    pause: () => { player.current?.pause(); sync() },
    resume: () => { player.current?.play(); sync() },
    step: () => { player.current?.step(); sync() },
    prev: () => refold(p => p.back()),
    seek: (n: number) => refold(p => p.seek(n)),
    setAutoPause: (on: boolean) => {
      autoPauseRef.current = on
      setAutoPauseState(on)
    },
    setSpeed: (s: number) => {
      speedRef.current = s
      setSpeedState(s)
      player.current?.setSpeed(s)
    },
    restart: () => setEpoch(e => e + 1),
  }

  const send = useCallback((cmd: Command) => {
    const delivered = src.mode === 'live'
    if (delivered) live.current?.send(cmd)
    setSent(s => [...s, { cmd, t: Date.now(), delivered }])
  }, [src.mode])

  return {
    view, dispatch, controls, send, sent, speed, paused, ended, index, now, status, error, notices, compression, autoPause,
    // every event of the session: the whole log in mock and replay, what has arrived so far in live mode
    events: src.mode === 'live' ? liveEvents.current.slice(0, liveCount) : events ?? [],
    retry: () => live.current?.retry(),
  }
}
