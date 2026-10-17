import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import type { Command, DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { Player } from './player'
import { initialView, reduce } from './state/reducer'
import { compress, loadLog, type Source } from './source'
import { connectLive, defaultUrl, type LiveStatus } from './live'

export interface Sent { cmd: Command; t: number; delivered: boolean }

export function useSession(src: Source) {
  const [view, dispatch] = useReducer(reduce, undefined, () => initialView())
  const [events, setEvents] = useState<DemoEvent[]>()
  const [speed, setSpeedState] = useState(1)
  const [paused, setPaused] = useState(false)
  const [ended, setEnded] = useState(false)
  const [now, setNow] = useState(0)
  const [epoch, setEpoch] = useState(0)
  const [status, setStatus] = useState<LiveStatus>()
  const [error, setError] = useState<string>()
  const [sent, setSent] = useState<Sent[]>([])
  const [notices, setNotices] = useState<string[]>([])
  const player = useRef<Player | undefined>(undefined)
  const live = useRef<ReturnType<typeof connectLive> | undefined>(undefined)
  const last = useRef({ t: 0, wall: Date.now() })
  const speedRef = useRef(speed)

  useEffect(() => {
    if (src.mode === 'mock') setEvents(demoScript)
    else if (src.mode === 'replay') {
      if (!src.file) setError('Replay mode needs a log, for example ?mode=replay&file=/runs/session.jsonl')
      else loadLog(src.file).then(e => setEvents(compress(e)), (err: Error) => setError(err.message))
    }
  }, [src.mode, src.file])

  useEffect(() => {
    if (!events) return
    dispatch({ type: 'ui/reset' })
    const p = new Player(events, dispatch, speedRef.current)
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
      } else if (src.mode === 'live') setNow(last.current.t + Date.now() - last.current.wall)
    }, 200)
    return () => clearInterval(id)
  }, [src.mode])

  const sync = () => {
    const p = player.current!
    setPaused(p.paused)
    setNow(p.position())
    setEnded(p.ended)
  }

  const controls = {
    pause: () => { player.current?.pause(); sync() },
    resume: () => { player.current?.play(); sync() },
    step: () => { player.current?.step(); sync() },
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
    view, dispatch, controls, send, sent, speed, paused, ended, now, status, error, notices,
    retry: () => live.current?.retry(),
  }
}
