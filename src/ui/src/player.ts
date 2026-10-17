import type { DemoEvent } from '@contract/types'

// Fires events at t / speed of wall time, like replay.ts, with pause, resume, step and speed changes. The cursor only moves forward one index at a time, so no event is ever skipped or reordered.
export class Player {
  private i = 0
  private pos = 0
  private anchor = 0
  private timer: ReturnType<typeof setTimeout> | undefined
  paused = true
  cancelled = false

  constructor(
    private events: DemoEvent[],
    private emit: (e: DemoEvent) => void,
    public speed = 1,
  ) {}

  get index() { return this.i }
  get ended() { return this.i >= this.events.length }

  position() {
    return this.paused ? this.pos : this.pos + (Date.now() - this.anchor) * this.speed
  }

  play() {
    if (!this.paused || this.cancelled || this.ended) return
    this.paused = false
    this.anchor = Date.now()
    this.schedule()
  }

  pause() {
    if (this.paused) return
    this.pos = this.position()
    this.paused = true
    clearTimeout(this.timer)
  }

  step() {
    this.pause()
    if (this.cancelled || this.ended) return
    const e = this.events[this.i++]
    this.pos = Math.max(this.pos, e.t)
    this.emit(e)
  }

  setSpeed(speed: number) {
    const playing = !this.paused
    this.pause()
    this.speed = speed
    if (playing) this.play()
  }

  cancel() {
    this.pause()
    this.cancelled = true
  }

  private schedule() {
    clearTimeout(this.timer)
    if (this.ended) {
      this.pause()
      return
    }
    const delay = Math.max(0, (this.events[this.i].t - this.position()) / this.speed)
    this.timer = setTimeout(() => this.tick(), delay)
  }

  private tick() {
    const now = this.position()
    while (!this.paused && !this.ended && this.events[this.i].t <= now) this.emit(this.events[this.i++])
    if (!this.paused) this.schedule()
  }
}
