import { afterEach } from 'vitest'
import { cleanup } from '@testing-library/react'

class RO {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= RO as unknown as typeof ResizeObserver

afterEach(cleanup)
