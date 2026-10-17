import { defineConfig, mergeConfig } from 'vitest/config'
import base from './vite.config'

export default mergeConfig(base, defineConfig({
  server: { fs: { allow: ['../..'] } },
  test: {
    dir: '../..',
    include: ['src/ui/test/**/*.test.{ts,tsx}', 'tests/ui/**/*.test.ts'],
    environment: 'jsdom',
    testTimeout: 30000,
    setupFiles: ['./test/setup.ts'],
  },
}))
