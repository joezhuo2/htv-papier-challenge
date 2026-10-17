import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import { join, normalize, sep } from 'node:path'

const contract = fileURLToPath(new URL('../contract', import.meta.url))
const runs = fileURLToPath(new URL('../../runs', import.meta.url))

// serves recorded JSONL logs from the repo runs/ folder at /runs/... in dev
const serveRuns = (): Plugin => ({
  name: 'serve-runs',
  configureServer(server) {
    server.middlewares.use('/runs', (req, res) => {
      const path = normalize(join(runs, decodeURIComponent((req.url ?? '').split('?')[0])))
      if (!path.startsWith(runs + sep) || !existsSync(path) || !statSync(path).isFile()) {
        res.statusCode = 404
        return res.end('not found')
      }
      res.setHeader('Content-Type', 'text/plain; charset=utf-8')
      createReadStream(path).pipe(res)
    })
  },
})

export default defineConfig({
  plugins: [react(), serveRuns()],
  resolve: { alias: { '@contract': contract } },
  server: {
    fs: { allow: ['..'] },
    proxy: { '/ws': { target: 'ws://127.0.0.1:8000', ws: true } },
  },
})
