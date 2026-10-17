import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import { createReadStream, existsSync, statSync } from 'node:fs'
import { execFile } from 'node:child_process'
import { fileURLToPath } from 'node:url'
import { delimiter, dirname, join, normalize, sep } from 'node:path'

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

// read-only Papiers search through the papiers CLI, so the token stays on this machine and never reaches the browser
const papiers = (): Plugin => ({
  name: 'papiers',
  configureServer(server) {
    server.middlewares.use('/papiers/search', (req, res) => {
      const q = new URL(req.url ?? '', 'http://x').searchParams.get('q')?.trim()
      const reply = (code: number, body: unknown) => {
        res.statusCode = code
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(body))
      }
      if (!q) return reply(400, { error: 'missing q' })
      const bin = process.env.PAPIERS_BIN ?? join(dirname(process.execPath), 'papiers')
      const env = { ...process.env, PATH: `${dirname(process.execPath)}${delimiter}${process.env.PATH}` }
      execFile(bin, ['search', q, '--limit', '8', '--json'], { env, timeout: 30000 }, (err, stdout) => {
        if (err) return reply(502, { error: `papiers search failed: ${err.message.split('\n')[0]}` })
        const results = (JSON.parse(stdout).results ?? []).map((r: { id: string; kind: string; title?: string; snippet?: string }) => ({
          id: r.id, kind: r.kind, title: r.title ?? r.id, excerpt: (r.snippet ?? '').replace(/\s*\[\[src:[^\]]*\]\]/g, '').trim(),
        }))
        reply(200, { results })
      })
    })
  },
})

export default defineConfig({
  plugins: [react(), serveRuns(), papiers()],
  resolve: { alias: { '@contract': contract } },
  server: {
    fs: { allow: ['..'] },
    proxy: { '/ws': { target: 'ws://127.0.0.1:8000', ws: true } },
  },
})
