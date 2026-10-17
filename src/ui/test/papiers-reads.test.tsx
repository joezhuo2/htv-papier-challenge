import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { PapiersPanel } from '../src/components/PapiersPanel'
import { SummaryFeed } from '../src/components/SummaryFeed'
import { fold } from '../src/state/reducer'

const start = demoScript[0]
const mockLog: DemoEvent[] = [
  start,
  { t: 3000, type: 'papiers_context', query: 'Triton matmul tiling', note: { id: 'cnv_1', title: 'Conversation overview', excerpt: 'Accumulation is performed in FP32 for numerical accuracy.' } },
  { t: 3100, type: 'papiers_context', query: 'Triton matmul tiling', note: { id: 'hl_2', title: 'Highlight', excerpt: 'Each program computes one block of C.' } },
  { t: 3200, type: 'papiers_context', query: 'accumulation numerical error precision bound' },
  { t: 5000, type: 'hypothesis_proposed', hypothesis: { ...(demoScript[3] as Extract<DemoEvent, { type: 'hypothesis_proposed' }>).hypothesis, note: 'Context from Papiers note cnv_1' } },
  { t: 6000, type: 'run_started', run: 'r1', branch: 'main', hypothesis: 'H0' },
  demoScript[5], demoScript[6], { ...demoScript[7], t: 9000 },
]

describe('Papiers reads in the event log', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('lists each read with query, id, excerpt and time, highlighted when read', () => {
    render(<PapiersPanel view={fold(mockLog)} />)
    const reads = screen.getAllByTestId('papiers-read')
    expect(reads).toHaveLength(3)
    expect(reads[0].textContent).toContain('0:03')
    expect(reads[0].textContent).toContain('query "Triton matmul tiling"')
    expect(reads[0].textContent).toContain('cnv_1')
    expect(reads[0].textContent).toContain('Accumulation is performed in FP32')
    expect(reads[1].textContent).toContain('hl_2')
    expect(reads.every(r => r.classList.contains('flash'))).toBe(true)
    expect(reads[0].id).toBe('papiers-cnv_1')
    expect(within(screen.getByTestId('papiers-panel')).getByText('read-only')).toBeTruthy()
  })

  it('shows the answer of a conversation with a cleaned title and a copyable full id', async () => {
    const id = 'cnv_fb1bac55818cfbd648a74abec8d21b7e'
    const excerpt = 'Joe Zhuo: Summarize this article Agent: ## Summary The article is a Triton tutorial on implementing a high-performance FP16 matrix-multiplication kernel. Its goal is to show'
    const writeText = vi.fn(async () => {})
    vi.stubGlobal('navigator', { clipboard: { writeText } })
    render(<PapiersPanel view={fold([start, { t: 3000, type: 'papiers_context', query: 'q', note: { id, title: 'Conversation messages', excerpt } }])} />)
    const read = screen.getByTestId('papiers-read')
    expect(read.textContent).not.toContain('Summarize this article')
    expect(read.textContent).not.toContain('##')
    expect(within(read).getByText('Triton tutorial on implementing a high-performance FP16 matrix-multiplication kernel')).toBeTruthy()
    expect(read.textContent).toContain('Its goal is to show')
    await act(async () => { fireEvent.click(within(read).getByTestId('copy-id')) })
    expect(writeText).toHaveBeenCalledWith(id)
    expect(within(read).getByTestId('copy-id').textContent).toBe(`${id} copied`)
  })

  it('shows no matching notes for an empty search, in the panel and the feed', () => {
    const v = fold([start, { t: 3200, type: 'papiers_context', query: 'accumulation numerical error' }])
    render(<><PapiersPanel view={v} /><SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} /></>)
    expect(screen.getByTestId('papiers-read').textContent).toBe('0:03query "accumulation numerical error"No matching notes.')
    expect(screen.getByText(/no matching notes for "accumulation numerical error"/)).toBeTruthy()
    expect(v.notes).toHaveLength(0)
  })

  it('hypothesis cards show the note they were informed by and jump to it', () => {
    const v = fold(mockLog)
    render(
      <div className="col"><PapiersPanel view={v} /></div>,
    )
    render(<SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} />)
    const link = screen.getByTestId('informed-by')
    expect(link.textContent).toBe('Informed by note cnv_1Accumulation is performed in FP32 for numerical accuracy.')
    const target = document.getElementById('papiers-cnv_1')!
    target.classList.remove('flash')
    fireEvent.click(link)
    expect(target.classList.contains('flash')).toBe(true)
  })

  it('an empty library search says no matching notes', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ results: [] }))))
    render(<PapiersPanel view={fold([])} />)
    fireEvent.change(screen.getByLabelText('Search your library'), { target: { value: 'quantization' } })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Search' })) })
    expect(screen.getByText('No matching notes.')).toBeTruthy()
  })

  it('the mock replay cites n1 on H0 with its query', () => {
    const v = fold(demoScript)
    expect(v.reads.map(r => [r.query, r.note?.id])).toEqual([
      ['Triton matrix multiplication tiling warps pipeline stages', 'n1'],
      ['FP16 accumulation throughput tensor cores', 'n2'],
    ])
  })
})
