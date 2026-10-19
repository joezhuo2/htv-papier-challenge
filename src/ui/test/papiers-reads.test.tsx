import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'
import { NoteChip, PapiersPanel } from '../src/components/PapiersPanel'
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
    expect(within(screen.getByTestId('papiers-panel')).getByText('read, write on approval')).toBeTruthy()
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

  it('raw Papiers ids in summary text render as note chips', () => {
    const id = 'cnv_6630997cd59e09a4331ad143fa6088c4'
    const summary = { ...(demoScript.find(e => e.type === 'summary') as Extract<DemoEvent, { type: 'summary' }>).summary, branch: 'main', tried: `Bigger tiles. Context from Papiers note ${id}` }
    const v = fold([start, { t: 3000, type: 'papiers_context', query: 'q', note: { id, title: 'Conversation overview', excerpt: 'Summarize this article Agent: ## Summary The article is a Triton tutorial on tiling.' } }, { t: 4000, type: 'summary', summary }])
    render(<SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} />)
    const feed = screen.getByLabelText('summary feed')
    expect(feed.textContent).not.toContain(id)
    expect(within(feed).getAllByTestId('note-chip')[0].textContent).toBe('Conversation: Triton tutorial on tiling')
  })

  it('shows no matching notes for an empty search, in the panel and the feed', () => {
    const v = fold([start, { t: 3200, type: 'papiers_context', query: 'accumulation numerical error' }])
    render(<><PapiersPanel view={v} /><SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} /></>)
    expect(screen.getByTestId('papiers-read').textContent).toBe('0:03query "accumulation numerical error"No matching note.')
    expect(screen.getByText(/no matching note for "accumulation numerical error"/)).toBeTruthy()
    expect(v.notes).toHaveLength(0)
  })

  it('hypothesis cards show the note they were informed by and jump to it', () => {
    const v = fold(mockLog)
    render(
      <div className="col"><PapiersPanel view={v} /></div>,
    )
    render(<SummaryFeed view={v} onSelectRun={() => {}} onPostMortem={() => {}} onReport={() => {}} onSend={() => {}} onHybrid={() => {}} />)
    const link = screen.getByTestId('informed-by')
    expect(link.textContent).toBe('Informed by Papiers noteAccumulation is performed in FP32 for numerical accuracy')
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
    expect(screen.getByText('No matching note.')).toBeTruthy()
  })

  it('the mock replay cites n1 on H0 with its query', () => {
    const v = fold(demoScript)
    expect(v.reads.map(r => [r.query, r.note?.id])).toEqual([
      ['Triton matrix multiplication tiling warps pipeline stages', 'n1'],
      ['FP16 accumulation throughput tensor cores', 'n2'],
    ])
  })
})

describe('Papiers highlights', () => {
  const t = 1000
  const hl = { id: 'hl_da32a6e4a587', title: 'MIXED PRECISION TRAINING', excerpt: 'FP16 dot products accumulate into FP32.', kind: 'highlight' as const,
    document_title: 'MIXED PRECISION TRAINING', highlight_text: 'FP16 dot products accumulate into FP32.', annotation: 'This is why our FP16 accumulator fails.' }

  it('renders a highlight with its document title, the highlighted text and the annotation labelled your note', () => {
    const view = fold([{ t, type: 'papiers_context', query: 'accumulation precision FP16', note: hl }])
    render(<PapiersPanel view={view} />)
    const body = within(screen.getByTestId('papiers-read')).getByTestId('note-body')
    expect(body.textContent).toContain('your highlight')
    expect(body.textContent).toContain('MIXED PRECISION TRAINING')
    expect(body.textContent).toContain('FP16 dot products accumulate into FP32.')
    expect(within(body).getByTestId('your-note').textContent).toBe('your note: This is why our FP16 accumulator fails.')
  })

  it('drops LaTeX dollars from an annotation and shows highlights, notes and conversations as distinct chips', () => {
    const note = { id: 'nte_1', title: 'Tiling plan', excerpt: 'Keep 128 per thread.', kind: 'note' as const }
    const view = fold([
      { t, type: 'papiers_context', query: 'q', note: { ...hl, annotation: 'Breaks at $K = 4096$, about $$\\sqrt{K}$$.' } },
      { t, type: 'papiers_context', query: 'q', note },
    ])
    render(<PapiersPanel view={view} />)
    expect(screen.getAllByTestId('your-note')[0].textContent).toBe('your note: Breaks at K = 4096, about \\sqrt{K}.')
    render(<>{['hl_da32a6e4a587', 'nte_1', 'cnv_9'].map(id => <NoteChip key={id} view={view} id={id} />)}</>)
    const chips = screen.getAllByTestId('note-chip')
    expect(chips.map(c => [c.dataset.kind, c.textContent])).toEqual([
      ['highlight', 'Your highlight: MIXED PRECISION TRAINING'], ['note', 'Your note: Tiling plan'], ['conversation', 'Conversation'],
    ])
    expect(new Set(chips.map(c => c.className)).size).toBe(3)
  })

  it('links a hypothesis to its Papiers item by the papiers field, so the note needs no raw id', async () => {
    const { papiersNote, untex } = await import('../src/state/derive')
    expect(papiersNote({ id: 'H1', branch: 'main', claim: 'c', prediction: demoScript[3].type === 'hypothesis_proposed' ? demoScript[3].hypothesis.prediction : ({} as never),
      params: hl as never, note: 'Context from the Papiers conversation "Triton tutorial"', papiers: 'cnv_6630997c' })).toBe('cnv_6630997c')
    expect(untex('like \\(\\sqrt{K}\\) or $K$')).toBe('like \\sqrt{K} or K')
  })

  it('keeps an old-shape note without kind valid', () => {
    const view = fold([{ t, type: 'papiers_context', query: 'q', note: { id: 'cnv_6630997c', title: 'Triton tutorial', excerpt: 'Tiles.' } }])
    render(<PapiersPanel view={view} />)
    expect(screen.getByTestId('note-body').textContent).toContain('Triton tutorial')
    expect(screen.queryByTestId('your-note')).toBeNull()
  })
})
