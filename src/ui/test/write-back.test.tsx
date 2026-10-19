import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { TopBar } from '../src/components/TopBar'
import { WriteModal } from '../src/components/WriteModal'
import { PapiersPanel } from '../src/components/PapiersPanel'
import { fold } from '../src/state/reducer'
import type { Mode } from '../src/source'
import type { DemoEvent } from '@contract/types'
import { demoScript } from '@contract/demo-script'

const bar = (mode: Mode, finished: boolean, extra: Partial<Parameters<typeof TopBar>[0]> = {}) => {
  const fn = { onStep: vi.fn(), onPrev: vi.fn(), onWrite: vi.fn() }
  render(
    <TopBar
      mode={mode} mock={false} judging={false} now={0} paused ended={false} finished={finished} speed={1} chapters={[]} autoPause={false} canPrev
      onPause={() => {}} onResume={() => {}} onSpeed={() => {}} onRestart={() => {}} onMode={() => {}} onChapter={() => {}} onAutoPause={() => {}}
      onStop={() => {}} onSave={() => {}} onNewSession={() => {}} live={{ started: true, finished, stopping: false }} {...fn} {...extra}
    />,
  )
  return fn
}

const PREVIEW = {
  session: 's1', apply: true, reason: null,
  pending: [
    { id: 'H0', verdict: 'confirmed', doc: 'MIXED PRECISION TRAINING', pages: '3', passage: 'FP16 accumulates into FP32', note: '[RWR] run s1/H0 | confirmed | 1.54x vs >= 1.20x | Prediction holds.', color: '#5fb85f', written: false },
    { id: 'H1', verdict: 'rejected', doc: 'Triton tutorial', pages: null, passage: 'Tiles share memory', note: '[RWR] run s1/H1 | rejected | std 1.3e-2 vs < 1.0e-3 | missed', color: '#e05d5d', written: true },
  ],
  skipped: [{ id: 'H2', verdict: 'inconclusive', reason: 'cited a conversation, not a document passage' }],
}

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status })

describe('write to Papiers', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('shows the button only once the run has finished, live or replay, next to the existing controls', () => {
    bar('live', false)
    expect(screen.queryByTestId('write-papiers')).toBeNull()
    expect(screen.getByRole('button', { name: 'save partial replay' })).toBeTruthy()
  })

  it.each(['live', 'replay', 'mock'] as Mode[])('a finished %s run shows Write to Papiers', mode => {
    const fn = bar(mode, true)
    fireEvent.click(screen.getByRole('button', { name: 'Write to Papiers' }))
    expect(fn.onWrite).toHaveBeenCalledOnce()
  })

  it('the replay step buttons are < and > with accessible names and the same handlers', () => {
    const fn = bar('replay', false)
    const prev = screen.getByRole('button', { name: 'Previous' })
    const step = screen.getByRole('button', { name: 'Step' })
    expect([prev.textContent, step.textContent]).toEqual(['<', '>'])
    expect([prev.title, step.title]).toEqual(['Previous (Left arrow)', 'Step (Right arrow)'])
    expect(prev.className).toBe('icon')
    fireEvent.click(prev)
    fireEvent.click(step)
    expect([fn.onPrev.mock.calls.length, fn.onStep.mock.calls.length]).toEqual([1, 1])
    expect(screen.queryByTestId('write-papiers')).toBeNull()
  })

  it('applies only the checked items and says written only for a success', async () => {
    const calls: [string, RequestInit | undefined][] = []
    vi.stubGlobal('fetch', vi.fn(async (url: string, init?: RequestInit) => {
      calls.push([url, init])
      return init?.method === 'POST'
        ? json({ session: 's1', results: [{ id: 'H0', status: 'failed', error: 'error: anchor text not found' }, { id: 'H1', status: 'already_written' }] })
        : json(PREVIEW)
    }))
    render(<WriteModal mode="live" onClose={() => {}} />)
    await act(async () => {})
    expect(calls[0][0]).toBe('/papiers/writes')
    const items = screen.getAllByTestId('write-item')
    expect(items[0].textContent).toContain('MIXED PRECISION TRAINING, page 3')
    expect(items[1].textContent).toContain('Triton tutorial, webpage')
    expect(within(items[1]).getByText('already written')).toBeTruthy()
    const apply = screen.getByTestId('write-apply') as HTMLButtonElement
    expect(apply.disabled).toBe(true)
    fireEvent.click(screen.getByLabelText('Select all'))
    expect(apply.textContent).toBe('Apply 2 checked')
    fireEvent.click(screen.getByLabelText('Write H1'))
    fireEvent.click(screen.getByLabelText('Write H1'))
    await act(async () => { fireEvent.click(apply) })
    expect(JSON.parse(String(calls[1][1]!.body))).toEqual({ session: 's1', ids: ['H0', 'H1'] })
    const shown = screen.getAllByTestId('write-result').map(x => x.textContent)
    expect(shown).toEqual(['failed: error: anchor text not found', 'already written'])
    expect(shown.some(t => t === 'written')).toBe(false)
    expect((screen.getByTestId('write-skipped') as HTMLDetailsElement).open).toBe(false)
    expect(screen.getByTestId('write-skipped').textContent).toContain('H2 inconclusive: cited a conversation, not a document passage')
  })

  it('a replay opens the preview with Apply disabled and the reason shown', async () => {
    const fetchMock = vi.fn(async () => json({ ...PREVIEW, apply: false, reason: 'A replay is read-only: Apply works only on the live session that produced the run.' }))
    vi.stubGlobal('fetch', fetchMock)
    render(<WriteModal mode="replay" file="/demo/replay.jsonl" onClose={() => {}} />)
    await act(async () => {})
    expect(fetchMock).toHaveBeenCalledWith('/papiers/writes?file=%2Fdemo%2Freplay.jsonl')
    fireEvent.click(screen.getByLabelText('Select all'))
    expect((screen.getByTestId('write-apply') as HTMLButtonElement).disabled).toBe(true)
    expect(screen.getByTestId('write-disabled').textContent).toContain('A replay is read-only')
  })

  it('mock mode never asks the server and cannot apply', () => {
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<WriteModal mode="mock" onClose={() => {}} />)
    expect(fetchMock).not.toHaveBeenCalled()
    expect(screen.getByText('Mock data has no Papiers passages, so there is nothing to preview.')).toBeTruthy()
  })

  it('the Papiers panel counts the prior verdicts read at start, zero included', () => {
    const start = demoScript[0]
    const prior = (count: number, items: object[] = []) => fold([start, { t: 1, type: 'prior_verdicts', count, items } as DemoEvent])
    render(<PapiersPanel view={prior(0)} />)
    expect(screen.getByTestId('prior-count').textContent).toBe('Prior verdicts read: 0')
  })
})
