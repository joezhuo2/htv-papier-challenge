import { afterEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { PapiersPanel } from '../src/components/PapiersPanel'
import { fold } from '../src/state/reducer'

describe('Papiers panel', () => {
  afterEach(() => vi.unstubAllGlobals())

  it('lists what the agent read and searches the account only on demand', async () => {
    const fetch = vi.fn(async () => new Response(JSON.stringify({ results: [{ id: 'cnv_1', kind: 'conversation', title: 'Conversation overview', excerpt: 'Triton FP16 matmul tutorial' }] })))
    vi.stubGlobal('fetch', fetch)
    render(<PapiersPanel view={fold(demoScript.slice(0, 3))} />)
    const panel = screen.getByTestId('papiers-panel')
    expect(within(panel).getByText('Triton matmul tiling notes')).toBeTruthy()
    expect(within(panel).getByText('read-only')).toBeTruthy()
    expect(fetch).not.toHaveBeenCalled()
    fireEvent.change(screen.getByLabelText('Search your library'), { target: { value: 'FP16 accumulation' } })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Search' })) })
    expect(fetch).toHaveBeenCalledWith('/papiers/search?q=FP16%20accumulation')
    expect(screen.getByText('Triton FP16 matmul tutorial')).toBeTruthy()
  })

  it('shows the server error instead of results', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(JSON.stringify({ error: 'papiers search failed: not authenticated' }), { status: 502 })))
    render(<PapiersPanel view={fold([])} />)
    fireEvent.change(screen.getByLabelText('Search your library'), { target: { value: 'x' } })
    await act(async () => { fireEvent.click(screen.getByRole('button', { name: 'Search' })) })
    expect(screen.getByRole('alert').textContent).toBe('papiers search failed: not authenticated')
  })
})
