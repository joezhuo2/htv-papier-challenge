import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { act, fireEvent, render, screen, within } from '@testing-library/react'
import { demoScript } from '@contract/demo-script'
import { App } from '../src/App'
import { HelpCard } from '../src/components/HelpCard'
import { fold } from '../src/state/reducer'

const at = (t: number) => act(() => { vi.advanceTimersByTime(t) })

describe('help request', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('appears on the Speed tab with a badge and marks the blocked branch', () => {
    render(<App source={{ mode: 'mock' }} />)
    fireEvent.keyDown(window, { key: '2' })
    at(99000)
    const speedTab = screen.getByTestId('tab-speed')
    expect(within(speedTab).getByLabelText('help request waiting')).toBeTruthy()
    expect(within(screen.getByTestId('tab-precision')).queryByLabelText('help request waiting')).toBeNull()
    expect(screen.queryByTestId('help-hq1')).toBeNull()

    const node = screen.getByTestId('node-fast')
    expect(node.getAttribute('data-status')).toBe('waiting_on_human')
    expect(within(node).getByText('blocked')).toBeTruthy()
    expect(screen.getByTestId('blocked-label').textContent).toBe('waiting for Speed owner')

    fireEvent.click(speedTab)
    const card = screen.getByTestId('help-hq1')
    expect(within(card).getByLabelText(/Keep tuning tiles and warps/)).toHaveProperty('checked', true)
    expect(within(card).getByTestId('countdown').textContent).toBe('43s left')
    expect(within(card).getByTestId('help-status').textContent).toContain('meanwhile prec')
  })

  it('answering emits answer_help with the chosen option and note', () => {
    const view = { ...fold(demoScript.slice(0, demoScript.findIndex(e => e.type === 'help_requested') + 1)), lens: 'speed' as const }
    const onSend = vi.fn()
    render(<HelpCard view={view} hv={view.helps[0]} now={100000} sent={[]} onSend={onSend} />)
    fireEvent.click(screen.getByLabelText('Suggest something else'))
    fireEvent.change(screen.getByLabelText('answer note'), { target: { value: 'Try tanh GELU' } })
    fireEvent.click(screen.getByRole('button', { name: 'Send answer' }))
    expect(onSend).toHaveBeenCalledWith({ type: 'answer_help', help: 'hq1', option: 'other', note: 'Try tanh GELU', by: 'speed' })
  })

  it('after the deadline shows proceeding on assumption', () => {
    const view = { ...fold(demoScript.slice(0, demoScript.findIndex(e => e.type === 'help_requested') + 1)), lens: 'speed' as const }
    render(<HelpCard view={view} hv={view.helps[0]} now={97000 + 46000} sent={[]} onSend={() => {}} />)
    expect(screen.getByTestId('countdown').textContent).toBe('deadline passed')
    expect(screen.getByRole('status').textContent).toBe('Proceeding on assumption: Keep tuning tiles and warps')
  })

  it('closes an unanswered request when the session ends before its deadline', () => {
    const asked = demoScript.findIndex(e => e.type === 'help_requested')
    const end = demoScript.find(e => e.type === 'session_finished')!
    const view = { ...fold([...demoScript.slice(0, asked + 1), { ...end, t: demoScript[asked].t + 1000 }]), lens: 'speed' as const }
    expect(view.helps[0].expired).toBe(true)
    render(<HelpCard view={view} hv={view.helps[0]} now={demoScript[asked].t + 1000} sent={[]} onSend={() => {}} />)
    expect(screen.queryByTestId('countdown')).toBeNull()
    expect(screen.queryByRole('button', { name: 'Send answer' })).toBeNull()
    expect(screen.getByText('closed')).toBeTruthy()
  })

  it('labels summaries written while another branch waits as meanwhile', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(111000)
    const cards = screen.getAllByTestId('summary-card')
    expect(within(cards.at(-1)!).getByTestId('meanwhile').textContent).toBe('meanwhile, fast waits on Speed')
    expect(within(cards.at(-2)!).queryByTestId('meanwhile')).toBeNull()
  })

  it('clears the badge, unblocks the branch and logs a plan change once answered', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(117500)
    expect(within(screen.getByTestId('tab-speed')).queryByLabelText('help request waiting')).toBeNull()
    expect(screen.getByTestId('node-fast').getAttribute('data-status')).toBe('running')
    expect(screen.getByTestId('plan-change').textContent).toContain('approximate GELU')
    expect(within(screen.getByTestId('help-hq1')).getByText('answered')).toBeTruthy()
  })
})

describe('post-mortem', () => {
  beforeEach(() => vi.useFakeTimers())
  afterEach(() => vi.useRealTimers())

  it('pins the H1 post-mortem with the error versus K sweep', () => {
    render(<App source={{ mode: 'mock' }} />)
    at(90000)
    const pm = screen.getByTestId('postmortem-H1')
    expect(within(pm).getByTestId('sweep').querySelectorAll('circle')).toHaveLength(5)
    expect(within(pm).getByText(/acc fp16/)).toBeTruthy()
    expect(within(pm).getByText(/square root of K/)).toBeTruthy()
    const cards = screen.getAllByTestId('summary-card')
    expect(within(cards.at(-1)!).getByRole('button', { name: 'Open post-mortem' })).toBeTruthy()
  })
})

describe('help request after a verdict flip', () => {
  const base = demoScript.slice(0, demoScript.findIndex(e => e.type === 'help_requested') + 1)
  const req = base.at(-1) as Extract<(typeof demoScript)[number], { type: 'help_requested' }>

  it('a reissued request replaces the card in place with the fresh evidence', () => {
    const again = { ...req, t: req.t + 1000, help: { ...req.help, blocks: 'fast2', evidence: ['r2: 0.520 ms, rejected'] } }
    const view = fold([...base, again])
    expect(view.helps.filter(h => h.help.id === req.help.id)).toHaveLength(1)
    expect(view.helps[0].help.blocks).toBe('fast2')
    expect(view.helps[0].help.evidence).toEqual(['r2: 0.520 ms, rejected'])
  })

  it('a withdrawn request shows the reason and no answer form', () => {
    const view = { ...fold([...base, { t: req.t + 1000, type: 'help_cancelled', help: req.help.id, reason: 'fast was rejected and no branch continues it.' }]), lens: 'speed' as const }
    render(<HelpCard view={view} hv={view.helps[0]} now={req.t + 2000} sent={[]} onSend={vi.fn()} />)
    const card = screen.getByTestId(`help-${req.help.id}`)
    expect(within(card).getByText('withdrawn')).toBeTruthy()
    expect(within(card).getByText(/no branch continues it/)).toBeTruthy()
    expect(within(card).queryByText('Send answer')).toBeNull()
  })
})
