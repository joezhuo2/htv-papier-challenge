import type { ReactNode } from 'react'
import type { Verdict } from '@contract/types'
import { GLOSSARY, type Term } from '../glossary'

export type Mark = Verdict | 'provisional' | 'pending'

const paths: Record<Mark, string> = {
  confirmed: 'M2 6.5 L5 9.5 L10 3',
  rejected: 'M3 3 L9 9 M9 3 L3 9',
  inconclusive: 'M2 7 C4 4 5 4 6 6 C7 8 8 8 10 5',
  provisional: '',
  pending: '',
}

export function Glyph({ mark, size = 12 }: { mark: Mark; size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 12 12" aria-hidden="true" style={{ flex: 'none' }}>
      {mark === 'provisional' || mark === 'pending' ? (
        <circle cx="6" cy="6" r="4" fill="none" stroke="currentColor" strokeWidth="1.2" strokeDasharray={mark === 'provisional' ? '1.5 1.5' : undefined} />
      ) : (
        <path d={paths[mark]} fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      )}
    </svg>
  )
}

export function VerdictChip({ verdict, provisional }: { verdict?: Verdict; provisional?: boolean }) {
  const mark: Mark = !verdict ? 'pending' : provisional ? 'provisional' : verdict
  const word = !verdict ? 'pending' : provisional ? `provisionally ${verdict}` : verdict
  const tip = !verdict ? undefined : GLOSSARY[provisional ? 'provisional' : verdict]
  return <span className={`verdict ${mark}`} title={tip}><Glyph mark={mark} />{word}</span>
}

export function Lock({ x = 0, y = 0, size = 12 }: { x?: number; y?: number; size?: number }) {
  return (
    <svg x={x} y={y} width={size} height={size} viewBox="0 0 12 12" aria-hidden="true">
      <rect x="2.5" y="5.5" width="7" height="5.5" rx="1" fill="none" stroke="currentColor" strokeWidth="1.2" />
      <path d="M4 5.5 V4 a2 2 0 0 1 4 0 V5.5" fill="none" stroke="currentColor" strokeWidth="1.2" />
    </svg>
  )
}

export const MockTag = ({ on }: { on: boolean }) => (on ? <span className="tag mock">mock</span> : null)

export const Def = ({ term, children }: { term: Term; children: ReactNode }) => <span className="term" title={GLOSSARY[term]}>{children}</span>
