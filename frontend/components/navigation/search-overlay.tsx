'use client'

import { useEffect } from 'react'
import { Search, X } from 'lucide-react'
import { RiskBadge } from '@/components/risk/risk-badge'
import { searchProjects, stateRisk, type Project } from '@/lib/sanket-data'

export function SearchOverlay({
  open,
  query,
  setQuery,
  onClose,
  onProject,
  onState,
}: {
  open: boolean
  query: string
  setQuery: (value: string) => void
  onClose: () => void
  onProject: (p: Project) => void
  onState: (s: string) => void
}) {
  const matches = searchProjects(query).slice(0, 8)

  useEffect(() => {
    if (!open) return
    const handler = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [open, onClose])

  if (!open) return null

  return (
    <div className="search-overlay" role="dialog" aria-label="Global product search">
      <div className="search-dialog">
        <div className="search-dialog-head">
          <Search />
          <input
            autoFocus
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search project, code, agency, ministry, sector or state"
          />
          <kbd>ESC</kbd>
          <button onClick={onClose} aria-label="Close search">
            <X />
          </button>
        </div>
        <div className="search-results">
          <div className="eyebrow">{query ? `${matches.length} MATCHES` : 'QUICK SEARCH'}</div>
          {query &&
            matches.map((p) => (
              <button
                className="search-result"
                key={p.id}
                onClick={() => {
                  onProject(p)
                  onClose()
                }}
              >
                <div>
                  <strong>{p.name}</strong>
                  <span>
                    {p.code} · {p.agency} · {p.state}
                  </span>
                </div>
                <RiskBadge risk={p.risk} />
              </button>
            ))}
          {query && !matches.length && <div className="search-empty">No matching portfolio records.</div>}
          {!query &&
            stateRisk.slice(0, 5).map((s) => (
              <button
                className="search-result"
                key={s.code}
                onClick={() => {
                  onState(s.state)
                  onClose()
                }}
              >
                <div>
                  <strong>{s.state}</strong>
                  <span>
                    {s.projects} projects · {s.level} focus state
                  </span>
                </div>
                <span className="signal-neutral">STATE VIEW</span>
              </button>
            ))}
        </div>
      </div>
    </div>
  )
}
