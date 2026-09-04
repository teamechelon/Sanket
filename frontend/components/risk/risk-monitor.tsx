'use client'

import { useEffect, useMemo, useState } from 'react'
import { ArrowUpRight, Bookmark, BookmarkCheck, ChevronRight, Download } from 'lucide-react'
import { DataStatus } from '@/components/layout/data-status'
import { SectionHeading } from '@/components/metrics/section-heading'
import { RiskBadge } from '@/components/risk/risk-badge'
import { EmptyState } from '@/components/ui/data-states'
import {
  costSignalLabel,
  getElevatedProjects,
  getNewlyElevated,
  getProjectById,
  PREVIOUS_SNAPSHOT_DATE,
  projects,
  scheduleSignalLabel,
  SNAPSHOT_DATE,
  type Project,
} from '@/lib/sanket-data'

type SectionKey = 'elevated' | 'review' | 'critical' | 'watchlist' | 'all'

export function RiskMonitor({
  onProject,
  onToggleWatch,
  isWatched,
  watchIds,
  initialSection,
  initialRiskFilter,
}: {
  onProject: (project: Project) => void
  onToggleWatch: (id: string) => void
  isWatched: (id: string) => boolean
  watchIds: string[]
  initialSection?: SectionKey
  initialRiskFilter?: string
}) {
  const [section, setSection] = useState<SectionKey>(initialSection ?? 'all')
  const [riskFilter, setRiskFilter] = useState<string | undefined>(initialRiskFilter)

  useEffect(() => {
    if (initialSection) setSection(initialSection)
  }, [initialSection])

  useEffect(() => {
    setRiskFilter(initialRiskFilter)
  }, [initialRiskFilter])

  const newlyElevated = useMemo(() => getNewlyElevated(), [])
  const elevated = useMemo(() => getElevatedProjects(), [])
  const review = useMemo(
    () => projects.filter((p) => ['High', 'Medium'].includes(p.risk) && p.trend !== 'down'),
    [],
  )
  const critical = useMemo(() => projects.filter((p) => p.risk === 'Critical'), [])
  const watchlist = useMemo(
    () => watchIds.map((id) => getProjectById(id)).filter((p): p is Project => Boolean(p)),
    [watchIds],
  )

  const filtered = useMemo(() => {
    let list: Project[]
    if (section === 'elevated') list = elevated
    else if (section === 'review') list = review
    else if (section === 'critical') list = critical
    else if (section === 'watchlist') list = watchlist
    else list = [...projects].sort((a, b) => b.riskScore - a.riskScore)

    if (riskFilter && section === 'review') {
      list = list.filter((p) => p.risk === riskFilter)
    }
    return list
  }, [section, elevated, review, critical, watchlist, riskFilter])

  const sections: { key: SectionKey; label: string; count: number }[] = [
    { key: 'all', label: 'All', count: projects.length },
    { key: 'elevated', label: 'Newly Elevated', count: elevated.length },
    { key: 'review', label: 'Requires Review', count: review.length },
    { key: 'critical', label: 'Critical', count: critical.length },
    { key: 'watchlist', label: 'My Watchlist', count: watchlist.length },
  ]

  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="eyebrow">EARLY DETECTION / MONTHLY CHANGE</div>
          <h1>Early Warning Monitor</h1>
          <p>
            Operational heart of SANKET — why a project is here, and what moved. <span className="demo-label">PROTOTYPE DATA</span>
          </p>
        </div>
        <button className="outline-button">
          <Download />
          Export risk list
        </button>
      </div>

      <div className="risk-tabs">
        {sections.map((item) => (
          <button
            key={item.key}
            className={section === item.key ? 'active' : ''}
            onClick={() => {
              setSection(item.key)
              if (item.key !== 'review') setRiskFilter(undefined)
            }}
          >
            {item.label}
            <span>{item.count}</span>
          </button>
        ))}
      </div>

      <section className="panel table-panel">
        <div className="table-toolbar">
          <span>
            {filtered.length} projects ranked by risk score
            {riskFilter && section === 'review' ? ` · filtered to ${riskFilter}` : ''}
          </span>
          <span className="table-note">
            <span className="status-pip" />
            Updated {SNAPSHOT_DATE}
          </span>
        </div>
        {!filtered.length ? (
          <EmptyState
            title={section === 'watchlist' ? 'Watchlist empty' : 'No projects in this section'}
            detail={
              section === 'watchlist'
                ? 'Bookmark projects from the portfolio to build your watchlist.'
                : 'No matching risk records for this filter.'
            }
          />
        ) : (
          <div className="table-wrap">
            <table className="data-table risk-table">
              <thead>
                <tr>
                  <th />
                  <th>Project</th>
                  <th>State</th>
                  <th>Risk</th>
                  <th>Risk Change</th>
                  <th>Primary Signal</th>
                  <th>Delay Signal</th>
                  <th>Cost Signal</th>
                  <th>Last Updated</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {filtered.map((p) => (
                  <tr key={p.id} onClick={() => onProject(p)}>
                    <td>
                      <button
                        className="watch-button"
                        onClick={(e) => {
                          e.stopPropagation()
                          onToggleWatch(p.id)
                        }}
                        aria-label={isWatched(p.id) ? 'Remove from watchlist' : 'Add to watchlist'}
                      >
                        {isWatched(p.id) ? <BookmarkCheck /> : <Bookmark />}
                      </button>
                    </td>
                    <td>
                      <div className="project-cell">
                        <strong>{p.name}</strong>
                        <span>{p.code}</span>
                      </div>
                    </td>
                    <td>{p.state}</td>
                    <td>
                      <RiskBadge risk={p.risk} />
                    </td>
                    <td>
                      <span className={p.riskChange > 0 ? 'signal-alert' : p.riskChange < 0 ? 'signal-ok' : 'signal-neutral'}>
                        {p.riskChange > 0 ? '↑' : p.riskChange < 0 ? '↓' : '→'}{' '}
                        {p.riskChange > 0 ? '+' : ''}
                        {p.riskChange} pts
                      </span>
                    </td>
                    <td className="primary-signal">{p.primarySignal}</td>
                    <td>
                      <span className={p.delayMonths > 6 ? 'signal-alert' : 'signal-neutral'}>
                        {scheduleSignalLabel(p)}
                      </span>
                    </td>
                    <td>
                      <span className={p.revisedCost > p.originalCost * 1.1 ? 'signal-alert' : 'signal-ok'}>
                        {costSignalLabel(p)}
                      </span>
                    </td>
                    <td className="date-cell">{p.lastUpdated}</td>
                    <td>
                      <button className="small-action">
                        Review <ChevronRight />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      <section className="panel elevated-panel">
        <SectionHeading
          eyebrow="CHANGE DETECTION"
          title="Newly Elevated Risks"
          action={<span className="table-note">Previous monthly update: {PREVIOUS_SNAPSHOT_DATE}</span>}
        />
        <div className="elevated-grid">
          {newlyElevated.map((item) => {
            const project = getProjectById(item.projectId)
            return (
              <button
                className="elevated-item elevated-clickable"
                key={item.projectId}
                onClick={() => project && onProject(project)}
              >
                <div className="elevated-top">
                  <span className="elevation-arrow">
                    <ArrowUpRight />
                  </span>
                  <strong>{item.change}</strong>
                  <RiskBadge risk={item.to} />
                </div>
                <h3>{item.name}</h3>
                <p>{item.reason}</p>
                <div className="elevated-from">
                  {item.from} <ChevronRight /> {item.to}
                </div>
              </button>
            )
          })}
        </div>
      </section>
      <DataStatus />
    </div>
  )
}
