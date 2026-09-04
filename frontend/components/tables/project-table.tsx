'use client'

import { Bookmark, BookmarkCheck, ChevronRight } from 'lucide-react'
import { RiskBadge } from '@/components/risk/risk-badge'
import { costSignalLabel, scheduleSignalLabel, type Project } from '@/lib/sanket-data'

function TrendCell({ trend }: { trend: Project['trend'] }) {
  if (trend === 'up') return <span className="trend-cell trend-up">↑</span>
  if (trend === 'down') return <span className="trend-cell trend-down">↓</span>
  return <span className="trend-cell trend-stable">→</span>
}

export function ProjectTable({
  projects: rows,
  onProject,
  onToggleWatch,
  isWatched,
  showWatch = false,
}: {
  projects: Project[]
  onProject: (project: Project) => void
  onToggleWatch?: (id: string) => void
  isWatched?: (id: string) => boolean
  showWatch?: boolean
}) {
  if (!rows.length) {
    return (
      <div className="ui-state empty-state">
        <strong>No projects match filters</strong>
        <span>Adjust search or filter criteria to see portfolio records.</span>
      </div>
    )
  }

  return (
    <div className="table-wrap">
      <table className="data-table">
        <thead>
          <tr>
            {showWatch && <th />}
            <th>Project</th>
            <th>Project Code</th>
            <th>Ministry</th>
            <th>Sector</th>
            <th>State</th>
            <th>Progress</th>
            <th>Cost Signal</th>
            <th>Delay Signal</th>
            <th>Risk</th>
            <th>Trend</th>
            <th>Last Updated</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {rows.map((project) => (
            <tr
              key={project.id}
              onClick={() => onProject(project)}
              tabIndex={0}
              onKeyDown={(e) => {
                if (e.key === 'Enter') onProject(project)
              }}
            >
              {showWatch && (
                <td>
                  <button
                    className="watch-button"
                    onClick={(e) => {
                      e.stopPropagation()
                      onToggleWatch?.(project.id)
                    }}
                    aria-label={isWatched?.(project.id) ? 'Remove from watchlist' : 'Add to watchlist'}
                  >
                    {isWatched?.(project.id) ? <BookmarkCheck /> : <Bookmark />}
                  </button>
                </td>
              )}
              <td>
                <div className="project-cell">
                  <strong>{project.name}</strong>
                </div>
              </td>
              <td className="mono-cell">{project.code}</td>
              <td>{project.ministry}</td>
              <td>{project.sector}</td>
              <td>{project.state}</td>
              <td>
                <div className="progress-cell">
                  <div className="mini-progress">
                    <i style={{ width: `${project.progress}%` }} />
                  </div>
                  <span>{project.progress}%</span>
                </div>
              </td>
              <td>
                <span className={project.revisedCost > project.originalCost * 1.1 ? 'signal-alert' : 'signal-ok'}>
                  {costSignalLabel(project)}
                </span>
              </td>
              <td>
                <span className={project.delayMonths > 6 ? 'signal-alert' : 'signal-neutral'}>
                  {scheduleSignalLabel(project)}
                </span>
              </td>
              <td>
                <RiskBadge risk={project.risk} />
              </td>
              <td>
                <TrendCell trend={project.trend} />
              </td>
              <td className="date-cell">{project.lastUpdated}</td>
              <td>
                <ChevronRight className="row-chevron" />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
