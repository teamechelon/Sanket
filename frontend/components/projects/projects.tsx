'use client'

import { useEffect, useMemo, useState } from 'react'
import { Download, Filter, Search, SlidersHorizontal, X } from 'lucide-react'
import { DataStatus } from '@/components/layout/data-status'
import { ProjectTable } from '@/components/tables/project-table'
import { EmptyState } from '@/components/ui/data-states'
import {
  costSignal,
  ministries,
  projects,
  scheduleSignal,
  sectors,
  states,
  type Project,
} from '@/lib/sanket-data'

export function Projects({
  onProject,
  selectedState,
  onClearState,
  initialRisk,
  initialTrend,
  onToggleWatch,
  isWatched,
}: {
  onProject: (project: Project) => void
  selectedState: string
  onClearState: () => void
  initialRisk?: string
  initialTrend?: 'up' | 'stable' | 'down' | null
  onToggleWatch: (id: string) => void
  isWatched: (id: string) => boolean
}) {
  const [query, setQuery] = useState('')
  const [risk, setRisk] = useState(initialRisk ?? 'All risks')
  const [ministry, setMinistry] = useState('All Ministries')
  const [sector, setSector] = useState('All Sectors')
  const [stateFilter, setStateFilter] = useState(selectedState)
  const [progress, setProgress] = useState('All progress')
  const [cost, setCost] = useState('All cost')
  const [schedule, setSchedule] = useState('All schedule')
  const [trend, setTrend] = useState<'All trends' | 'up' | 'stable' | 'down'>(
    initialTrend ?? 'All trends',
  )

  useEffect(() => {
    setStateFilter(selectedState)
  }, [selectedState])

  useEffect(() => {
    if (initialRisk) setRisk(initialRisk)
  }, [initialRisk])

  useEffect(() => {
    if (initialTrend) setTrend(initialTrend)
  }, [initialTrend])

  const filtered = useMemo(() => {
    const effectiveState = selectedState !== 'All States' ? selectedState : stateFilter
    return projects.filter((p) => {
      const matchesQuery =
        !query ||
        `${p.name} ${p.code} ${p.agency} ${p.ministry} ${p.sector} ${p.state}`
          .toLowerCase()
          .includes(query.toLowerCase())
      const matchesState = effectiveState === 'All States' || p.state === effectiveState
      const matchesRisk = risk === 'All risks' || p.risk === risk
      const matchesMinistry = ministry === 'All Ministries' || p.ministry === ministry
      const matchesSector = sector === 'All Sectors' || p.sector === sector
      const matchesProgress =
        progress === 'All progress' ||
        (progress === '<40%' && p.progress < 40) ||
        (progress === '40–70%' && p.progress >= 40 && p.progress <= 70) ||
        (progress === '>70%' && p.progress > 70)
      const matchesCost =
        cost === 'All cost' ||
        (cost === 'Escalated' && costSignal(p) === 'escalated') ||
        (cost === 'Watch' && costSignal(p) === 'watch') ||
        (cost === 'Within plan' && costSignal(p) === 'within')
      const matchesSchedule =
        schedule === 'All schedule' ||
        (schedule === 'Delayed' && scheduleSignal(p) === 'delayed') ||
        (schedule === 'Watch' && scheduleSignal(p) === 'watch') ||
        (schedule === 'On plan' && scheduleSignal(p) === 'on_plan')
      const matchesTrend = trend === 'All trends' || p.trend === trend
      return (
        matchesQuery &&
        matchesState &&
        matchesRisk &&
        matchesMinistry &&
        matchesSector &&
        matchesProgress &&
        matchesCost &&
        matchesSchedule &&
        matchesTrend
      )
    })
  }, [query, risk, ministry, sector, selectedState, stateFilter, progress, cost, schedule, trend])

  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="eyebrow">PORTFOLIO EXPLORER / {filtered.length} RESULTS</div>
          <h1>Project Portfolio</h1>
          <p>
            Search and inspect monitored infrastructure projects. <span className="demo-label">PROTOTYPE DATA</span>
          </p>
        </div>
        <button className="outline-button">
          <Download />
          Export portfolio
        </button>
      </div>

      <div className="filters">
        <div className="search-field">
          <Search />
          <input
            placeholder="Search name, code or agency"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
          />
          <kbd>/</kbd>
        </div>
        <select
          aria-label="State filter"
          value={selectedState !== 'All States' ? selectedState : stateFilter}
          onChange={(e) => setStateFilter(e.target.value)}
        >
          {states.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select aria-label="Ministry filter" value={ministry} onChange={(e) => setMinistry(e.target.value)}>
          {ministries.map((m) => (
            <option key={m}>{m}</option>
          ))}
        </select>
        <select aria-label="Sector filter" value={sector} onChange={(e) => setSector(e.target.value)}>
          {sectors.map((s) => (
            <option key={s}>{s}</option>
          ))}
        </select>
        <select aria-label="Risk filter" value={risk} onChange={(e) => setRisk(e.target.value)}>
          <option>All risks</option>
          <option>Low</option>
          <option>Medium</option>
          <option>High</option>
          <option>Critical</option>
        </select>
        <select aria-label="Progress filter" value={progress} onChange={(e) => setProgress(e.target.value)}>
          <option>All progress</option>
          <option>&lt;40%</option>
          <option>40–70%</option>
          <option>&gt;70%</option>
        </select>
        <select aria-label="Cost signal filter" value={cost} onChange={(e) => setCost(e.target.value)}>
          <option>All cost</option>
          <option>Escalated</option>
          <option>Watch</option>
          <option>Within plan</option>
        </select>
        <select aria-label="Schedule signal filter" value={schedule} onChange={(e) => setSchedule(e.target.value)}>
          <option>All schedule</option>
          <option>Delayed</option>
          <option>Watch</option>
          <option>On plan</option>
        </select>
        <button className="filter-button">
          <Filter />
          Dense view
        </button>
      </div>

      {selectedState !== 'All States' && (
        <div className="active-filter">
          Filtered to <strong>{selectedState}</strong>
          <button onClick={onClearState} aria-label="Clear state filter">
            <X />
          </button>
        </div>
      )}
      {trend !== 'All trends' && (
        <div className="active-filter">
          Trend: <strong>{trend === 'up' ? 'Elevated' : trend === 'down' ? 'Improving' : 'Stable'}</strong>
          <button onClick={() => setTrend('All trends')} aria-label="Clear trend filter">
            <X />
          </button>
        </div>
      )}

      <section className="panel table-panel portfolio-table">
        <div className="table-toolbar">
          <span>{filtered.length} projects found</span>
          <button className="text-button">
            <SlidersHorizontal />
            Sort: Risk score
          </button>
        </div>
        {filtered.length ? (
          <ProjectTable
            projects={[...filtered].sort((a, b) => b.riskScore - a.riskScore)}
            onProject={onProject}
            showWatch
            onToggleWatch={onToggleWatch}
            isWatched={isWatched}
          />
        ) : (
          <EmptyState title="No matching projects" detail="Clear filters or broaden search criteria." />
        )}
      </section>
      <DataStatus />
    </div>
  )
}
