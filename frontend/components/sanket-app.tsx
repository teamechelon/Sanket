'use client'

import { useEffect, useState } from 'react'
import { Analytics } from '@/components/analytics/analytics'
import { Header } from '@/components/layout/header'
import { DataSourcesPage, SettingsPage } from '@/components/layout/system-pages'
import { SearchOverlay } from '@/components/navigation/search-overlay'
import { Sidebar } from '@/components/navigation/sidebar'
import { Overview } from '@/components/overview/overview'
import { ProjectDetail } from '@/components/project/project-detail'
import { Projects } from '@/components/projects/projects'
import { Reports } from '@/components/reports/reports'
import { RiskMonitor } from '@/components/risk/risk-monitor'
import { useWatchlist } from '@/lib/hooks/use-watchlist'
import { attentionProjects, getProjectById, type Project } from '@/lib/sanket-data'

type AppPage =
  | 'Overview'
  | 'Projects'
  | 'Risk Monitor'
  | 'Analytics'
  | 'Reports'
  | 'Data Sources'
  | 'Settings'
  | 'Project detail'

export default function SanketApp() {
  const [active, setActive] = useState<AppPage>('Overview')
  const [selected, setSelected] = useState<Project | null>(null)
  const [detailReturnTo, setDetailReturnTo] = useState<AppPage>('Projects')
  const [selectedState, setSelectedState] = useState('All States')
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [searchOpen, setSearchOpen] = useState(false)
  const [query, setQuery] = useState('')
  const [projectRiskFilter, setProjectRiskFilter] = useState<string | undefined>()
  const [projectTrendFilter, setProjectTrendFilter] = useState<'up' | 'stable' | 'down' | null>(null)
  const [riskSection, setRiskSection] = useState<'elevated' | 'review' | 'critical' | 'watchlist' | 'all' | undefined>()
  const { ids: watchIds, toggle, isWatched } = useWatchlist()

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') {
        event.preventDefault()
        setSearchOpen(true)
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  const selectProject = (project: Project, from?: AppPage) => {
    setDetailReturnTo(from ?? (active === 'Project detail' ? detailReturnTo : active))
    setSelected(project)
    setActive('Project detail')
  }

  const selectState = (state: string) => {
    setSelectedState(state)
    setSelected(null)
    setProjectRiskFilter(undefined)
    setProjectTrendFilter(null)
    setActive('Projects')
  }

  const highlightState = (state: string) => {
    setSelectedState(state)
  }

  const clearState = () => {
    setSelectedState('All States')
  }

  const selectRisk = (risk: string) => {
    setSelected(null)
    if (risk === 'All') {
      setRiskSection('all')
      setProjectRiskFilter(undefined)
    } else if (risk === 'Critical') {
      setRiskSection('critical')
      setProjectRiskFilter('Critical')
    } else {
      setRiskSection('review')
      setProjectRiskFilter(risk)
    }
    setActive('Risk Monitor')
  }

  const selectPulse = (kind: 'elevated' | 'stable' | 'improving') => {
    setSelected(null)
    if (kind === 'elevated') {
      setRiskSection('elevated')
      setProjectRiskFilter(undefined)
      setActive('Risk Monitor')
      return
    }
    setProjectTrendFilter(kind === 'improving' ? 'down' : 'stable')
    setProjectRiskFilter(undefined)
    setActive('Projects')
  }

  const openChange = (projectId: string) => {
    const project = getProjectById(projectId)
    if (project) selectProject(project, active === 'Project detail' ? detailReturnTo : active)
  }

  const navigate = (item: string) => {
    setSelected(null)
    setActive(item as AppPage)
  }

  let page: React.ReactNode
  if (selected) {
    page = (
      <ProjectDetail
        project={selected}
        onBack={() => {
          setSelected(null)
          setActive(detailReturnTo)
        }}
        returnLabel={detailReturnTo}
        onToggleWatch={toggle}
        isWatched={isWatched}
      />
    )
  } else if (active === 'Overview') {
    page = (
      <Overview
        onProject={(p) => selectProject(p, 'Overview')}
        onState={selectState}
        onHighlightState={highlightState}
        onRisk={selectRisk}
        onPulse={selectPulse}
        onChange={openChange}
        selectedState={selectedState}
        onClearState={clearState}
      />
    )
  } else if (active === 'Projects') {
    page = (
      <Projects
        onProject={(p) => selectProject(p, 'Projects')}
        selectedState={selectedState}
        onClearState={clearState}
        initialRisk={projectRiskFilter}
        initialTrend={projectTrendFilter}
        onToggleWatch={toggle}
        isWatched={isWatched}
      />
    )
  } else if (active === 'Risk Monitor') {
    page = (
      <RiskMonitor
        onProject={(p) => selectProject(p, 'Risk Monitor')}
        onToggleWatch={toggle}
        isWatched={isWatched}
        watchIds={watchIds}
        initialSection={riskSection}
        initialRiskFilter={projectRiskFilter}
      />
    )
  } else if (active === 'Analytics') {
    page = <Analytics onProject={(p) => selectProject(p, 'Analytics')} />
  } else if (active === 'Reports') {
    page = <Reports />
  } else if (active === 'Data Sources') {
    page = <DataSourcesPage />
  } else if (active === 'Settings') {
    page = <SettingsPage />
  } else {
    page = (
      <Overview
        onProject={(p) => selectProject(p, 'Overview')}
        onState={selectState}
        onHighlightState={highlightState}
        onRisk={selectRisk}
        onPulse={selectPulse}
        onChange={openChange}
        selectedState={selectedState}
        onClearState={clearState}
      />
    )
  }

  return (
    <div className="app-shell">
      <Header onMenu={() => setSidebarOpen(true)} onSearch={() => setSearchOpen(true)} />
      <Sidebar
        active={active === 'Project detail' ? detailReturnTo : active}
        setActive={navigate}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        riskBadgeCount={attentionProjects.length}
      />
      <main className="main-area">{page}</main>
      <SearchOverlay
        open={searchOpen}
        query={query}
        setQuery={setQuery}
        onClose={() => {
          setSearchOpen(false)
          setQuery('')
        }}
        onProject={(p) => {
          selectProject(p, active === 'Project detail' ? detailReturnTo : active)
        }}
        onState={selectState}
      />
      {sidebarOpen && (
        <button className="sidebar-overlay" onClick={() => setSidebarOpen(false)} aria-label="Close navigation overlay" />
      )}
    </div>
  )
}
