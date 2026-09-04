'use client'

import { useMemo, useState } from 'react'
import { ChevronRight } from 'lucide-react'
import {
  CartesianGrid,
  Cell,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DataStatus } from '@/components/layout/data-status'
import { SectionHeading } from '@/components/metrics/section-heading'
import { ProjectTable } from '@/components/tables/project-table'
import {
  getChangeEvents,
  getAnalytics,
  getAnomalies,
  PERIOD_LABEL,
  projects,
  sectorPerformance,
  type Project,
} from '@/lib/sanket-data'

export function Analytics({
  onProject,
}: {
  onProject: (project: Project) => void
}) {
  const [view, setView] = useState('Risk Movement')
  const [sector, setSector] = useState<string | null>(null)
  const [anomalyIds, setAnomalyIds] = useState<string[] | null>(null)
  const analytics = getAnalytics()
  const anomalies = getAnomalies()
  const changeEvents = getChangeEvents()

  const scatter = useMemo(() => {
    const base = projects.map((p) => ({
      ...p,
      change: p.riskChange,
      spend: Math.round((p.expenditure / p.revisedCost) * 100),
      riskY: p.riskScore,
      x: Math.round((p.expenditure / p.revisedCost) * 100),
      y: p.progress,
      costEsc: Math.round((p.revisedCost / p.originalCost - 1) * 100),
      delayY: p.delayMonths,
    }))
    return sector ? base.filter((p) => p.sector === sector) : base
  }, [sector])

  const filteredProjects = useMemo(() => {
    if (anomalyIds) return projects.filter((p) => anomalyIds.includes(p.id))
    if (sector) return projects.filter((p) => p.sector === sector)
    return []
  }, [anomalyIds, sector])

  const health = Math.round(projects.reduce((sum, p) => sum + p.progress, 0) / projects.length)
  const sectorsView = sectorPerformance.map((item) => ({
    ...item,
    health: Math.round((item.progress + (100 - item.risk)) / 2),
  }))
  const views = ['Risk Movement', 'Cost × Progress', 'Cost × Schedule', 'What Changed', 'Anomalies']

  const openFromScatter = (point: unknown) => {
    const payload = point as { id?: string; payload?: { id?: string } }
    const id = payload.id ?? payload.payload?.id
    const project = projects.find((p) => p.id === id)
    if (project) onProject(project)
  }

  const selectSector = (name: string) => {
    setSector((current) => (current === name ? null : name))
    setAnomalyIds(null)
  }

  return (
    <div className="page-content analytics-workspace">
      <div className="page-header">
        <div>
          <div className="eyebrow">ANALYTICAL WORKSPACE / PROTOTYPE SIGNALS</div>
          <h1>Analytics Workspace</h1>
          <p>Signature matrices for portfolio investigation. Cross-filter to projects.</p>
        </div>
        <div className="analytics-filter">
          <span>PERIOD</span>
          <strong>{PERIOD_LABEL}</strong>
          <button
            onClick={() => {
              setSector(null)
              setAnomalyIds(null)
            }}
          >
            Reset filters
          </button>
        </div>
      </div>

      <div className="analytics-tabs">
        {views.map((item) => (
          <button key={item} className={view === item ? 'active' : ''} onClick={() => setView(item)}>
            {item}
          </button>
        ))}
      </div>

      {(sector || anomalyIds) && (
        <div className="active-filter">
          {sector && (
            <>
              Sector <strong>{sector}</strong>
            </>
          )}
          {anomalyIds && (
            <>
              Anomaly set <strong>{anomalyIds.length} projects</strong>
            </>
          )}
          <button
            onClick={() => {
              setSector(null)
              setAnomalyIds(null)
            }}
          >
            Clear
          </button>
        </div>
      )}

      {view === 'Risk Movement' && (
        <section className="panel signature-panel">
          <div className="panel-intro">
            <div>
              <div className="eyebrow">RISK MOVEMENT MATRIX</div>
              <h2>Current risk × change in risk</h2>
              <p>Projects moving toward or away from intervention thresholds.</p>
            </div>
            <span className="signal-note">Derived signal · prototype data</span>
          </div>
          <div className="matrix-chart">
            <div className="quadrant q1">HIGH RISK / RISING</div>
            <div className="quadrant q2">HIGH RISK / FALLING</div>
            <div className="quadrant q3">LOW RISK / RISING</div>
            <div className="quadrant q4">LOW RISK / FALLING</div>
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 24, right: 24, bottom: 25, left: 34 }}>
                <CartesianGrid stroke="var(--border)" />
                <ReferenceLine y={60} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
                <ReferenceLine x={0} stroke="var(--muted-foreground)" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="change"
                  domain={[-20, 22]}
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Change in risk since previous month', position: 'insideBottom', offset: -12, fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="riskY"
                  domain={[0, 100]}
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Current risk', angle: -90, position: 'insideLeft', fontSize: 10 }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: '3 3' }}
                  content={({ active, payload }) =>
                    active && payload?.[0] ? (
                      <div className="chart-tooltip">
                        <strong>{payload[0].payload.name}</strong>
                        <span>{payload[0].payload.state}</span>
                        <b>
                          Risk {payload[0].payload.riskScore} · {payload[0].payload.change > 0 ? '+' : ''}
                          {payload[0].payload.change}
                        </b>
                        <small>{payload[0].payload.driver}</small>
                      </div>
                    ) : null
                  }
                />
                <Scatter
                  data={scatter}
                  fill="var(--accent)"
                  onClick={openFromScatter}
                >
                  {scatter.map((entry) => (
                    <Cell key={entry.id} fill={entry.riskChange > 0 ? 'var(--risk-high)' : 'var(--accent)'} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="sector-chips">
            {sectorsView.map((item) => (
              <button key={item.name} className={sector === item.name ? 'active' : ''} onClick={() => selectSector(item.name)}>
                {item.name}
              </button>
            ))}
          </div>
        </section>
      )}

      {view === 'Cost × Progress' && (
        <section className="panel signature-panel">
          <div className="panel-intro">
            <div>
              <div className="eyebrow">COST × PROGRESS</div>
              <h2>Expenditure efficiency</h2>
              <p>High expenditure relative to achieved physical progress.</p>
            </div>
            <span className="signal-note">Not proven inefficiency</span>
          </div>
          <div className="matrix-chart">
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 20, right: 24, bottom: 30, left: 28 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="x"
                  unit="%"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Expenditure / revised cost', position: 'insideBottom', offset: -16, fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="y"
                  unit="%"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Physical progress', angle: -90, position: 'insideLeft', fontSize: 10 }}
                />
                <ReferenceLine x={65} stroke="var(--risk-medium)" strokeDasharray="3 3" />
                <ReferenceLine y={60} stroke="var(--risk-medium)" strokeDasharray="3 3" />
                <Tooltip />
                <Scatter
                  data={scatter}
                  fill="var(--risk-high)"
                  onClick={openFromScatter}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
          <div className="quadrant-summary">
            <span>
              High spend / low progress <b>{scatter.filter((p) => p.spend > 65 && p.progress < 60).length} projects</b>
            </span>
            <span>
              Low spend / high progress <b>{scatter.filter((p) => p.spend <= 65 && p.progress >= 60).length} projects</b>
            </span>
          </div>
        </section>
      )}

      {view === 'Cost × Schedule' && (
        <section className="panel signature-panel">
          <div className="panel-intro">
            <div>
              <div className="eyebrow">COST × SCHEDULE RISK</div>
              <h2>Escalation vs delay</h2>
              <p>Projects combining cost pressure with schedule slippage.</p>
            </div>
          </div>
          <div className="matrix-chart">
            <ResponsiveContainer width="100%" height={360}>
              <ScatterChart margin={{ top: 20, right: 24, bottom: 30, left: 28 }}>
                <CartesianGrid stroke="var(--border)" />
                <XAxis
                  type="number"
                  dataKey="costEsc"
                  unit="%"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Cost escalation %', position: 'insideBottom', offset: -16, fontSize: 10 }}
                />
                <YAxis
                  type="number"
                  dataKey="delayY"
                  tick={{ fontSize: 10 }}
                  label={{ value: 'Schedule delay (months)', angle: -90, position: 'insideLeft', fontSize: 10 }}
                />
                <Tooltip />
                <Scatter
                  data={scatter}
                  fill="var(--risk-medium)"
                  onClick={openFromScatter}
                />
              </ScatterChart>
            </ResponsiveContainer>
          </div>
        </section>
      )}

      {view === 'What Changed' && (
        <section className="panel change-feed">
          <div className="panel-intro">
            <div>
              <div className="eyebrow">WHAT CHANGED THIS MONTH?</div>
              <h2>Operational change feed</h2>
              <p>Material movements from the prototype reporting periods.</p>
            </div>
          </div>
          {changeEvents.map((event) => {
            const project = projects.find((p) => p.id === event.projectId)
            return (
              <button
                className="change-row"
                key={event.id}
                onClick={() => project && onProject(project)}
              >
                <span className="change-type">{event.title}</span>
                <strong>{event.projectName}</strong>
                <span>{event.date}</span>
                <b className={event.type === 'improvement' ? 'change-positive' : 'change-negative'}>{event.magnitude}</b>
                <ChevronRight />
              </button>
            )
          })}
        </section>
      )}

      {view === 'Anomalies' && (
        <>
          <section className="anomaly-grid">
            <div className="anomaly-heading">
              <div className="eyebrow">OUTLIER DETECTION</div>
              <h2>Anomalies detected</h2>
              <span>Unusual combinations requiring analyst review</span>
            </div>
            {anomalies.map((item) => (
              <button
                className="anomaly-item"
                key={item.id}
                onClick={() => {
                  setAnomalyIds(item.projectIds)
                  setSector(null)
                }}
              >
                <span>{item.label}</span>
                <strong className={item.tone === 'alert' ? 'signal-alert' : 'signal-neutral'}>{item.count}</strong>
                <ChevronRight />
              </button>
            ))}
          </section>
          <section className="panel analytics-panel" style={{ marginTop: 16, padding: 18 }}>
            <SectionHeading eyebrow="PORTFOLIO HEALTH" title={`Index ${health}/100`} />
            <div className="health-components" style={{ border: 0, padding: 0 }}>
              {analytics.healthComponents.map((row) => (
                <div className="health-row" key={row.label}>
                  <span>{row.label}</span>
                  <div>
                    <i style={{ width: `${row.value}%` }} />
                  </div>
                  <strong>{row.value}</strong>
                </div>
              ))}
            </div>
          </section>
        </>
      )}

      {filteredProjects.length > 0 && (
        <section className="panel table-panel" style={{ marginTop: 18 }}>
          <SectionHeading eyebrow="FILTERED SET" title="Related projects" />
          <ProjectTable projects={filteredProjects} onProject={onProject} />
        </section>
      )}

      <DataStatus />
    </div>
  )
}
