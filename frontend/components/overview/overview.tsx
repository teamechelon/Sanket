'use client'

import { ChevronRight, Download, SlidersHorizontal } from 'lucide-react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DataStatus } from '@/components/layout/data-status'
import { NationalRiskLandscape } from '@/components/maps/national-risk-landscape'
import { Metric } from '@/components/metrics/metric'
import { SectionHeading } from '@/components/metrics/section-heading'
import { RiskBadge } from '@/components/risk/risk-badge'
import { ProjectTable } from '@/components/tables/project-table'
import {
  attentionProjects,
  getChangeEvents,
  getOverview,
  riskCounts,
  TOTAL_MONITORED,
  trendData,
  type Project,
} from '@/lib/sanket-data'

export function Overview({
  onProject,
  onState,
  onHighlightState,
  onRisk,
  onPulse,
  onChange,
  selectedState,
  onClearState,
}: {
  onProject: (project: Project) => void
  onState: (state: string) => void
  onHighlightState: (state: string) => void
  onRisk: (risk: string) => void
  onPulse: (kind: 'elevated' | 'stable' | 'improving') => void
  onChange: (projectId: string) => void
  selectedState?: string
  onClearState?: () => void
}) {
  const overview = getOverview()
  const pulse = overview.riskPulse
  const changeEvents = getChangeEvents()
  const costTrend = trendData[trendData.length - 1]
  const costTrendStart = trendData[0]
  const delayTrend = trendData[trendData.length - 1]
  const delayTrendPrev = trendData[trendData.length - 2]

  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="eyebrow">EXECUTIVE MONITORING / {overview.periodLabel}</div>
          <h1>National Infrastructure Overview</h1>
          <p>
            PAIMANA monitored projects <span className="demo-label">PROTOTYPE DATA</span>
          </p>
        </div>
        <button className="outline-button">
          <Download />
          Export snapshot
        </button>
      </div>

      <div className="metric-strip">
        {overview.metrics.map((metric) => (
          <Metric
            key={metric.id}
            label={metric.label}
            value={metric.value}
            change={metric.change}
            note={metric.note}
            negative={metric.negative}
            values={metric.sparkline}
          />
        ))}
      </div>

      <div className="content-grid overview-grid">
        <section className="panel map-panel landscape-panel">
          <SectionHeading
            eyebrow="NATIONAL RISK LANDSCAPE"
            title="State-level portfolio risk and movement"
            action={
              <button className="text-button" onClick={() => onRisk('All')}>
                View risk monitor <ChevronRight />
              </button>
            }
          />
          <div className="map-layout landscape-layout">
            <NationalRiskLandscape
              onHighlight={onHighlightState}
              onViewProjects={onState}
              selectedState={selectedState}
              onClear={onClearState}
            />

            <aside className="risk-summary">
              <div className="summary-block">
                <div className="eyebrow">RISK SUMMARY</div>
                <div className="summary-title">
                  Distribution <span>{TOTAL_MONITORED} total</span>
                </div>
                {riskCounts.map((item) => (
                  <button
                    className="risk-row risk-row-btn"
                    key={item.label}
                    onClick={() => onRisk(item.label)}
                  >
                    <span className="risk-row-name">
                      <span className="legend-swatch" style={{ background: item.color }} />
                      {item.label}
                    </span>
                    <strong>{item.count}</strong>
                    <div className="risk-bar">
                      <i style={{ width: `${(item.count / TOTAL_MONITORED) * 100}%`, background: item.color }} />
                    </div>
                    <span className="risk-percent">{Math.round((item.count / TOTAL_MONITORED) * 100)}%</span>
                  </button>
                ))}
              </div>

              <div className="summary-divider" />

              <div className="summary-block">
                <div className="eyebrow">RISK MOVEMENT</div>
                <div className="movement-items">
                  <button className="movement-btn movement-up" onClick={() => onPulse('elevated')}>
                    ↑ <b>{pulse.elevated}</b> elevated
                  </button>
                  <button className="movement-btn movement-stable" onClick={() => onPulse('stable')}>
                    → <b>{pulse.stable}</b> stable
                  </button>
                  <button className="movement-btn movement-down" onClick={() => onPulse('improving')}>
                    ↓ <b>{pulse.improving}</b> improving
                  </button>
                </div>
                <div className="pulse-stats">
                  <span>
                    <b>{pulse.delaySignals}</b> delay signals
                  </span>
                  <span>
                    <b>{pulse.costSignals}</b> cost signals
                  </span>
                </div>
              </div>

              <div className="summary-divider" />

              <div className="summary-block">
                <div className="summary-title">
                  Projects requiring attention <span className="text-danger">{attentionProjects.length} open</span>
                </div>
                {attentionProjects.slice(0, 4).map((project) => (
                  <button className="attention-row" key={project.id} onClick={() => onProject(project)}>
                    <div>
                      <strong>{project.name}</strong>
                      <span>
                        {project.state} · {project.code}
                      </span>
                    </div>
                    <RiskBadge risk={project.risk} />
                    <ChevronRight />
                  </button>
                ))}
              </div>
            </aside>
          </div>
        </section>

        <section className="panel signals-panel">
          <SectionHeading
            eyebrow="NATIONAL TREND"
            title="Cost & Schedule Signals"
            action={
              <button className="filter-button">
                <SlidersHorizontal />
                Last 6 months
              </button>
            }
          />
          <div className="charts-grid">
            <div className="chart-box">
              <div className="chart-title">
                Cost escalation trend <span>Index, baseline = 100</span>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <AreaChart data={trendData}>
                  <defs>
                    <linearGradient id="costFill" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="var(--accent)" stopOpacity={0.18} />
                      <stop offset="100%" stopColor="var(--accent)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} width={28} />
                  <Tooltip
                    contentStyle={{
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      background: 'var(--card)',
                      color: 'var(--foreground)',
                      fontSize: 12,
                    }}
                  />
                  <Area type="monotone" dataKey="escalation" stroke="var(--accent)" fill="url(#costFill)" strokeWidth={2} />
                </AreaChart>
              </ResponsiveContainer>
              <div className="chart-foot">
                <strong>{costTrend?.escalation ?? 0}%</strong>
                <span>
                  {costTrend && costTrendStart
                    ? `${costTrend.escalation - costTrendStart.escalation >= 0 ? '+' : ''}${(
                        Math.round((costTrend.escalation - costTrendStart.escalation) * 10) / 10
                      )} pts since ${costTrendStart.month}`
                    : 'vs prior months'}
                </span>
              </div>
            </div>
            <div className="chart-box">
              <div className="chart-title">
                Schedule delay trend <span>Projects with delay signals</span>
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={trendData}>
                  <CartesianGrid vertical={false} stroke="var(--border)" />
                  <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} width={28} />
                  <Tooltip
                    contentStyle={{
                      border: '1px solid var(--border)',
                      borderRadius: 6,
                      background: 'var(--card)',
                      color: 'var(--foreground)',
                      fontSize: 12,
                    }}
                  />
                  <Bar dataKey="delay" fill="var(--risk-high)" radius={[2, 2, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
              <div className="chart-foot">
                <strong>{delayTrend?.delay ?? 0} projects</strong>
                <span>
                  {delayTrend && delayTrendPrev
                    ? `${delayTrend.delay - delayTrendPrev.delay >= 0 ? '+' : ''}${
                        delayTrend.delay - delayTrendPrev.delay
                      } since ${delayTrendPrev.month}`
                    : 'latest period'}
                </span>
              </div>
            </div>
          </div>

          <section className="what-changed">
            <SectionHeading eyebrow="WHAT CHANGED" title="Since previous snapshot" />
            <div className="what-changed-list">
              {changeEvents.slice(0, 5).map((event) => (
                <button key={event.id} className="change-item" onClick={() => onChange(event.projectId)}>
                  <span className={`change-type type-${event.type}`}>{event.title}</span>
                  <strong>{event.projectName}</strong>
                  <b className={event.type === 'improvement' ? 'change-positive' : 'change-negative'}>{event.magnitude}</b>
                  <ChevronRight />
                </button>
              ))}
            </div>
          </section>
        </section>
      </div>

      <section className="panel table-panel">
        <SectionHeading
          eyebrow="PRIORITY QUEUE"
          title="Projects Requiring Attention"
          action={
            <button className="text-button" onClick={() => onState('All States')}>
              View all projects <ChevronRight />
            </button>
          }
        />
        <ProjectTable projects={attentionProjects} onProject={onProject} />
      </section>
      <DataStatus />
    </div>
  )
}
