'use client'

import { AlertTriangle, Bookmark, BookmarkCheck, CircleHelp, Download, Target } from 'lucide-react'
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { DataStatus } from '@/components/layout/data-status'
import { SectionHeading } from '@/components/metrics/section-heading'
import { RiskBadge } from '@/components/risk/risk-badge'
import { formatCr, getProjectIntelligence, type Project } from '@/lib/sanket-data'

export function ProjectDetail({
  project,
  onBack,
  returnLabel = 'Projects',
  onToggleWatch,
  isWatched,
}: {
  project: Project
  onBack: () => void
  returnLabel?: string
  onToggleWatch: (id: string) => void
  isWatched: (id: string) => boolean
}) {
  const intel = getProjectIntelligence(project.id)
  const costPct = Math.round((project.revisedCost / project.originalCost - 1) * 100)
  const spendPct = Math.round((project.expenditure / project.revisedCost) * 100)
  const watched = isWatched(project.id)

  return (
    <div className="page-content detail-page">
      <button className="back-button" onClick={onBack}>
        ← Back to {returnLabel === 'Risk Monitor' ? 'Risk Monitor' : returnLabel === 'Overview' ? 'Overview' : 'portfolio'}
      </button>

      <div className="detail-header">
        <div>
          <div className="eyebrow">PROJECT INTELLIGENCE / {project.code}</div>
          <h1>{project.name}</h1>
          <p>
            {project.ministry} · {project.agency} · {project.state}
          </p>
        </div>
        <div className="detail-header-actions">
          <button
            className="outline-button"
            onClick={() => onToggleWatch(project.id)}
            aria-label={watched ? 'Remove from watchlist' : 'Add to watchlist'}
          >
            {watched ? <BookmarkCheck /> : <Bookmark />}
            {watched ? 'On watchlist' : 'Watch'}
          </button>
          <RiskBadge risk={project.risk} />
        </div>
      </div>

      <div className="detail-meta">
        <span>
          <strong>Project code</strong>
          <span className="mono-inline">{project.code}</span>
        </span>
        <span>
          <strong>Current risk score</strong>
          <b className="score-value">
            {project.riskScore}
            <small>/100</small>
          </b>
        </span>
        <span>
          <strong>Risk change</strong>
          <b className={project.riskChange > 0 ? 'signal-alert' : 'signal-ok'}>
            {project.riskChange > 0 ? '+' : ''}
            {project.riskChange} pts
          </b>
        </span>
        <span>
          <strong>Last updated</strong>
          {project.lastUpdated}
        </span>
        <button className="outline-button">
          <Download />
          Project brief
        </button>
      </div>

      <section className="panel status-panel">
        <SectionHeading eyebrow="BASELINE VS CURRENT" title="Project Status" />
        <div className="status-grid">
          {[
            ['Physical progress', `${project.progress}%`, `Target ${intel.targetProgress}%`],
            ['Expenditure', formatCr(project.expenditure), `${spendPct}% of revised`],
            ['Cost position', `+${costPct}%`, 'vs original sanction'],
            ['Schedule position', `+${project.delayMonths} mo.`, `${intel.originalCompletion} → ${intel.currentCompletion}`],
            ['Original cost', formatCr(project.originalCost), 'Approved baseline'],
            ['Revised cost', formatCr(project.revisedCost), costPct ? `+${costPct}% revision` : 'Unchanged'],
          ].map(([label, value, note]) => (
            <div className="status-item" key={label}>
              <span>{label}</span>
              <strong>{value}</strong>
              <small>{note}</small>
            </div>
          ))}
        </div>
      </section>

      <section className="panel trajectory-panel">
        <SectionHeading
          eyebrow="HOW DID THIS PROJECT GET HERE?"
          title="Project trajectory"
          action={
            <div className="chart-legend">
              <span>
                <i className="legend-line physical" />
                Physical progress
              </span>
              <span>
                <i className="legend-line expenditure" />
                Expenditure
              </span>
              <span>
                <i className="legend-line planned" />
                Plan
              </span>
            </div>
          }
        />
        <ResponsiveContainer width="100%" height={250}>
          <LineChart data={intel.trajectory}>
            <CartesianGrid vertical={false} stroke="var(--border)" />
            <XAxis dataKey="month" axisLine={false} tickLine={false} tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }} />
            <YAxis
              domain={[0, 100]}
              axisLine={false}
              tickLine={false}
              tick={{ fontSize: 11, fill: 'var(--muted-foreground)' }}
              tickFormatter={(v) => `${v}%`}
              width={42}
            />
            <Tooltip
              contentStyle={{
                border: '1px solid var(--border)',
                borderRadius: 6,
                background: 'var(--card)',
                color: 'var(--foreground)',
                fontSize: 12,
              }}
            />
            <Line type="monotone" dataKey="physical" stroke="var(--accent)" strokeWidth={2.5} dot={{ r: 3, fill: 'var(--accent)' }} />
            <Line type="monotone" dataKey="expenditure" stroke="var(--risk-high)" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="planned" stroke="var(--muted-foreground)" strokeDasharray="5 4" strokeWidth={1.5} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>

      <div className="detail-columns">
        <section className="panel risk-intelligence">
          <SectionHeading eyebrow="WHY THIS PROJECT IS FLAGGED" title="Ranked drivers" />
          <div className="score-row">
            <div className="risk-score-ring">
              <strong>{project.riskScore}</strong>
              <span>risk score</span>
            </div>
            <div>
              <div className="score-context">SANKET SIGNAL</div>
              <p>
                Risk changed <strong>{project.riskChange > 0 ? '+' : ''}{project.riskChange} points</strong> since the previous monthly update.
              </p>
            </div>
          </div>
          <div className="contributions ranked-drivers">
            {intel.drivers.map((item) => (
              <div className="contribution" key={item.label}>
                <div className="contribution-label">
                  <span>
                    <em className="driver-rank">{String(item.rank).padStart(2, '0')}</em> {item.label}
                  </span>
                  <strong>{item.magnitude}</strong>
                </div>
                <div className="contribution-bar">
                  <i style={{ width: `${(item.value / 40) * 100}%`, background: item.color }} />
                </div>
              </div>
            ))}
          </div>
          <div className="signal-list">
            <div className="eyebrow">SANKET SIGNAL</div>
            {intel.signals.map((signal) => (
              <div key={signal} className="signal-chip">
                {signal}
              </div>
            ))}
          </div>
        </section>

        <section className="panel action-panel">
          <SectionHeading eyebrow="RECOMMENDED ATTENTION" title="Review prompts" />
          {intel.attention.map((item, index) => (
            <div className="recommendation" key={item.id}>
              {index === 0 ? <Target /> : index === 1 ? <AlertTriangle /> : <CircleHelp />}
              <div>
                <strong>{item.title}</strong>
                <p>{item.detail}</p>
              </div>
            </div>
          ))}
          <div className="signal-note">
            <span>SANKET SIGNAL</span> indicates a detected pattern from prototype data.{' '}
            <span>RECOMMENDED ATTENTION</span> is a review prompt — not an official government decision.
          </div>
        </section>
      </div>

      <section className="panel event-timeline-panel">
        <SectionHeading eyebrow="CASE FILE" title="Project event timeline" />
        <div className="event-timeline">
          {intel.events.map((event) => (
            <div className="event-item" key={event.id}>
              <div className="event-month">{event.month}</div>
              <div className="event-body">
                <strong>{event.title}</strong>
                <p>{event.detail}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <DataStatus />
    </div>
  )
}
