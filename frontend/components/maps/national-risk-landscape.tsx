'use client'

import { useMemo, useState } from 'react'
import {
  CartesianGrid,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from 'recharts'
import {
  landscapeLabeledStates,
  RISK_AXIS_THRESHOLD,
  stateRisk,
} from '@/lib/data/overview'
import type { RiskLevel, StateRisk } from '@/lib/types'

const RISK_FILL: Record<RiskLevel, string> = {
  Low: '#5f8971',
  Medium: '#ba7c24',
  High: '#c7513e',
  Critical: '#8d2f2e',
}

type LandscapePoint = StateRisk & {
  x: number
  y: number
  z: number
  labeled: boolean
}

function nodeRadius(projects: number, active: boolean, hovered: boolean) {
  const base = 7 + Math.sqrt(projects) * 2.2
  if (active || hovered) return base + 2.5
  return base
}

function LandscapeTooltip({
  active,
  payload,
}: {
  active?: boolean
  payload?: Array<{ payload: LandscapePoint }>
}) {
  if (!active || !payload?.[0]) return null
  const d = payload[0].payload
  return (
    <div className="landscape-tooltip">
      <div className="landscape-tooltip-state">{d.state.toUpperCase()}</div>
      <div className="landscape-tooltip-row">
        <span>{d.projects} active projects</span>
      </div>
      <div className="landscape-tooltip-row">
        <span>{d.highRisk} high-risk</span>
      </div>
      <div className="landscape-tooltip-metrics">
        <div>
          <span>Risk index</span>
          <strong>{d.riskIndex}</strong>
        </div>
        <div>
          <span>Change</span>
          <strong className={d.changeVsPrev > 0 ? 'change-negative' : d.changeVsPrev < 0 ? 'change-positive' : ''}>
            {d.changeVsPrev > 0 ? '+' : ''}
            {d.changeVsPrev} vs previous month
          </strong>
        </div>
        <div>
          <span>Average progress</span>
          <strong>{d.avgProgress}%</strong>
        </div>
      </div>
    </div>
  )
}

export function NationalRiskLandscape({
  onHighlight,
  onViewProjects,
  selectedState,
  onClear,
}: {
  onHighlight: (state: string) => void
  onViewProjects: (state: string) => void
  selectedState?: string
  onClear?: () => void
}) {
  const [hovered, setHovered] = useState<string | null>(null)
  const focus = selectedState && selectedState !== 'All States' ? selectedState : null

  const data = useMemo<LandscapePoint[]>(
    () =>
      stateRisk.map((s) => ({
        ...s,
        x: s.riskIndex,
        y: s.changeVsPrev,
        z: s.projects,
        labeled: (landscapeLabeledStates as readonly string[]).includes(s.state),
      })),
    [],
  )

  const selectedPoint = focus ? data.find((d) => d.state === focus) : null

  return (
    <div className="landscape-area">
      <div className="landscape-head">
        <div>
          <div className="eyebrow">RISK MOVEMENT MATRIX</div>
          <p className="landscape-subtitle">Each node is a state · size = active projects · color = severity</p>
        </div>
        {focus && onClear && (
          <button className="map-national-return" onClick={onClear}>
            ← Return to national view
          </button>
        )}
      </div>

      <div className="landscape-chart">
        <div className="landscape-quadrant q-hi-det">HIGH RISK / DETERIORATING</div>
        <div className="landscape-quadrant q-hi-imp">HIGH RISK / IMPROVING</div>
        <div className="landscape-quadrant q-lo-det">LOW RISK / DETERIORATING</div>
        <div className="landscape-quadrant q-lo-imp">LOW RISK / IMPROVING</div>
        <div className="landscape-attention-label">ATTENTION ZONE</div>

        <ResponsiveContainer width="100%" height={340}>
          <ScatterChart margin={{ top: 28, right: 28, bottom: 36, left: 18 }}>
            <CartesianGrid stroke="var(--border)" strokeDasharray="0" strokeOpacity={0.85} />
            <ReferenceArea
              x1={RISK_AXIS_THRESHOLD}
              x2={100}
              y1={0}
              y2={12}
              fill="var(--risk-high)"
              fillOpacity={0.06}
              ifOverflow="extendDomain"
            />
            <ReferenceLine x={RISK_AXIS_THRESHOLD} stroke="var(--muted-foreground)" strokeDasharray="4 4" strokeOpacity={0.55} />
            <ReferenceLine y={0} stroke="var(--muted-foreground)" strokeDasharray="4 4" strokeOpacity={0.55} />
            <XAxis
              type="number"
              dataKey="x"
              name="Current risk"
              domain={[15, 95]}
              tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
              label={{
                value: 'Current Risk →',
                position: 'insideBottom',
                offset: -18,
                fontSize: 10,
                fill: 'var(--muted-foreground)',
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="Change"
              domain={[-8, 12]}
              tick={{ fontSize: 10, fill: 'var(--muted-foreground)' }}
              axisLine={{ stroke: 'var(--border)' }}
              tickLine={false}
              width={36}
              label={{
                value: '↑ Change vs previous month',
                angle: -90,
                position: 'insideLeft',
                offset: 8,
                fontSize: 10,
                fill: 'var(--muted-foreground)',
              }}
            />
            <ZAxis type="number" dataKey="z" range={[80, 420]} />
            <Tooltip
              cursor={{ strokeDasharray: '3 3', stroke: 'var(--muted-foreground)' }}
              content={<LandscapeTooltip />}
              wrapperStyle={{ outline: 'none', zIndex: 20 }}
            />
            <Scatter
              data={data}
              isAnimationActive
              animationDuration={220}
              animationBegin={40}
              shape={(props) => {
                const { cx, cy, payload } = props as {
                  cx?: number
                  cy?: number
                  payload?: LandscapePoint
                }
                if (cx == null || cy == null || !payload) return <g />
                const isActive = focus === payload.state
                const isHovered = hovered === payload.state
                const dimmed = Boolean(focus && !isActive)
                const r = nodeRadius(payload.projects, isActive, isHovered)
                const showLabel = payload.labeled || isHovered || isActive
                return (
                  <g
                    className={`landscape-node ${isActive ? 'is-active' : ''} ${dimmed ? 'is-dimmed' : ''}`}
                    style={{ cursor: 'pointer' }}
                    onMouseEnter={() => setHovered(payload.state)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={(event) => {
                      event.stopPropagation()
                      onHighlight(payload.state)
                    }}
                  >
                    <circle
                      cx={cx}
                      cy={cy}
                      r={r}
                      fill={RISK_FILL[payload.level]}
                      fillOpacity={dimmed ? 0.35 : 0.92}
                      stroke={isActive ? 'var(--foreground)' : '#fff'}
                      strokeWidth={isActive ? 2 : 1.25}
                      style={{ transition: 'r 180ms ease, fill-opacity 180ms ease' }}
                    />
                    {showLabel && (
                      <text
                        x={cx}
                        y={cy - r - 6}
                        textAnchor="middle"
                        className="landscape-node-label"
                        fill="var(--foreground)"
                        fontSize={10}
                        fontWeight={600}
                      >
                        {payload.state}
                      </text>
                    )}
                  </g>
                )
              }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {selectedPoint && (
        <div className="landscape-selection">
          <div>
            <strong>{selectedPoint.state}</strong>
            <span>
              Risk index {selectedPoint.riskIndex} · {selectedPoint.changeVsPrev > 0 ? '+' : ''}
              {selectedPoint.changeVsPrev} MoM · {selectedPoint.projects} projects
            </span>
          </div>
          <button className="text-button" onClick={() => onViewProjects(selectedPoint.state)}>
            View {selectedPoint.state} projects →
          </button>
        </div>
      )}

      <div className="landscape-legend">
        <span>
          <i style={{ background: RISK_FILL.Low }} /> Low
        </span>
        <span>
          <i style={{ background: RISK_FILL.Medium }} /> Medium
        </span>
        <span>
          <i style={{ background: RISK_FILL.High }} /> High
        </span>
        <span>
          <i style={{ background: RISK_FILL.Critical }} /> Critical
        </span>
        <span className="landscape-legend-note">Node size ∝ active projects · PROTOTYPE DATA</span>
      </div>
    </div>
  )
}
