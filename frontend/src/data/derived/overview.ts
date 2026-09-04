import {
  PERIOD_LABEL,
  RISK_COLORS,
  SNAPSHOT_DATE,
  STATE_CODES,
} from '../constants'
import { costSignal, getProjectRecords, projects, scheduleSignal } from '../projects'
import type {
  OverviewData,
  RiskCount,
  RiskLevel,
  StateRisk,
  TrendPoint,
} from '../types'
import { getChangeEvents } from './changes'
import { getElevatedProjects, getNewlyElevated } from './risk'

export const TOTAL_MONITORED = projects.length

function formatLakhCr(value: number) {
  const lakhs = value / 100000
  if (lakhs >= 1) return `₹${lakhs.toFixed(2)}L Cr`
  return `₹${(value / 1000).toFixed(2)}k Cr`
}

function sparklineFrom(values: number[]) {
  return values.map((v) => Math.round(v * 100) / 100)
}

export function getRiskCounts(): RiskCount[] {
  const counts: Record<RiskLevel, number> = { Low: 0, Medium: 0, High: 0, Critical: 0 }
  for (const p of projects) counts[p.risk] += 1
  return (Object.keys(counts) as RiskLevel[]).map((label) => ({
    label,
    count: counts[label],
    color: RISK_COLORS[label],
  }))
}

export function getStateRisk(): StateRisk[] {
  const byState = new Map<string, typeof projects>()
  for (const p of projects) {
    const list = byState.get(p.state) ?? []
    list.push(p)
    byState.set(p.state, list)
  }

  return [...byState.entries()]
    .map(([state, list]) => {
      const highRisk = list.filter((p) => p.risk === 'High' || p.risk === 'Critical').length
      const riskIndex = Math.round(list.reduce((s, p) => s + p.riskScore, 0) / list.length)
      const changeVsPrev = Math.round(list.reduce((s, p) => s + p.riskChange, 0) / list.length)
      const avgProgress = Math.round(list.reduce((s, p) => s + p.progress, 0) / list.length)
      let level: RiskLevel = 'Low'
      if (list.some((p) => p.risk === 'Critical')) level = 'Critical'
      else if (list.some((p) => p.risk === 'High')) level = 'High'
      else if (list.some((p) => p.risk === 'Medium')) level = 'Medium'
      return {
        state,
        code: STATE_CODES[state] ?? state.slice(0, 2).toUpperCase(),
        level,
        projects: list.length,
        highRisk,
        riskIndex,
        changeVsPrev,
        avgProgress,
      }
    })
    .sort((a, b) => b.riskIndex - a.riskIndex)
}

export function getTrendData(): TrendPoint[] {
  const records = getProjectRecords()
  const monthKeys = records[0]?.snapshots.map((s) => s.month) ?? []
  // National trend uses last 6 months (drop first if 7).
  const keys = monthKeys.length > 6 ? monthKeys.slice(-6) : monthKeys

  return keys.map((month) => {
    let costSum = 0
    let origSum = 0
    let delayCount = 0
    for (const record of records) {
      const snap = record.snapshots.find((s) => s.month === month)
      if (!snap) continue
      costSum += snap.revisedCost
      origSum += snap.originalCost
      if (snap.delayMonths > 2) delayCount += 1
    }
    const escalation = origSum ? Math.round(((costSum / origSum - 1) * 100) * 10) / 10 : 0
    return {
      month: month.replace(/ 2[56]/, '').replace('Oct', 'Oct').slice(0, 3),
      escalation,
      delay: delayCount,
    }
  })
}

export function getOverview(): OverviewData {
  const totalCost = projects.reduce((s, p) => s + p.revisedCost, 0)
  const totalOrig = projects.reduce((s, p) => s + p.originalCost, 0)
  const totalSpend = projects.reduce((s, p) => s + p.expenditure, 0)
  const atRisk = projects.filter((p) => p.risk === 'High' || p.risk === 'Critical').length
  const delaySignals = projects.filter((p) => scheduleSignal(p) !== 'on_plan').length
  const costSignals = projects.filter((p) => costSignal(p) !== 'within').length
  const elevated = getElevatedProjects().length
  const stable = projects.filter((p) => p.trend === 'stable').length
  const improving = projects.filter((p) => p.trend === 'down').length
  const newly = getNewlyElevated().length
  const costPct = totalOrig ? Math.round(((totalCost / totalOrig - 1) * 100) * 10) / 10 : 0
  const spendPct = totalCost ? Math.round((totalSpend / totalCost) * 1000) / 10 : 0

  const trend = getTrendData()
  const delaySpark = trend.map((t) => t.delay)

  return {
    periodLabel: PERIOD_LABEL,
    snapshotDate: SNAPSHOT_DATE,
    totalProjects: TOTAL_MONITORED,
    metrics: [
      {
        id: 'active',
        label: 'Active Projects',
        value: String(TOTAL_MONITORED),
        change: `${getChangeEvents().length} signals`,
        note: 'in monitored portfolio',
        sparkline: sparklineFrom([8, 9, 10, 11, 12, TOTAL_MONITORED]),
      },
      {
        id: 'revised-cost',
        label: 'Revised Project Cost',
        value: formatLakhCr(totalCost),
        change: `+${costPct}%`,
        note: 'from baseline',
        negative: costPct > 0,
        sparkline: sparklineFrom([
          totalOrig * 0.92,
          totalOrig * 0.95,
          totalOrig * 0.98,
          totalCost * 0.94,
          totalCost * 0.97,
          totalCost,
        ].map((v) => v / 100000)),
      },
      {
        id: 'expenditure',
        label: 'Cumulative Expenditure',
        value: formatLakhCr(totalSpend),
        change: `${spendPct}%`,
        note: 'of revised cost',
        sparkline: sparklineFrom([
          totalSpend * 0.7,
          totalSpend * 0.78,
          totalSpend * 0.85,
          totalSpend * 0.9,
          totalSpend * 0.95,
          totalSpend,
        ].map((v) => v / 100000)),
      },
      {
        id: 'at-risk',
        label: 'Projects At Risk',
        value: String(atRisk),
        change: `${newly} newly elevated`,
        note: 'since last update',
        negative: true,
        sparkline: sparklineFrom([
          Math.max(1, atRisk - 3),
          Math.max(1, atRisk - 2),
          Math.max(1, atRisk - 2),
          Math.max(1, atRisk - 1),
          Math.max(1, atRisk - 1),
          atRisk,
        ]),
      },
      {
        id: 'delay',
        label: 'Delay Signals',
        value: String(delaySignals),
        change: `+${Math.max(0, delaySignals - (delaySpark[delaySpark.length - 2] ?? delaySignals))}`,
        note: 'requiring review',
        negative: true,
        sparkline: delaySpark.length ? delaySpark : [delaySignals],
      },
    ],
    riskPulse: {
      elevated,
      stable,
      improving,
      delaySignals,
      costSignals,
      vsPreviousMonth: newly,
    },
  }
}

export const overviewData = getOverview()
export const riskCounts = getRiskCounts()
export const stateRisk = getStateRisk()
export const trendData = getTrendData()
