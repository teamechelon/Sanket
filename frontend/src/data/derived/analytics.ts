import { costSignal, projects, scheduleSignal } from '../projects'
import type { AnalyticsData, Anomaly, SectorPerformance } from '../types'
import { getChangeEvents } from './changes'

export function getSectorPerformance(): SectorPerformance[] {
  const sectors = [...new Set(projects.map((p) => p.sector))]
  return sectors.map((name) => {
    const list = projects.filter((p) => p.sector === name)
    return {
      name,
      progress: Math.round(list.reduce((s, p) => s + p.progress, 0) / list.length),
      risk: Math.round(list.reduce((s, p) => s + p.riskScore, 0) / list.length),
      delay: Math.round((list.reduce((s, p) => s + p.delayMonths, 0) / list.length) * 10) / 10,
    }
  })
}

function costBucketLabel(pct: number) {
  if (pct < 5) return '0–5%'
  if (pct < 10) return '5–10%'
  if (pct < 20) return '10–20%'
  if (pct < 40) return '20–40%'
  return '40%+'
}

function delayBucketLabel(months: number) {
  if (months <= 0) return 'On schedule'
  if (months < 3) return '< 3 months'
  if (months < 6) return '3–6 months'
  if (months < 12) return '6–12 months'
  return '12+ months'
}

export function getAnomalies(): Anomaly[] {
  const highSpendLowProgress = projects.filter(
    (p) => p.expenditure / p.revisedCost > 0.55 && p.progress < 50,
  )
  const rapidCost = projects.filter((p) => p.revisedCost / p.originalCost - 1 > 0.15)
  const scheduleDet = projects.filter((p) => p.delayMonths > 6 && p.riskChange > 0)
  const stagnation = projects.filter((p) => {
    const change = getChangeEvents().find((e) => e.projectId === p.id && e.type === 'progress')
    return Boolean(change)
  })

  return [
    {
      id: 'a1',
      label: 'High expenditure / low progress',
      count: highSpendLowProgress.length,
      tone: 'alert',
      projectIds: highSpendLowProgress.map((p) => p.id),
    },
    {
      id: 'a2',
      label: 'Rapid cost escalation',
      count: rapidCost.length,
      tone: 'alert',
      projectIds: rapidCost.map((p) => p.id),
    },
    {
      id: 'a3',
      label: 'Schedule deterioration',
      count: scheduleDet.length,
      tone: 'alert',
      projectIds: scheduleDet.map((p) => p.id),
    },
    {
      id: 'a4',
      label: 'Progress stagnation',
      count: stagnation.length,
      tone: 'neutral',
      projectIds: stagnation.map((p) => p.id),
    },
  ]
}

export function getAnalytics(): AnalyticsData {
  const scheduleHealth = Math.round(
    (projects.filter((p) => scheduleSignal(p) === 'on_plan').length / projects.length) * 100,
  )
  const costHealth = Math.round(
    (projects.filter((p) => costSignal(p) === 'within').length / projects.length) * 100,
  )
  const progressHealth = Math.round(projects.reduce((s, p) => s + p.progress, 0) / projects.length)
  const spendEff = Math.round(
    projects.reduce((s, p) => s + Math.min(100, (p.progress / Math.max(1, (p.expenditure / p.revisedCost) * 100)) * 100), 0) /
      projects.length,
  )

  const costBucketOrder = ['0–5%', '5–10%', '10–20%', '20–40%', '40%+']
  const costCounts = Object.fromEntries(costBucketOrder.map((l) => [l, 0])) as Record<string, number>
  for (const p of projects) {
    const pct = (p.revisedCost / p.originalCost - 1) * 100
    costCounts[costBucketLabel(pct)] += 1
  }

  const delayOrder = ['On schedule', '< 3 months', '3–6 months', '6–12 months', '12+ months']
  const delayCounts = Object.fromEntries(delayOrder.map((l) => [l, 0])) as Record<string, number>
  for (const p of projects) {
    delayCounts[delayBucketLabel(p.delayMonths)] += 1
  }

  const concentration = getSectorPerformance()
    .map((s) => ({ name: s.name, value: s.risk }))
    .sort((a, b) => b.value - a.value)

  return {
    healthComponents: [
      { label: 'Schedule health', value: scheduleHealth },
      { label: 'Cost health', value: costHealth },
      { label: 'Progress health', value: progressHealth },
      { label: 'Expenditure efficiency', value: Math.min(100, spendEff) },
    ],
    costBuckets: costBucketOrder.map((label) => ({ label, value: costCounts[label] })),
    delayBuckets: delayOrder.map((label) => ({ label, value: delayCounts[label] })),
    concentration,
    anomalies: getAnomalies(),
  }
}

export const sectorPerformance = getSectorPerformance()
export const analyticsData = getAnalytics()
