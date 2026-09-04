import { getLatestAndPrevious, getRecordById, projects } from '../projects'
import type {
  AttentionItem,
  DriverContribution,
  ProjectEvent,
  ProjectIntelligence,
  TrajectoryPoint,
} from '../types'
import { costSignal, scheduleSignal } from '../projects'

function trajectoryFromSnapshots(projectId: string): TrajectoryPoint[] {
  const record = getRecordById(projectId)
  if (!record) return []
  return record.snapshots.map((snap, index) => {
    const planned = Math.min(95, Math.round(snap.physicalProgress + (record.snapshots.length - 1 - index) * 2 + 8))
    const costChange = Math.round((snap.revisedCost / snap.originalCost - 1) * 100)
    return {
      month: snap.month,
      physical: snap.physicalProgress,
      expenditure: Math.round((snap.expenditure / snap.revisedCost) * 100),
      planned,
      revised: snap.physicalProgress,
      riskScore: snap.riskScore,
      costChange,
      scheduleChange: snap.delayMonths,
    }
  })
}

function buildDrivers(projectId: string): DriverContribution[] {
  const project = projects.find((p) => p.id === projectId)
  const pair = getLatestAndPrevious(projectId)
  if (!project || !pair) {
    return [
      { rank: 1, label: 'Schedule deterioration', value: 28, magnitude: 'Primary driver', color: 'var(--risk-high)' },
      { rank: 2, label: 'Cost escalation', value: 22, magnitude: 'Secondary', color: 'var(--accent)' },
      { rank: 3, label: 'Slow progress', value: 16, magnitude: 'Tertiary', color: 'var(--risk-medium)' },
    ]
  }
  const { latest, previous } = pair
  const costPct = Math.round((latest.revisedCost / latest.originalCost - 1) * 100)
  const progressDelta = previous ? latest.physicalProgress - previous.physicalProgress : 0
  const spendShare = Math.round((latest.expenditure / latest.revisedCost) * 100)

  const candidates: DriverContribution[] = [
    {
      rank: 1,
      label: 'Schedule deterioration',
      value: Math.min(40, 12 + latest.delayMonths * 2),
      magnitude: `+${latest.delayMonths} months`,
      color: 'var(--risk-high)',
    },
    {
      rank: 2,
      label: 'Cost escalation',
      value: Math.min(40, Math.max(8, costPct)),
      magnitude: `+${costPct}%`,
      color: costPct > 25 ? 'var(--risk-critical)' : 'var(--accent)',
    },
    {
      rank: 3,
      label: progressDelta <= 1 ? 'Slow progress' : 'Expenditure / progress gap',
      value: Math.min(30, Math.max(10, spendShare - latest.physicalProgress)),
      magnitude:
        progressDelta <= 1
          ? `${progressDelta} pts MoM`
          : `Spend ${spendShare}% · Progress ${latest.physicalProgress}%`,
      color: 'var(--risk-medium)',
    },
  ]

  return candidates
    .sort((a, b) => b.value - a.value)
    .map((d, i) => ({ ...d, rank: i + 1 }))
}

function buildEvents(projectId: string): ProjectEvent[] {
  const record = getRecordById(projectId)
  const pair = getLatestAndPrevious(projectId)
  if (!record || !pair?.previous) {
    return [
      { id: 'e1', month: 'APR 2026', title: 'Monthly risk reassessment', detail: 'Updated risk position.' },
      { id: 'e2', month: 'MAR 2026', title: 'Reporting snapshot ingested', detail: 'PAIMANA flash report signals refreshed.' },
    ]
  }
  const { latest, previous } = pair
  const events: ProjectEvent[] = []
  if (latest.riskLevel !== previous.riskLevel || latest.riskScore - previous.riskScore >= 6) {
    events.push({
      id: 'e1',
      month: 'APR 2026',
      title: latest.riskLevel === 'Critical' ? 'Risk elevated to Critical' : 'Risk elevated',
      detail: `Moved from ${previous.riskLevel} to ${latest.riskLevel} (${previous.riskScore} → ${latest.riskScore}).`,
    })
  }
  if (latest.revisedCompletion !== previous.revisedCompletion || latest.delayMonths > previous.delayMonths) {
    events.push({
      id: 'e2',
      month: 'MAR 2026',
      title: 'Schedule revised',
      detail: `Target completion moved to ${latest.revisedCompletion}.`,
    })
  }
  if (latest.revisedCost > previous.revisedCost) {
    events.push({
      id: 'e3',
      month: 'FEB 2026',
      title: 'Cost revision recorded',
      detail: `Revised cost increased to ₹${latest.revisedCost.toLocaleString('en-IN')} Cr.`,
    })
  }
  if (latest.physicalProgress - previous.physicalProgress < 3) {
    events.push({
      id: 'e4',
      month: 'JAN 2026',
      title: 'Progress slowdown detected',
      detail: 'Monthly physical progress below recovery plan.',
    })
  }
  if (!events.length) {
    events.push({
      id: 'e1',
      month: 'APR 2026',
      title: 'Monthly risk reassessment',
      detail: `Updated risk position for ${record.name}.`,
    })
  }
  return events
}

function buildAttention(projectId: string): AttentionItem[] {
  const project = projects.find((p) => p.id === projectId)
  if (!project) return []
  const items: AttentionItem[] = [
    { id: 'a1', title: 'Review primary signal', detail: project.primarySignal },
  ]
  if (scheduleSignal(project) !== 'on_plan') {
    items.push({
      id: 'a2',
      title: 'Review schedule recovery plan',
      detail: 'Validate contractor recovery milestones before next review.',
    })
  }
  if (costSignal(project) !== 'within') {
    items.push({
      id: 'a3',
      title: 'Investigate cost escalation',
      detail: 'Compare revised quantities against approved estimate.',
    })
  }
  items.push({
    id: 'a4',
    title: 'Monitor next reporting period',
    detail: 'Watch for further deterioration before escalation.',
  })
  return items.slice(0, 3)
}

export function getProjectIntelligence(projectId: string): ProjectIntelligence {
  const project = projects.find((p) => p.id === projectId) ?? projects[0]
  const id = project.id
  const pair = getLatestAndPrevious(id)
  const latest = pair?.latest
  const trajectory = trajectoryFromSnapshots(id)
  const signalHistory = trajectory.map((point) => ({
    month: point.month,
    risk: point.riskScore ?? project.riskScore,
    cost: point.costChange ?? 0,
    schedule: point.scheduleChange ?? 0,
    progress: point.physical,
  }))
  const outlook: ProjectIntelligence['outlook'] =
    project.riskChange < -2 ? 'Improving' : project.riskChange > 2 ? 'Deteriorating' : 'Stable'

  return {
    projectId: id,
    trajectory,
    drivers: buildDrivers(id),
    events: buildEvents(id),
    signals: [
      project.primarySignal,
      `Risk change ${project.riskChange > 0 ? '+' : ''}${project.riskChange} pts`,
    ],
    attention: buildAttention(id),
    signalHistory,
    outlook,
    originalCompletion: latest?.originalCompletion ?? 'Dec 2026',
    currentCompletion: latest?.revisedCompletion ?? 'Aug 2027',
    targetProgress: Math.min(project.progress + 8, 95),
  }
}

export const trajectoryData = trajectoryFromSnapshots(projects[0]?.id ?? 'p1')
export const driverContributions = buildDrivers(projects[0]?.id ?? 'p1')
