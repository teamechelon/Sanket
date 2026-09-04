import { CHANGE_PERIOD_LABEL } from '../constants'
import { getLatestAndPrevious, getProjectById, projects } from '../projects'
import type { ChangeEvent, ChangeSummary, RiskLevel } from '../types'
import { getNewlyElevated, getRiskEvents } from './risk'

const LEVEL_RANK: Record<RiskLevel, number> = {
  Low: 0,
  Medium: 1,
  High: 2,
  Critical: 3,
}

/**
 * What Changed — derived from latest vs previous monthly snapshots.
 * Covers: elevated, cost, schedule, progress, improvement.
 */
export function getChangeEvents(): ChangeEvent[] {
  const events: ChangeEvent[] = []
  let seq = 1

  for (const project of projects) {
    const pair = getLatestAndPrevious(project.id)
    if (!pair?.previous) continue
    const { latest, previous } = pair
    const scoreDelta = latest.riskScore - previous.riskScore
    const levelUp = LEVEL_RANK[latest.riskLevel] > LEVEL_RANK[previous.riskLevel]
    const costDelta = latest.revisedCost - previous.revisedCost
    const costPct = Math.round((latest.revisedCost / latest.originalCost - 1) * 100)
    const delayDelta = latest.delayMonths - previous.delayMonths
    const progressDelta = latest.physicalProgress - previous.physicalProgress
    const expectedProgress = 3
    const date = CHANGE_PERIOD_LABEL

    if (levelUp || scoreDelta >= 6) {
      events.push({
        id: `c${seq++}`,
        type: 'elevated',
        title:
          latest.riskLevel === 'Critical'
            ? 'Newly elevated to Critical'
            : `Newly elevated to ${latest.riskLevel}`,
        projectId: project.id,
        projectName: project.name,
        magnitude: `${scoreDelta > 0 ? '+' : ''}${scoreDelta} pts`,
        detail: latest.riskSignal || project.primarySignal,
        date,
      })
    }

    if (costDelta > 50 || (costPct >= 15 && costDelta > 0)) {
      events.push({
        id: `c${seq++}`,
        type: 'cost',
        title: 'Cost revision recorded',
        projectId: project.id,
        projectName: project.name,
        magnitude: `+${costPct}%`,
        detail: `Revised cost moved to ₹${latest.revisedCost.toLocaleString('en-IN')} Cr`,
        date,
      })
    }

    if (delayDelta > 0 || latest.revisedCompletion !== previous.revisedCompletion) {
      events.push({
        id: `c${seq++}`,
        type: 'schedule',
        title: 'Schedule deterioration',
        projectId: project.id,
        projectName: project.name,
        magnitude: delayDelta > 0 ? `+${delayDelta} months` : latest.revisedCompletion,
        detail: `Completion revised to ${latest.revisedCompletion}`,
        date,
      })
    }

    if (progressDelta < expectedProgress) {
      const stagnant = progressDelta <= 0
      events.push({
        id: `c${seq++}`,
        type: 'progress',
        title: stagnant ? 'Progress stagnation' : 'Progress slowdown detected',
        projectId: project.id,
        projectName: project.name,
        magnitude: stagnant ? '0 pts MoM' : `−${expectedProgress - progressDelta} pts vs plan`,
        detail: stagnant
          ? 'No measured physical progress this period'
          : 'Monthly physical progress below plan',
        date,
      })
    }

    if (
      scoreDelta < -2 ||
      LEVEL_RANK[latest.riskLevel] < LEVEL_RANK[previous.riskLevel]
    ) {
      events.push({
        id: `c${seq++}`,
        type: 'improvement',
        title: 'Notable improvement',
        projectId: project.id,
        projectName: project.name,
        magnitude: `${scoreDelta} pts`,
        detail: 'Risk score declining vs previous month',
        date,
      })
    }
  }

  // Prefer one strong signal per project for the feed, ranked by severity.
  const priority: Record<ChangeEvent['type'], number> = {
    elevated: 0,
    cost: 1,
    schedule: 2,
    progress: 3,
    improvement: 4,
  }

  const byProject = new Map<string, ChangeEvent>()
  for (const event of events.sort((a, b) => priority[a.type] - priority[b.type])) {
    if (!byProject.has(event.projectId)) byProject.set(event.projectId, event)
  }

  // Ensure all five categories appear when data supports them.
  const selected = [...byProject.values()]
  const typesPresent = new Set(selected.map((e) => e.type))
  for (const event of events) {
    if (!typesPresent.has(event.type)) {
      selected.push(event)
      typesPresent.add(event.type)
    }
  }

  return selected.sort((a, b) => priority[a.type] - priority[b.type]).slice(0, 12)
}

export function getChangeSummaries(): ChangeSummary[] {
  const all = getChangeEvents()
  const elevated = getNewlyElevated().length
  const cost = all.filter((e) => e.type === 'cost').length
  const schedule = all.filter((e) => e.type === 'schedule').length
  const improving = all.filter((e) => e.type === 'improvement').length
  return [
    { id: 'cs1', label: 'projects newly elevated', count: elevated, tone: 'alert', filter: 'elevated' },
    { id: 'cs2', label: 'cost escalation signals', count: cost, tone: 'alert', filter: 'cost' },
    { id: 'cs3', label: 'schedule deterioration signals', count: schedule, tone: 'alert', filter: 'schedule' },
    { id: 'cs4', label: 'projects improved', count: improving, tone: 'positive', filter: 'improving' },
  ]
}

/** Eager snapshot for barrel consumers that import `changeEvents` as a const. */
export const changeEvents = getChangeEvents()

export function explainChange(projectId: string) {
  return getChangeEvents().filter((e) => e.projectId === projectId)
}

export function riskEventFor(projectId: string) {
  return getRiskEvents().find((e) => e.projectId === projectId) ?? null
}

export function projectName(projectId: string) {
  return getProjectById(projectId)?.name ?? projectId
}
