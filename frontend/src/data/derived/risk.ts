import { CHANGE_PERIOD_LABEL } from '../constants'
import { getLatestAndPrevious, getProjectById, projects } from '../projects'
import type { NewlyElevated, RiskEvent, RiskLevel, RiskProject } from '../types'

const LEVEL_RANK: Record<RiskLevel, number> = {
  Low: 0,
  Medium: 1,
  High: 2,
  Critical: 3,
}

export function isElevatedProject(projectId: string): boolean {
  const pair = getLatestAndPrevious(projectId)
  if (!pair?.previous) return false
  const { latest, previous } = pair
  const levelUp = LEVEL_RANK[latest.riskLevel] > LEVEL_RANK[previous.riskLevel]
  const scoreUp = latest.riskScore - previous.riskScore > 2
  return levelUp || (scoreUp && latest.riskScore >= 65)
}

export function getRiskEvents(): RiskEvent[] {
  return projects
    .map((project) => {
      const pair = getLatestAndPrevious(project.id)
      if (!pair?.previous) return null
      const { latest, previous } = pair
      const scoreDelta = latest.riskScore - previous.riskScore
      const levelUp = LEVEL_RANK[latest.riskLevel] > LEVEL_RANK[previous.riskLevel]
      const levelDown = LEVEL_RANK[latest.riskLevel] < LEVEL_RANK[previous.riskLevel]
      let kind: RiskEvent['kind'] = 'stable'
      if (levelUp || scoreDelta > 2) kind = 'elevated'
      else if (levelDown || scoreDelta < -2) kind = 'improved'

      return {
        id: `re-${project.id}`,
        projectId: project.id,
        projectName: project.name,
        kind,
        fromLevel: previous.riskLevel,
        toLevel: latest.riskLevel,
        scoreDelta,
        reason: latest.riskSignal || project.primarySignal,
        date: CHANGE_PERIOD_LABEL,
      } satisfies RiskEvent
    })
    .filter((e): e is RiskEvent => Boolean(e))
}

export function getNewlyElevated(): NewlyElevated[] {
  return getRiskEvents()
    .filter((e) => e.kind === 'elevated' && (LEVEL_RANK[e.toLevel] >= LEVEL_RANK.High || e.scoreDelta >= 6))
    .map((e) => ({
      projectId: e.projectId,
      name: e.projectName,
      change: `${e.scoreDelta > 0 ? '+' : ''}${e.scoreDelta} pts`,
      from: e.fromLevel,
      to: e.toLevel,
      reason: e.reason,
    }))
}

/** Same predicate used by Risk Monitor elevated tab and cards. */
export function getElevatedProjects() {
  return projects.filter((p) => isElevatedProject(p.id) || (p.trend === 'up' && p.riskChange > 0))
}

export function toRiskProject(projectId: string): RiskProject | null {
  const project = getProjectById(projectId)
  if (!project) return null
  const pair = getLatestAndPrevious(projectId)
  const fromLevel = pair?.previous?.riskLevel
  const elevated = isElevatedProject(projectId)
  return {
    ...project,
    fromLevel,
    elevated,
    attentionReason: project.primarySignal,
  }
}

export function getRiskProjects(): RiskProject[] {
  return projects
    .map((p) => toRiskProject(p.id))
    .filter((p): p is RiskProject => Boolean(p))
    .sort((a, b) => b.riskScore - a.riskScore)
}

/** @deprecated Prefer getNewlyElevated() — kept as live getter alias for barrel. */
export const newlyElevated = getNewlyElevated()
