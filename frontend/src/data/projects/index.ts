import type { Project, ProjectRecord, ProjectSnapshot, ProjectTrajectory, TrendDirection } from '../types'
import { projectRecords } from './records'

function latestSnapshot(record: ProjectRecord): ProjectSnapshot {
  return record.snapshots[record.snapshots.length - 1]
}

function previousSnapshot(record: ProjectRecord): ProjectSnapshot | null {
  return record.snapshots.length > 1 ? record.snapshots[record.snapshots.length - 2] : null
}

export function trendFromChange(riskChange: number): TrendDirection {
  if (riskChange > 2) return 'up'
  if (riskChange < -2) return 'down'
  return 'stable'
}

export function projectFromRecord(record: ProjectRecord): Project {
  const latest = latestSnapshot(record)
  const prev = previousSnapshot(record)
  const riskChange = prev ? latest.riskScore - prev.riskScore : 0
  return {
    id: record.id,
    name: record.name,
    code: record.code,
    ministry: record.ministry,
    agency: record.agency,
    sector: record.sector,
    state: record.state,
    progress: latest.physicalProgress,
    expenditure: latest.expenditure,
    originalCost: latest.originalCost,
    revisedCost: latest.revisedCost,
    delayMonths: latest.delayMonths,
    risk: latest.riskLevel,
    riskScore: latest.riskScore,
    riskChange,
    delayProbability: record.delayProbability,
    costOverrunProbability: record.costOverrunProbability,
    driver: record.driver,
    primarySignal: record.primarySignal || latest.riskSignal,
    trend: trendFromChange(riskChange),
    lastUpdated: record.lastUpdated,
  }
}

export function getProjectRecords(): ProjectRecord[] {
  return projectRecords
}

export function getRecordById(id: string): ProjectRecord | null {
  return projectRecords.find((r) => r.id === id) ?? null
}

export function getTrajectory(projectId: string): ProjectTrajectory | null {
  const record = getRecordById(projectId)
  if (!record) return null
  return { projectId, points: record.snapshots }
}

export function getSnapshots(projectId: string): ProjectSnapshot[] {
  return getRecordById(projectId)?.snapshots ?? []
}

export function getLatestAndPrevious(projectId: string): {
  latest: ProjectSnapshot
  previous: ProjectSnapshot | null
} | null {
  const record = getRecordById(projectId)
  if (!record) return null
  return { latest: latestSnapshot(record), previous: previousSnapshot(record) }
}

/** Flat latest-period project list — shared by all pages. */
export const projects: Project[] = projectRecords.map(projectFromRecord)

export const ministries = [
  'All Ministries',
  'Road Transport & Highways',
  'Railways',
  'Jal Shakti',
  'Housing & Urban Affairs',
]

export const sectors = ['All Sectors', 'Roads', 'Railways', 'Water Resources', 'Urban Transport']

export const states = [
  'All States',
  'Madhya Pradesh',
  'Uttar Pradesh',
  'Telangana',
  'Gujarat',
  'Tamil Nadu',
  'Bihar',
  'Karnataka',
  'Andhra Pradesh',
  'Maharashtra',
  'Odisha',
  'Rajasthan',
]

export const attentionProjects = projects.filter((project) =>
  ['High', 'Critical'].includes(project.risk),
)

export function getProjects() {
  return projects
}

export function getProjectById(id: string) {
  return projects.find((p) => p.id === id) ?? null
}

export function searchProjects(query: string) {
  const q = query.trim().toLowerCase()
  if (!q) return projects
  return projects.filter((p) =>
    `${p.name} ${p.code} ${p.agency} ${p.ministry} ${p.sector} ${p.state}`
      .toLowerCase()
      .includes(q),
  )
}

export function costSignal(project: Project) {
  const pct = project.revisedCost / project.originalCost - 1
  if (pct > 0.15) return 'escalated' as const
  if (pct > 0.05) return 'watch' as const
  return 'within' as const
}

export function scheduleSignal(project: Project) {
  if (project.delayMonths > 6) return 'delayed' as const
  if (project.delayMonths > 2) return 'watch' as const
  return 'on_plan' as const
}

export function costSignalLabel(project: Project) {
  const pct = Math.round((project.revisedCost / project.originalCost - 1) * 100)
  if (pct > 10) return `+${pct}%`
  return 'Within plan'
}

export function scheduleSignalLabel(project: Project) {
  if (!project.delayMonths) return 'On plan'
  return `${project.delayMonths} mo.`
}

/** Demo attention priority — not an official government metric. */
export function attentionPriority(project: Project) {
  const costPct = Math.max(0, project.revisedCost / project.originalCost - 1)
  const costPts = costPct > 0.15 ? 18 : costPct > 0.05 ? 8 : 0
  const schedulePts = Math.min(project.delayMonths * 1.8, 22)
  const changePts = Math.max(0, project.riskChange) * 1.6
  const progressPts = Math.max(0, (55 - project.progress) * 0.25)
  const levelPts = { Low: 0, Medium: 8, High: 18, Critical: 28 }[project.risk]
  return Math.round(
    project.riskScore * 0.35 + changePts + costPts + schedulePts + progressPts + levelPts,
  )
}

export function primarySignalKind(project: Project) {
  const lower = project.primarySignal.toLowerCase()
  if (lower.includes('cost')) return 'Cost'
  if (lower.includes('schedule') || lower.includes('delay')) return 'Schedule'
  if (lower.includes('progress')) return 'Progress'
  if (lower.includes('land') || lower.includes('contractor') || lower.includes('utility')) return 'Delivery'
  return 'Baseline'
}

export function getProjectByCode(code: string) {
  const decoded = decodeURIComponent(code)
  return projects.find((p) => p.code === decoded || p.code.replace(/\//g, '-') === decoded) ?? null
}

export function projectRouteSlug(code: string) {
  return encodeURIComponent(code)
}

export function rankedByAttention(list: Project[] = projects) {
  return [...list].sort((a, b) => attentionPriority(b) - attentionPriority(a))
}

export { projectRecords }
