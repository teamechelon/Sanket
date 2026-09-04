export type RiskLevel = 'Low' | 'Medium' | 'High' | 'Critical'
export type TrendDirection = 'up' | 'stable' | 'down'
export type CostSignal = 'escalated' | 'watch' | 'within'
export type ScheduleSignal = 'delayed' | 'watch' | 'on_plan'
export type DataStatus = 'loading' | 'empty' | 'error' | 'success'

/** Latest-period project view (derived from the newest snapshot). */
export interface Project {
  id: string
  name: string
  code: string
  ministry: string
  agency: string
  sector: string
  state: string
  progress: number
  expenditure: number
  originalCost: number
  revisedCost: number
  delayMonths: number
  risk: RiskLevel
  riskScore: number
  riskChange: number
  delayProbability: number
  costOverrunProbability: number
  driver: string
  primarySignal: string
  trend: TrendDirection
  lastUpdated: string
  watchlisted?: boolean
}

/** One monthly observation for a project. */
export interface ProjectSnapshot {
  projectId: string
  month: string
  physicalProgress: number
  expenditure: number
  originalCost: number
  revisedCost: number
  originalCompletion: string
  revisedCompletion: string
  delayMonths: number
  riskScore: number
  riskLevel: RiskLevel
  riskSignal: string
}

/** Full longitudinal series for one project. */
export interface ProjectTrajectory {
  projectId: string
  points: ProjectSnapshot[]
}

/** Project enriched for risk-monitor views. */
export interface RiskProject extends Project {
  fromLevel?: RiskLevel
  elevated: boolean
  attentionReason: string
}

/** Month-over-month risk transition event. */
export interface RiskEvent {
  id: string
  projectId: string
  projectName: string
  kind: 'elevated' | 'improved' | 'stable'
  fromLevel: RiskLevel
  toLevel: RiskLevel
  scoreDelta: number
  reason: string
  date: string
}

export interface OverviewMetric {
  id: string
  label: string
  value: string
  change: string
  note: string
  negative?: boolean
  sparkline: number[]
}

export interface OverviewData {
  periodLabel: string
  snapshotDate: string
  totalProjects: number
  metrics: OverviewMetric[]
  riskPulse: RiskPulseData
}

export interface RiskPulseData {
  elevated: number
  stable: number
  improving: number
  delaySignals: number
  costSignals: number
  vsPreviousMonth: number
}

export interface StateRisk {
  state: string
  code: string
  level: RiskLevel
  projects: number
  highRisk: number
  riskIndex: number
  changeVsPrev: number
  avgProgress: number
}

export interface RiskCount {
  label: RiskLevel
  count: number
  color: string
}

export interface ChangeEvent {
  id: string
  type: 'elevated' | 'schedule' | 'cost' | 'progress' | 'improvement'
  title: string
  projectId: string
  projectName: string
  magnitude: string
  detail: string
  date: string
}

export interface NewlyElevated {
  projectId: string
  name: string
  change: string
  from: RiskLevel
  to: RiskLevel
  reason: string
}

export interface TrendPoint {
  month: string
  escalation: number
  delay: number
}

export interface TrajectoryPoint {
  month: string
  physical: number
  expenditure: number
  planned: number
  revised: number
  riskScore?: number
  costChange?: number
  scheduleChange?: number
}

export interface DriverContribution {
  rank: number
  label: string
  value: number
  magnitude: string
  color: string
}

export interface ProjectEvent {
  id: string
  month: string
  title: string
  detail: string
}

export interface AttentionItem {
  id: string
  title: string
  detail: string
}

export interface ChangeSummary {
  id: string
  label: string
  count: number
  tone: 'alert' | 'neutral' | 'positive'
  filter: 'elevated' | 'cost' | 'schedule' | 'improving' | 'critical' | 'high'
}

export interface SignalHistoryPoint {
  month: string
  risk: number
  cost: number
  schedule: number
  progress: number
}

export interface ProjectIntelligence {
  projectId: string
  trajectory: TrajectoryPoint[]
  drivers: DriverContribution[]
  events: ProjectEvent[]
  signals: string[]
  attention: AttentionItem[]
  signalHistory: SignalHistoryPoint[]
  originalCompletion: string
  currentCompletion: string
  targetProgress: number
  outlook: 'Improving' | 'Stable' | 'Deteriorating'
}

export interface SectorPerformance {
  name: string
  progress: number
  risk: number
  delay: number
}

export interface Anomaly {
  id: string
  label: string
  count: number
  tone: 'alert' | 'neutral'
  projectIds: string[]
}

export interface AnalyticsData {
  healthComponents: { label: string; value: number }[]
  costBuckets: { label: string; value: number }[]
  delayBuckets: { label: string; value: number }[]
  concentration: { name: string; value: number }[]
  anomalies: Anomaly[]
}

export interface ReportType {
  id: string
  label: string
  description: string
}

export interface FilterOptions {
  ministries: string[]
  sectors: string[]
  states: string[]
}

/** Identity + longitudinal snapshots stored as the mock SSOT row. */
export interface ProjectRecord {
  id: string
  name: string
  code: string
  ministry: string
  agency: string
  sector: string
  state: string
  delayProbability: number
  costOverrunProbability: number
  driver: string
  primarySignal: string
  lastUpdated: string
  snapshots: ProjectSnapshot[]
}
