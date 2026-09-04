export type {
  RiskLevel,
  TrendDirection,
  CostSignal,
  ScheduleSignal,
  DataStatus,
  Project,
  ProjectSnapshot,
  ProjectTrajectory,
  RiskProject,
  RiskEvent,
  OverviewMetric,
  OverviewData,
  RiskPulseData,
  StateRisk,
  RiskCount,
  ChangeEvent,
  NewlyElevated,
  TrendPoint,
  TrajectoryPoint,
  DriverContribution,
  ProjectEvent,
  AttentionItem,
  ChangeSummary,
  SignalHistoryPoint,
  ProjectIntelligence,
  SectorPerformance,
  Anomaly,
  AnalyticsData,
  ReportType,
  FilterOptions,
  ProjectRecord,
} from './types'

export {
  SNAPSHOT_DATE,
  SNAPSHOT_LABEL,
  PERIOD_LABEL,
  PREVIOUS_SNAPSHOT_DATE,
  CHANGE_PERIOD_LABEL,
  MONTHS,
  landscapeLabeledStates,
  RISK_AXIS_THRESHOLD,
  RISK_COLORS,
  STATE_CODES,
  reportTypes,
} from './constants'

export {
  projects,
  projectRecords,
  ministries,
  sectors,
  states,
  attentionProjects,
  getProjects,
  getProjectById,
  getProjectRecords,
  getRecordById,
  getTrajectory,
  getSnapshots,
  getLatestAndPrevious,
  searchProjects,
  costSignal,
  scheduleSignal,
  costSignalLabel,
  scheduleSignalLabel,
  attentionPriority,
  primarySignalKind,
  getProjectByCode,
  projectRouteSlug,
  rankedByAttention,
  projectFromRecord,
} from './projects'

export {
  TOTAL_MONITORED,
  getOverview,
  getRiskCounts,
  getStateRisk,
  getTrendData,
  overviewData,
  riskCounts,
  stateRisk,
  trendData,
} from './derived/overview'

export {
  getChangeEvents,
  getChangeSummaries,
  changeEvents,
  explainChange,
} from './derived/changes'

export {
  getNewlyElevated,
  getElevatedProjects,
  getRiskEvents,
  getRiskProjects,
  toRiskProject,
  isElevatedProject,
  newlyElevated,
} from './derived/risk'

export {
  getAnalytics,
  getAnomalies,
  getSectorPerformance,
  analyticsData,
  sectorPerformance,
} from './derived/analytics'

export {
  getProjectIntelligence,
  trajectoryData,
  driverContributions,
} from './derived/intelligence'

import type { RiskLevel } from './types'
import { projects } from './projects'

export const riskClass = (risk: RiskLevel) =>
  ({ Low: 'risk-low', Medium: 'risk-medium', High: 'risk-high', Critical: 'risk-critical' })[risk]

export const formatCr = (value: number) => `₹${value.toLocaleString('en-IN')} Cr`

export const selectedProject = projects[0]
