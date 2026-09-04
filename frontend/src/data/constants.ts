export const SNAPSHOT_DATE = '30 APR 2026'
export const SNAPSHOT_LABEL = 'PAIMANA SNAPSHOT · 30 APR 2026'
export const PERIOD_LABEL = 'APRIL 2026'
export const PREVIOUS_SNAPSHOT_DATE = '31 MAR 2026'
export const CHANGE_PERIOD_LABEL = 'Apr 2026'

/** Chronological month keys used across trajectories and national trends. */
export const MONTHS = ['Oct 25', 'Nov 25', 'Dec 25', 'Jan 26', 'Feb 26', 'Mar 26', 'Apr 26'] as const

export const landscapeLabeledStates = [
  'Maharashtra',
  'Telangana',
  'Tamil Nadu',
  'Karnataka',
  'Gujarat',
] as const

export const RISK_AXIS_THRESHOLD = 55

export const RISK_COLORS = {
  Low: 'var(--risk-low)',
  Medium: 'var(--risk-medium)',
  High: 'var(--risk-high)',
  Critical: 'var(--risk-critical)',
} as const

export const STATE_CODES: Record<string, string> = {
  'Madhya Pradesh': 'MP',
  Telangana: 'TS',
  'Tamil Nadu': 'TN',
  'Uttar Pradesh': 'UP',
  Bihar: 'BR',
  Gujarat: 'GJ',
  Karnataka: 'KA',
  'Andhra Pradesh': 'AP',
  Maharashtra: 'MH',
  Odisha: 'OD',
  Rajasthan: 'RJ',
}

export const reportTypes = [
  { id: 'national', label: 'National Snapshot', description: 'Executive national overview for the selected period.' },
  { id: 'high-risk', label: 'High-Risk Projects', description: 'Projects at High or Critical risk with primary signals.' },
  { id: 'state', label: 'State Performance', description: 'State-level risk index and delivery signals.' },
  { id: 'sector', label: 'Sector Performance', description: 'Sector comparison of progress, cost, and schedule.' },
  { id: 'monthly', label: 'Monthly Change Report', description: 'What changed since the previous snapshot.' },
]
