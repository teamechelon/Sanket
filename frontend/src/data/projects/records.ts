import { MONTHS } from '../constants'
import type { ProjectRecord, ProjectSnapshot, RiskLevel } from '../types'

export function riskLevelFromScore(score: number): RiskLevel {
  if (score >= 85) return 'Critical'
  if (score >= 65) return 'High'
  if (score >= 40) return 'Medium'
  return 'Low'
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function round(n: number) {
  return Math.round(n)
}

type SeriesSeed = {
  id: string
  originalCost: number
  start: {
    physical: number
    expenditure: number
    revisedCost: number
    delayMonths: number
    riskScore: number
    revisedCompletion: string
    riskSignal: string
  }
  end: {
    physical: number
    expenditure: number
    revisedCost: number
    delayMonths: number
    riskScore: number
    revisedCompletion: string
    riskSignal: string
  }
  originalCompletion: string
  /** Optional mid-series overrides keyed by month index (0..6). */
  overrides?: Partial<Record<number, Partial<ProjectSnapshot>>>
}

function buildSnapshots(seed: SeriesSeed): ProjectSnapshot[] {
  const last = MONTHS.length - 1
  return MONTHS.map((month, index) => {
    const t = index / last
    const physical = round(lerp(seed.start.physical, seed.end.physical, t))
    const expenditure = round(lerp(seed.start.expenditure, seed.end.expenditure, t))
    const revisedCost = round(lerp(seed.start.revisedCost, seed.end.revisedCost, t))
    const delayMonths = round(lerp(seed.start.delayMonths, seed.end.delayMonths, t))
    const riskScore = round(lerp(seed.start.riskScore, seed.end.riskScore, t))
    const base: ProjectSnapshot = {
      projectId: seed.id,
      month,
      physicalProgress: physical,
      expenditure,
      originalCost: seed.originalCost,
      revisedCost,
      originalCompletion: seed.originalCompletion,
      revisedCompletion: t < 0.5 ? seed.start.revisedCompletion : seed.end.revisedCompletion,
      delayMonths,
      riskScore,
      riskLevel: riskLevelFromScore(riskScore),
      riskSignal: t < 0.85 ? seed.start.riskSignal : seed.end.riskSignal,
    }
    const override = seed.overrides?.[index]
    if (!override) return base
    const merged = { ...base, ...override }
    return {
      ...merged,
      riskLevel: override.riskLevel ?? riskLevelFromScore(merged.riskScore),
    }
  })
}

type RecordSeed = Omit<ProjectRecord, 'snapshots'> & {
  originalCost: number
  originalCompletion: string
  start: SeriesSeed['start']
  end: SeriesSeed['end']
  overrides?: SeriesSeed['overrides']
}

function record(seed: RecordSeed): ProjectRecord {
  const { originalCost, originalCompletion, start, end, overrides, ...identity } = seed
  return {
    ...identity,
    snapshots: buildSnapshots({
      id: seed.id,
      originalCost,
      originalCompletion,
      start,
      end,
      overrides,
    }),
  }
}

/**
 * Longitudinal SSOT — latest snapshot matches the former flat Project fields.
 * Months: Oct 25 → Apr 26.
 */
export const projectRecords: ProjectRecord[] = [
  record({
    id: 'p1',
    name: 'Delhi–Mumbai Expressway, Package 4',
    code: 'MORTH/NH-04/2019',
    ministry: 'Road Transport & Highways',
    agency: 'NHAI',
    sector: 'Roads',
    state: 'Madhya Pradesh',
    delayProbability: 72,
    costOverrunProbability: 61,
    driver: 'Schedule slippage',
    primarySignal: 'Schedule deterioration · +8 months',
    lastUpdated: '18 Apr 2026',
    originalCost: 9100,
    originalCompletion: 'Dec 2026',
    start: {
      physical: 42,
      expenditure: 4800,
      revisedCost: 9100,
      delayMonths: 2,
      riskScore: 48,
      revisedCompletion: 'Dec 2026',
      riskSignal: 'Within baseline',
    },
    end: {
      physical: 68,
      expenditure: 7420,
      revisedCost: 10480,
      delayMonths: 8,
      riskScore: 78,
      revisedCompletion: 'Aug 2027',
      riskSignal: 'Schedule deterioration · +8 months',
    },
    overrides: {
      5: { riskLevel: 'Medium', riskScore: 70, riskSignal: 'Contractor milestone at risk' },
      6: { riskLevel: 'High', riskScore: 78, delayMonths: 8, revisedCompletion: 'Aug 2027' },
    },
  }),
  record({
    id: 'p2',
    name: 'Dedicated Freight Corridor — East',
    code: 'RAIL/DFC/E-07',
    ministry: 'Railways',
    agency: 'DFCCIL',
    sector: 'Railways',
    state: 'Uttar Pradesh',
    delayProbability: 48,
    costOverrunProbability: 34,
    driver: 'Land acquisition',
    primarySignal: 'Land acquisition · residual parcels',
    lastUpdated: '16 Apr 2026',
    originalCost: 15200,
    originalCompletion: 'Mar 2026',
    start: {
      physical: 62,
      expenditure: 9800,
      revisedCost: 15600,
      delayMonths: 6,
      riskScore: 64,
      revisedCompletion: 'Sep 2026',
      riskSignal: 'Land acquisition pressure',
    },
    end: {
      physical: 81,
      expenditure: 12860,
      revisedCost: 16120,
      delayMonths: 4,
      riskScore: 56,
      revisedCompletion: 'Jul 2026',
      riskSignal: 'Land acquisition · residual parcels',
    },
  }),
  record({
    id: 'p3',
    name: 'Kaleshwaram Lift Irrigation — Phase II',
    code: 'JAL/TS/KL-02',
    ministry: 'Jal Shakti',
    agency: 'I&CAD Telangana',
    sector: 'Water Resources',
    state: 'Telangana',
    delayProbability: 88,
    costOverrunProbability: 83,
    driver: 'Cost escalation',
    primarySignal: 'Cost escalation · +30%',
    lastUpdated: '12 Apr 2026',
    originalCost: 6200,
    originalCompletion: 'Jun 2025',
    start: {
      physical: 28,
      expenditure: 2900,
      revisedCost: 6800,
      delayMonths: 8,
      riskScore: 68,
      revisedCompletion: 'Dec 2025',
      riskSignal: 'Cost pressure building',
    },
    end: {
      physical: 49,
      expenditure: 4860,
      revisedCost: 8040,
      delayMonths: 16,
      riskScore: 91,
      revisedCompletion: 'Oct 2026',
      riskSignal: 'Cost escalation · +30%',
    },
    overrides: {
      5: { riskLevel: 'High', riskScore: 73, delayMonths: 10, revisedCompletion: 'Apr 2026' },
      6: { riskLevel: 'Critical', riskScore: 91, delayMonths: 16, revisedCompletion: 'Oct 2026' },
    },
  }),
  record({
    id: 'p4',
    name: 'Mumbai–Ahmedabad High Speed Rail',
    code: 'RAIL/MAHSR/01',
    ministry: 'Railways',
    agency: 'NHSRCL',
    sector: 'Railways',
    state: 'Gujarat',
    delayProbability: 21,
    costOverrunProbability: 18,
    driver: 'Within baseline',
    primarySignal: 'Within baseline · minor delay',
    lastUpdated: '20 Apr 2026',
    originalCost: 108000,
    originalCompletion: 'Dec 2027',
    start: {
      physical: 28,
      expenditure: 6200,
      revisedCost: 108000,
      delayMonths: 5,
      riskScore: 36,
      revisedCompletion: 'Mar 2028',
      riskSignal: 'Minor schedule watch',
    },
    end: {
      physical: 44,
      expenditure: 9180,
      revisedCost: 110000,
      delayMonths: 3,
      riskScore: 28,
      revisedCompletion: 'Dec 2027',
      riskSignal: 'Within baseline · minor delay',
    },
  }),
  record({
    id: 'p5',
    name: 'Chennai Peripheral Ring Road',
    code: 'MORTH/TN/CPRR',
    ministry: 'Road Transport & Highways',
    agency: 'TNRDC',
    sector: 'Roads',
    state: 'Tamil Nadu',
    delayProbability: 69,
    costOverrunProbability: 58,
    driver: 'Slow physical progress',
    primarySignal: 'Progress slowdown · −4 pts',
    lastUpdated: '14 Apr 2026',
    originalCost: 3900,
    originalCompletion: 'Jun 2026',
    start: {
      physical: 22,
      expenditure: 1100,
      revisedCost: 4100,
      delayMonths: 4,
      riskScore: 52,
      revisedCompletion: 'Sep 2026',
      riskSignal: 'Progress lag emerging',
    },
    end: {
      physical: 35,
      expenditure: 2120,
      revisedCost: 4680,
      delayMonths: 11,
      riskScore: 74,
      revisedCompletion: 'May 2027',
      riskSignal: 'Progress slowdown · −4 pts',
    },
    overrides: {
      5: { physicalProgress: 34, riskScore: 63, riskLevel: 'Medium' },
      6: { physicalProgress: 35, riskScore: 74, riskLevel: 'High' },
    },
  }),
  record({
    id: 'p6',
    name: 'Eastern Dedicated Freight Corridor',
    code: 'RAIL/DFC/E-11',
    ministry: 'Railways',
    agency: 'DFCCIL',
    sector: 'Railways',
    state: 'Bihar',
    delayProbability: 44,
    costOverrunProbability: 31,
    driver: 'Contractor capacity',
    primarySignal: 'Contractor capacity · watch',
    lastUpdated: '17 Apr 2026',
    originalCost: 14200,
    originalCompletion: 'Dec 2026',
    start: {
      physical: 52,
      expenditure: 8200,
      revisedCost: 14500,
      delayMonths: 3,
      riskScore: 44,
      revisedCompletion: 'Mar 2027',
      riskSignal: 'Contractor capacity watch',
    },
    end: {
      physical: 73,
      expenditure: 11940,
      revisedCost: 14980,
      delayMonths: 5,
      riskScore: 51,
      revisedCompletion: 'May 2027',
      riskSignal: 'Contractor capacity · watch',
    },
  }),
  record({
    id: 'p7',
    name: 'Jal Jeevan Mission — Rural Grid',
    code: 'JAL/JJM/KA-22',
    ministry: 'Jal Shakti',
    agency: 'RWS Karnataka',
    sector: 'Water Resources',
    state: 'Karnataka',
    delayProbability: 18,
    costOverrunProbability: 16,
    driver: 'Within baseline',
    primarySignal: 'Improving · on recovery path',
    lastUpdated: '19 Apr 2026',
    originalCost: 4100,
    originalCompletion: 'Mar 2026',
    start: {
      physical: 38,
      expenditure: 1600,
      revisedCost: 4300,
      delayMonths: 5,
      riskScore: 42,
      revisedCompletion: 'Aug 2026',
      riskSignal: 'Recovery underway',
    },
    end: {
      physical: 62,
      expenditure: 2840,
      revisedCost: 4240,
      delayMonths: 2,
      riskScore: 24,
      revisedCompletion: 'May 2026',
      riskSignal: 'Improving · on recovery path',
    },
    overrides: {
      5: { riskScore: 35, riskLevel: 'Low', riskSignal: 'Recovery continuing' },
      6: { riskScore: 24, riskLevel: 'Low', riskSignal: 'Improving · on recovery path' },
    },
  }),
  record({
    id: 'p8',
    name: 'Polavaram Irrigation Project',
    code: 'JAL/AP/POL-01',
    ministry: 'Jal Shakti',
    agency: 'WRD Andhra Pradesh',
    sector: 'Water Resources',
    state: 'Andhra Pradesh',
    delayProbability: 85,
    costOverrunProbability: 79,
    driver: 'Cost escalation',
    primarySignal: 'Cost escalation · +34%',
    lastUpdated: '15 Apr 2026',
    originalCost: 16010,
    originalCompletion: 'Dec 2025',
    start: {
      physical: 38,
      expenditure: 7200,
      revisedCost: 18000,
      delayMonths: 14,
      riskScore: 76,
      revisedCompletion: 'Jun 2026',
      riskSignal: 'Cost revision pending',
    },
    end: {
      physical: 58,
      expenditure: 11240,
      revisedCost: 21500,
      delayMonths: 22,
      riskScore: 88,
      revisedCompletion: 'Oct 2027',
      riskSignal: 'Cost escalation · +34%',
    },
    overrides: {
      4: { revisedCost: 19800, riskScore: 82 },
      5: { revisedCost: 20800, riskScore: 85, riskLevel: 'Critical' },
      6: { revisedCost: 21500, riskScore: 88, riskLevel: 'Critical' },
    },
  }),
  record({
    id: 'p9',
    name: 'Nagpur–Mumbai Expressway, Package 12',
    code: 'MORTH/MH-12/2020',
    ministry: 'Road Transport & Highways',
    agency: 'MSRDC',
    sector: 'Roads',
    state: 'Maharashtra',
    delayProbability: 39,
    costOverrunProbability: 28,
    driver: 'Utility shifting',
    primarySignal: 'Utility shifting · residual',
    lastUpdated: '21 Apr 2026',
    originalCost: 6800,
    originalCompletion: 'Sep 2026',
    start: {
      physical: 48,
      expenditure: 3400,
      revisedCost: 6900,
      delayMonths: 2,
      riskScore: 42,
      revisedCompletion: 'Nov 2026',
      riskSignal: 'Utility shifting residual',
    },
    end: {
      physical: 71,
      expenditure: 5240,
      revisedCost: 7120,
      delayMonths: 3,
      riskScore: 48,
      revisedCompletion: 'Dec 2026',
      riskSignal: 'Utility shifting · residual',
    },
  }),
  record({
    id: 'p10',
    name: 'Eastern Peripheral Expressway Spur',
    code: 'MORTH/UP/EPE-03',
    ministry: 'Road Transport & Highways',
    agency: 'NHAI',
    sector: 'Roads',
    state: 'Uttar Pradesh',
    delayProbability: 15,
    costOverrunProbability: 12,
    driver: 'Within baseline',
    primarySignal: 'Within baseline',
    lastUpdated: '22 Apr 2026',
    originalCost: 3600,
    originalCompletion: 'Jun 2026',
    start: {
      physical: 58,
      expenditure: 2100,
      revisedCost: 3650,
      delayMonths: 3,
      riskScore: 32,
      revisedCompletion: 'Aug 2026',
      riskSignal: 'Within baseline',
    },
    end: {
      physical: 82,
      expenditure: 3180,
      revisedCost: 3720,
      delayMonths: 1,
      riskScore: 22,
      revisedCompletion: 'Jun 2026',
      riskSignal: 'Within baseline',
    },
  }),
  record({
    id: 'p11',
    name: 'Paradip Port Connectivity Road',
    code: 'MORTH/OD/PCR-02',
    ministry: 'Road Transport & Highways',
    agency: 'NHAI',
    sector: 'Roads',
    state: 'Odisha',
    delayProbability: 66,
    costOverrunProbability: 55,
    driver: 'Schedule slippage',
    primarySignal: 'Schedule deterioration · +9 months',
    lastUpdated: '13 Apr 2026',
    originalCost: 2400,
    originalCompletion: 'Mar 2026',
    start: {
      physical: 24,
      expenditure: 900,
      revisedCost: 2500,
      delayMonths: 3,
      riskScore: 54,
      revisedCompletion: 'Jun 2026',
      riskSignal: 'Schedule pressure',
    },
    end: {
      physical: 41,
      expenditure: 1860,
      revisedCost: 2980,
      delayMonths: 9,
      riskScore: 71,
      revisedCompletion: 'Dec 2026',
      riskSignal: 'Schedule deterioration · +9 months',
    },
  }),
  record({
    id: 'p12',
    name: 'Jaipur Metro Phase II Corridor',
    code: 'MOHUA/RJ/JM-P2',
    ministry: 'Housing & Urban Affairs',
    agency: 'JMRC',
    sector: 'Urban Transport',
    state: 'Rajasthan',
    delayProbability: 71,
    costOverrunProbability: 64,
    driver: 'Slow physical progress',
    primarySignal: 'Progress stagnation · 0 pts MoM',
    lastUpdated: '11 Apr 2026',
    originalCost: 3200,
    originalCompletion: 'Dec 2026',
    start: {
      physical: 18,
      expenditure: 720,
      revisedCost: 3400,
      delayMonths: 6,
      riskScore: 58,
      revisedCompletion: 'Jun 2027',
      riskSignal: 'Progress lag',
    },
    end: {
      physical: 29,
      expenditure: 1420,
      revisedCost: 3860,
      delayMonths: 14,
      riskScore: 76,
      revisedCompletion: 'Feb 2028',
      riskSignal: 'Progress stagnation · 0 pts MoM',
    },
    overrides: {
      5: { physicalProgress: 29, riskScore: 64, riskLevel: 'Medium' },
      6: { physicalProgress: 29, riskScore: 76, riskLevel: 'High' },
    },
  }),
]
