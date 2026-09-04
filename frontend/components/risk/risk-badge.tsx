import { riskClass, type RiskLevel } from '@/lib/sanket-data'

export function RiskBadge({ risk }: { risk: RiskLevel }) {
  return (
    <span className={`risk-badge ${riskClass(risk)}`}>
      <span className="risk-dot" />
      {risk}
    </span>
  )
}
