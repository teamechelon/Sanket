import { ArrowDownRight, ArrowUpRight } from 'lucide-react'
import { Sparkline } from '@/components/metrics/sparkline'

export function Metric({
  label,
  value,
  change,
  note,
  negative,
  values = [4, 6, 5, 8, 7, 9],
}: {
  label: string
  value: string
  change: string
  note: string
  negative?: boolean
  values?: number[]
}) {
  return (
    <div className="metric">
      <div className="metric-label">{label}</div>
      <div className="metric-main">
        <div className="metric-value">{value}</div>
        <Sparkline values={values} tone={negative ? 'risk' : 'healthy'} />
      </div>
      <div className={`metric-change ${negative ? 'negative' : ''}`}>
        {negative ? <ArrowUpRight /> : <ArrowDownRight />}
        {change}
        <span>{note}</span>
      </div>
    </div>
  )
}
