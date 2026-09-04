import { PERIOD_LABEL, TOTAL_MONITORED } from '@/lib/sanket-data'

export function DataStatus() {
  return (
    <div className="data-status">
      <div className="data-status-title">
        <span className="status-pip" />
        Data status
      </div>
      <span>
        Last updated: <strong>{PERIOD_LABEL}</strong>
      </span>
      <span>
        Source: <strong>PAIMANA Flash Reports</strong>
      </span>
      <span>
        Records processed: <strong>{TOTAL_MONITORED}</strong>
      </span>
      <span>
        Extraction quality: <strong className="quality">94.6%</strong>
      </span>
      <span className="prototype-note">Prototype / demo data — not a live feed</span>
    </div>
  )
}
