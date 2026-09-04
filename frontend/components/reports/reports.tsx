'use client'

import { useMemo, useState } from 'react'
import { ChevronRight, Download, Eye, FileText } from 'lucide-react'
import { DataStatus } from '@/components/layout/data-status'
import {
  ministries,
  PERIOD_LABEL,
  projects,
  reportTypes,
  sectors,
  SNAPSHOT_DATE,
  states,
  TOTAL_MONITORED,
} from '@/lib/sanket-data'

export function Reports() {
  const [reportId, setReportId] = useState(reportTypes[0].id)
  const [period, setPeriod] = useState(PERIOD_LABEL)
  const [state, setState] = useState('All States')
  const [ministry, setMinistry] = useState('All Ministries')
  const [sector, setSector] = useState('All Sectors')
  const [risk, setRisk] = useState('All risks')
  const [preview, setPreview] = useState(false)
  const [exported, setExported] = useState(false)

  const report = reportTypes.find((r) => r.id === reportId) ?? reportTypes[0]

  const scoped = useMemo(() => {
    return projects.filter((p) => {
      const matchState = state === 'All States' || p.state === state
      const matchMinistry = ministry === 'All Ministries' || p.ministry === ministry
      const matchSector = sector === 'All Sectors' || p.sector === sector
      const matchRisk = risk === 'All risks' || p.risk === risk
      if (reportId === 'high-risk') return ['High', 'Critical'].includes(p.risk) && matchState && matchMinistry && matchSector
      return matchState && matchMinistry && matchSector && matchRisk
    })
  }, [state, ministry, sector, risk, reportId])

  const handleExport = () => {
    const lines = [
      `SANKET ${report.label}`,
      `Period: ${period}`,
      `Snapshot: ${SNAPSHOT_DATE} · PROTOTYPE DATA`,
      `Scope: ${state} · ${ministry} · ${sector} · ${risk}`,
      `Records: ${scoped.length}`,
      '',
      ...scoped.map(
        (p) =>
          `${p.code}\t${p.name}\t${p.state}\t${p.risk}\t${p.riskScore}\t${p.primarySignal}`,
      ),
    ]
    const blob = new Blob([lines.join('\n')], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `sanket-${reportId}-${period.replace(/\s+/g, '-').toLowerCase()}.txt`
    a.click()
    URL.revokeObjectURL(url)
    setExported(true)
    setTimeout(() => setExported(false), 2500)
  }

  return (
    <div className="page-content reports-page">
      <div className="page-header">
        <div>
          <div className="eyebrow">DECISION PACKS / EXPORT CENTER</div>
          <h1>Reports</h1>
          <p>
            Generate structured monitoring reports from the current prototype dataset.{' '}
            <span className="demo-label">PROTOTYPE DATA</span>
          </p>
        </div>
      </div>

      <section className="panel report-builder">
        <div className="report-options">
          <div className="eyebrow">SELECT REPORT TYPE</div>
          {reportTypes.map((item) => (
            <button
              key={item.id}
              className={reportId === item.id ? 'active' : ''}
              onClick={() => {
                setReportId(item.id)
                setPreview(false)
              }}
            >
              <span>{item.label}</span>
              <ChevronRight />
            </button>
          ))}
        </div>
        <div className="report-form">
          <div className="eyebrow">REPORT PARAMETERS</div>
          <h2>{report.label}</h2>
          <p>{report.description}</p>

          <label>
            Period
            <select value={period} onChange={(e) => setPeriod(e.target.value)}>
              <option>{PERIOD_LABEL}</option>
              <option>MARCH 2026</option>
              <option>FEBRUARY 2026</option>
            </select>
          </label>
          <label>
            State
            <select value={state} onChange={(e) => setState(e.target.value)}>
              {states.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </label>
          <label>
            Ministry
            <select value={ministry} onChange={(e) => setMinistry(e.target.value)}>
              {ministries.map((m) => (
                <option key={m}>{m}</option>
              ))}
            </select>
          </label>
          <label>
            Sector
            <select value={sector} onChange={(e) => setSector(e.target.value)}>
              {sectors.map((s) => (
                <option key={s}>{s}</option>
              ))}
            </select>
          </label>
          <label>
            Risk
            <select value={risk} onChange={(e) => setRisk(e.target.value)}>
              <option>All risks</option>
              <option>Critical</option>
              <option>High</option>
              <option>Medium</option>
              <option>Low</option>
            </select>
          </label>

          <div className="report-meta">
            <span>
              Records included <strong>{scoped.length} of {TOTAL_MONITORED} monitored</strong>
            </span>
            <span>
              Data snapshot <strong>{SNAPSHOT_DATE}</strong>
            </span>
          </div>

          <div className="report-actions">
            <button className="outline-button" onClick={() => setPreview(true)}>
              <Eye />
              Preview
            </button>
            <button className="primary-button" onClick={handleExport}>
              <Download />
              Export
            </button>
          </div>
          {exported && <div className="export-toast">Mock report downloaded (frontend only).</div>}
        </div>
      </section>

      {preview && (
        <section className="panel report-preview">
          <div className="report-preview-head">
            <FileText />
            <div>
              <div className="eyebrow">PREVIEW</div>
              <h2>
                {report.label} — {period}
              </h2>
              <p>
                {state} · {ministry} · {sector} · {risk}
              </p>
            </div>
          </div>
          <div className="table-wrap">
            <table className="data-table">
              <thead>
                <tr>
                  <th>Project</th>
                  <th>Code</th>
                  <th>State</th>
                  <th>Risk</th>
                  <th>Score</th>
                  <th>Primary Signal</th>
                </tr>
              </thead>
              <tbody>
                {scoped.slice(0, 12).map((p) => (
                  <tr key={p.id}>
                    <td>{p.name}</td>
                    <td className="mono-cell">{p.code}</td>
                    <td>{p.state}</td>
                    <td>{p.risk}</td>
                    <td>{p.riskScore}</td>
                    <td>{p.primarySignal}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      <DataStatus />
    </div>
  )
}
