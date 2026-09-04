'use client'

import { DataStatus } from '@/components/layout/data-status'
import { PERIOD_LABEL, SNAPSHOT_DATE, SNAPSHOT_LABEL, TOTAL_MONITORED } from '@/lib/sanket-data'

export function DataSourcesPage() {
  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="eyebrow">SYSTEM / DATA PROVENANCE</div>
          <h1>Data Sources</h1>
          <p>
            Prototype data layer structured for a future REST API. <span className="demo-label">PROTOTYPE DATA</span>
          </p>
        </div>
      </div>
      <section className="panel settings-panel">
        <div className="settings-row">
          <div>
            <strong>PAIMANA Flash Reports</strong>
            <span>Primary prototype source — PDF/CSV flash report snapshots</span>
          </div>
          <span className="signal-ok">Connected (mock)</span>
        </div>
        <div className="settings-row">
          <div>
            <strong>Snapshot</strong>
            <span>{SNAPSHOT_LABEL}</span>
          </div>
          <span className="mono-cell">{SNAPSHOT_DATE}</span>
        </div>
        <div className="settings-row">
          <div>
            <strong>Monitored records</strong>
            <span>All pages share the same {TOTAL_MONITORED} longitudinal project records</span>
          </div>
          <span>{TOTAL_MONITORED}</span>
        </div>
        <div className="settings-row">
          <div>
            <strong>Live feed</strong>
            <span>Not connected. Do not treat UI as real-time PAIMANA.</span>
          </div>
          <span className="signal-neutral">Offline prototype</span>
        </div>
      </section>
      <DataStatus />
    </div>
  )
}

export function SettingsPage() {
  return (
    <div className="page-content">
      <div className="page-header">
        <div>
          <div className="eyebrow">SYSTEM / PREFERENCES</div>
          <h1>Settings</h1>
          <p>Local monitoring cell preferences for this prototype session.</p>
        </div>
      </div>
      <section className="panel settings-panel">
        <div className="settings-row">
          <div>
            <strong>Operator</strong>
            <span>A. Sharma · Monitoring Cell</span>
          </div>
        </div>
        <div className="settings-row">
          <div>
            <strong>Default period</strong>
            <span>{PERIOD_LABEL} snapshot</span>
          </div>
        </div>
        <div className="settings-row">
          <div>
            <strong>Watchlist storage</strong>
            <span>Browser localStorage only — no server persistence</span>
          </div>
        </div>
        <div className="settings-row">
          <div>
            <strong>Theme</strong>
            <span>Institutional light — fixed for this release</span>
          </div>
        </div>
      </section>
      <DataStatus />
    </div>
  )
}
