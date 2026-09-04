'use client'

import { useState } from 'react'
import { Activity, AlertTriangle, Download, LayoutDashboard, Settings, Table2, TrendingUp, X } from 'lucide-react'
import { SNAPSHOT_DATE } from '@/lib/sanket-data'

const navItems = [
  { label: 'Overview', icon: LayoutDashboard },
  { label: 'Projects', icon: Table2 },
  { label: 'Risk Monitor', icon: AlertTriangle, badgeKey: 'risk' as const },
  { label: 'Analytics', icon: TrendingUp },
  { label: 'Reports', icon: Download },
]

const utilityItems = [
  { label: 'Data Sources', icon: Activity },
  { label: 'Settings', icon: Settings },
]

export function Sidebar({
  active,
  setActive,
  open,
  onClose,
  riskBadgeCount,
}: {
  active: string
  setActive: (value: string) => void
  open: boolean
  onClose: () => void
  riskBadgeCount?: number
}) {
  const [expanded, setExpanded] = useState(false)
  const go = (label: string) => {
    setActive(label)
    onClose()
  }

  return (
    <aside
      className={`sidebar rail ${open ? 'sidebar-open' : ''} ${expanded ? 'rail-expanded' : ''}`}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      <div className="rail-header">
        <div className="rail-mark" title="SANKET">
          S
        </div>
        <button
          className="rail-toggle"
          onClick={() => setExpanded((value) => !value)}
          aria-label={expanded ? 'Collapse navigation' : 'Expand navigation'}
        >
          {expanded ? '‹' : '›'}
        </button>
        <button className="icon-button close-sidebar" onClick={onClose} aria-label="Close navigation">
          <X />
        </button>
      </div>
      {expanded && <div className="rail-section-label">MONITOR</div>}
      <nav className="nav-list">
        {navItems.map(({ label, icon: Icon, badgeKey }) => {
          const badge = badgeKey === 'risk' && riskBadgeCount != null ? String(riskBadgeCount) : undefined
          return (
            <button
              key={label}
              title={!expanded ? label : undefined}
              className={`nav-item ${active === label ? 'active' : ''}`}
              onClick={() => go(label)}
            >
              <Icon />
              <span>{label}</span>
              {badge && <span className="nav-badge">{badge}</span>}
            </button>
          )
        })}
      </nav>
      <div className="rail-divider" />
      {expanded && <div className="rail-section-label">SYSTEM</div>}
      <nav className="nav-list">
        {utilityItems.map(({ label, icon: Icon }) => (
          <button
            key={label}
            title={!expanded ? label : undefined}
            className={`nav-item ${active === label ? 'active' : ''}`}
            onClick={() => go(label)}
          >
            <Icon />
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-footer">
        <div className="sidebar-status" title="Prototype PAIMANA snapshot">
          <span className="status-dot" />
          <span>PAIMANA</span>
        </div>
        {expanded && (
          <div className="sidebar-version">
            Prototype environment
            <br />
            <small>Snapshot {SNAPSHOT_DATE}</small>
          </div>
        )}
      </div>
    </aside>
  )
}
