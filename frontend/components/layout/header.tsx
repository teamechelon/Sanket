'use client'

import { Bell, ChevronRight, Menu, Search } from 'lucide-react'
import { SNAPSHOT_LABEL } from '@/lib/sanket-data'

export function Header({ onMenu, onSearch }: { onMenu: () => void; onSearch: () => void }) {
  return (
    <header className="topbar">
      <button className="mobile-menu" onClick={onMenu} aria-label="Open navigation">
        <Menu />
      </button>
      <div className="brand">
        <div className="brand-mark">S</div>
        <div>
          <div className="brand-name">SANKET</div>
          <div className="brand-subtitle">Infrastructure Project Intelligence</div>
        </div>
      </div>
      <button className="global-search-trigger" onClick={onSearch}>
        <Search />
        <span>Search projects, states, ministries...</span>
        <kbd>⌘ K</kbd>
      </button>
      <div className="topbar-right">
        <div className="snapshot">
          <span className="status-pip" />
          <strong className="system-online">SYSTEM ONLINE</strong>
          <span className="snapshot-label">SNAPSHOT</span>
          <strong>{SNAPSHOT_LABEL.replace('PAIMANA SNAPSHOT · ', '')}</strong>
        </div>
        <button className="icon-button" aria-label="Notifications">
          <Bell />
          <span className="notification-dot" />
        </button>
        <div className="profile">
          <div className="avatar">AS</div>
          <div className="profile-copy">
            <strong>A. Sharma</strong>
            <span>Monitoring Cell</span>
          </div>
          <ChevronRight className="profile-chevron" />
        </div>
      </div>
    </header>
  )
}
