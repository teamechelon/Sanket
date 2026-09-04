export function SkeletonBlock({ className = '' }: { className?: string }) {
  return <div className={`skeleton-block ${className}`} aria-hidden="true" />
}

export function LoadingState({ label = 'Loading snapshot…' }: { label?: string }) {
  return (
    <div className="ui-state" role="status">
      <div className="skeleton-stack">
        <SkeletonBlock className="skeleton-wide" />
        <SkeletonBlock />
        <SkeletonBlock className="skeleton-mid" />
      </div>
      <span>{label}</span>
    </div>
  )
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="ui-state empty-state">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  )
}

export function ErrorState({ title = 'Unable to load data', detail }: { title?: string; detail: string }) {
  return (
    <div className="ui-state error-state" role="alert">
      <strong>{title}</strong>
      <span>{detail}</span>
    </div>
  )
}
