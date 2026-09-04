export function Sparkline({
  values,
  tone = 'accent',
}: {
  values: number[]
  tone?: 'accent' | 'risk' | 'healthy'
}) {
  const min = Math.min(...values)
  const max = Math.max(...values)
  const points = values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * 100
      const y = 28 - ((value - min) / Math.max(max - min, 1)) * 22
      return `${x},${y}`
    })
    .join(' ')

  return (
    <svg className={`sparkline sparkline-${tone}`} viewBox="0 0 100 32" preserveAspectRatio="none" aria-hidden="true">
      <polyline points={points} fill="none" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}
