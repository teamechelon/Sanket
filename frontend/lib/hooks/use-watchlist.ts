'use client'

import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'sanket-watchlist'

function readWatchlist(): string[] {
  if (typeof window === 'undefined') return []
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as unknown
    return Array.isArray(parsed) ? parsed.filter((id): id is string => typeof id === 'string') : []
  } catch {
    return []
  }
}

export function useWatchlist() {
  const [ids, setIds] = useState<string[]>([])

  useEffect(() => {
    setIds(readWatchlist())
  }, [])

  const persist = useCallback((next: string[]) => {
    setIds(next)
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  }, [])

  const toggle = useCallback(
    (id: string) => {
      persist(ids.includes(id) ? ids.filter((item) => item !== id) : [...ids, id])
    },
    [ids, persist],
  )

  const isWatched = useCallback((id: string) => ids.includes(id), [ids])

  return { ids, toggle, isWatched }
}
