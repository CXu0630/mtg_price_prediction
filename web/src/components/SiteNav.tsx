import { useEffect, useRef, useState } from 'react'
import { PAGES, type PageId } from '../content/pages'

const STEP_MS = 70
const TURN_MS = 700

/**
 * The four page buttons, drawn as little white cards stacked down the left edge under the brand. They don't
 * exist until the brand is clicked; then they turn up out of the background one after another (top to bottom,
 * each turning left to right), and turn away again on close. Each card is two-sided: white, and a black side
 * for the current page; picking a page turns its card over to black and the old one back to white.
 */
export function SiteNav({ open, page, onPick }: { open: boolean; page: PageId; onPick: (p: PageId) => void }) {
  const [mounted, setMounted] = useState(open)
  const [up, setUp] = useState(false)
  // While the cards deal in or out, each card's unseen side is hidden: otherwise a card turned away from the
  // viewer would show that other side instead of disappearing into the background.
  const [dealing, setDealing] = useState(false)
  useEffect(() => {
    const dealMs = TURN_MS + (PAGES.length - 1) * STEP_MS
    setDealing(true)
    if (open) {
      setMounted(true)
      const t = requestAnimationFrame(() => requestAnimationFrame(() => setUp(true)))
      const d = window.setTimeout(() => setDealing(false), dealMs + 50)
      return () => { cancelAnimationFrame(t); window.clearTimeout(d) }
    }
    setUp(false)
    const t = window.setTimeout(() => setMounted(false), dealMs)
    return () => window.clearTimeout(t)
  }, [open])
  // Per-card angle of the white/black flip: only ever grows, so every turn goes left to right.
  const [degs, setDegs] = useState<Record<string, number>>(() =>
    Object.fromEntries(PAGES.map((p) => [p.id, p.id === page ? 180 : 0])))
  const last = useRef(page)
  useEffect(() => {
    if (page === last.current) return
    const prev = last.current
    last.current = page
    setDegs((d) => ({ ...d, [prev]: d[prev] + 180, [page]: d[page] + 180 }))
  }, [page])

  if (!mounted) return null
  return (
    <nav className={`site-nav ${dealing ? 'dealing' : ''}`} aria-label="Pages">
      {PAGES.map((p, i) => (
        <button key={p.id} type="button" className={`nav-card ${up ? 'face-up' : 'turned-out'} ${p.id === page ? 'current' : ''}`}
          style={{ transitionDelay: `${(up ? i : PAGES.length - 1 - i) * STEP_MS}ms` }}
          aria-current={p.id === page ? 'page' : undefined} tabIndex={up ? 0 : -1}
          onClick={() => onPick(p.id)}>
          <span className={`nav-flip ${Math.round(degs[p.id] / 180) % 2 === 0 ? 'white-up' : 'black-up'}`} style={{ transform: `rotateY(${degs[p.id]}deg)` }}>
            <span className="nav-face"><span className="nav-label">{p.label}</span></span>
            <span className="nav-face nav-back" aria-hidden="true"><span className="nav-label">{p.label}</span></span>
          </span>
        </button>
      ))}
    </nav>
  )
}
