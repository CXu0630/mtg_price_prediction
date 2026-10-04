import { useEffect, useMemo, useRef, useState } from 'react'
import { scryfallImage } from '../data'
import type { CustomCardInput } from '../types'
import { CARD_BACK, CustomFront } from './CardGrid'

/** A face the showcase can show: a real printing's image, or a custom card's proxy. */
export interface Face { key: string; src?: string; custom?: CustomCardInput }

export const TREASURE: Face = { key: 'treasure', src: scryfallImage('46a2c363-c879-4c36-b08a-dd57e3da19ae', 'normal') }

/**
 * What the page is doing decides what the showcase holds. Every change of state spins the card up and
 * lands it on the new state's card:
 *   cards   the list grid      cycles through the list (featured cards while it's empty)
 *   prices  predictions shown  Treasure token (LTR)
 *   search  adding a card      Vampiric Tutor (CMR)
 *   custom  custom-card form   Fabricate (HOP)
 *   card    one card's results the card itself
 *   docs    Documentation      Library of Leng (ME4)
 *   bots    Bot Trading        Stock Up (DFT)
 *   data    Data Exploration   Exploration (DMR)
 */
export type ShowcaseState = 'cards' | 'prices' | 'search' | 'custom' | 'card' | 'docs' | 'bots' | 'data'
export const STATE_CARD: Record<Exclude<ShowcaseState, 'cards' | 'card'>, Face> = {
  prices: TREASURE,
  search: { key: 'vampiric-tutor', src: scryfallImage('18bd50f2-c3ba-4217-a2d5-bb771e199706', 'normal') },
  custom: { key: 'fabricate', src: scryfallImage('44e49332-b1e1-4b88-b508-516b55c67a1d', 'normal') },
  docs: { key: 'library-of-leng', src: scryfallImage('c2fbf45f-9599-42e2-8510-7e20731e6c68', 'normal') },
  bots: { key: 'stock-up', src: scryfallImage('0a786855-6eb4-42c0-a528-4842db46809d', 'normal') },
  data: { key: 'exploration', src: scryfallImage('5b372045-a4a0-44c8-96ec-1e201d61ed26', 'normal') },
}

/** Shown while the list is empty: a few iconic printings. */
export const FEATURED: Face[] = [
  'b0faa7f2-b547-42c4-a810-839da50dadfe', // Black Lotus
  '10a4e202-b564-4ac8-8bdf-a2a563662563', // The One Ring
  'a9738cda-adb1-47fb-9f4c-ecd930228c4d', // Ragavan, Nimble Pilferer
  '0e606072-a3aa-4300-ba90-ec92a721fa76', // Jace, the Mind Sculptor
  '4cbc6901-6a4a-4d0a-83ea-7eefa3b35021', // Sol Ring
].map((id) => ({ key: id, src: scryfallImage(id, 'normal') }))

const CRUISE_DEG_PER_S = 360 / 11
const BURST_TURNS = 3
const BURST_MS = 1300

interface Props {
  /** faces to cycle through while cruising (empty -> featured cards) */
  deck: Face[]
  /** bump `id` to spin up fast and land face-up on `face` (or on the current card when omitted) */
  burst: { id: number; face?: Face } | null
}

const reduceMotion = () => window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false
const easeOutCubic = (t: number) => 1 - (1 - t) ** 3

/**
 * A card turning on a vertical axis, driven by requestAnimationFrame so it can cruise slowly,
 * burst into a fast spin, and land exactly face-up. The front face is only ever replaced while
 * the back is toward the viewer (on crossing 180 deg mod 360), so swaps are never visible.
 */
export function CardShowcase({ deck: deckIn, burst }: Props) {
  const deck = useMemo(() => (deckIn.length ? deckIn : FEATURED), [deckIn])
  const [front, setFront] = useState<Face>(deck[0])
  const cardRef = useRef<HTMLDivElement>(null)
  const angle = useRef(180)
  const index = useRef(0)
  const deckRef = useRef(deck)
  const burstState = useRef<{ a0: number; aT: number; t0: number; face?: Face; swapped: boolean } | null>(null)

  useEffect(() => { // a new deck: re-sync, and show its first card if we're not mid-burst
    deckRef.current = deck
    if (!burstState.current && !deck.some((f) => f.key === front.key)) {
      index.current = 0
      if (reduceMotion()) setFront(deck[0])
    }
  }, [deck]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!burst) return
    if (reduceMotion()) { if (burst.face) setFront(burst.face); return }
    const a0 = angle.current
    burstState.current = { a0, aT: Math.ceil((a0 + BURST_TURNS * 360) / 360) * 360, t0: performance.now(), face: burst.face, swapped: false }
  }, [burst?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (reduceMotion()) {
      if (cardRef.current) cardRef.current.style.transform = 'none'
      return
    }
    let raf = 0
    let last = performance.now()
    const tick = (now: number) => {
      const dt = Math.min(now - last, 100) / 1000
      last = now
      const prev = angle.current
      const b = burstState.current
      if (b) {
        // Decelerate onto the cruise speed rather than to a stop, so the hand-off is seamless:
        // a(t) = a0 + v*t + (span - v*D) * easeOut(t/D) lands on aT moving at exactly v.
        const elapsed = Math.min((now - b.t0) / 1000, BURST_MS / 1000)
        const span = b.aT - b.a0
        const d = BURST_MS / 1000
        angle.current = b.a0 + CRUISE_DEG_PER_S * elapsed + (span - CRUISE_DEG_PER_S * d) * easeOutCubic(elapsed / d)
        const t = elapsed / d
        if (t >= 1) { // landed face-up on the burst face: cruise on from it
          burstState.current = null
          const i = b.face ? deckRef.current.findIndex((f) => f.key === b.face!.key) : -1
          index.current = i >= 0 ? i : index.current
        }
      } else {
        angle.current += CRUISE_DEG_PER_S * dt
      }
      // crossed a back-facing point (180 + 360k): safe moment to change the front face
      if (Math.floor((angle.current - 180) / 360) !== Math.floor((prev - 180) / 360)) {
        if (b && !b.swapped) {
          b.swapped = true
          if (b.face) setFront(b.face)
        } else if (!b) {
          const d = deckRef.current
          index.current = (index.current + 1) % d.length
          setFront(d[index.current])
        }
      }
      if (cardRef.current) cardRef.current.style.transform = `rotateY(${angle.current % 720}deg)`
      raf = requestAnimationFrame(tick)
    }
    raf = requestAnimationFrame(tick)
    return () => cancelAnimationFrame(raf)
  }, [])

  useEffect(() => { // warm the cache for every face so swaps never show a blank card
    for (const f of [...deck, ...Object.values(STATE_CARD)]) if (f.src) new Image().src = f.src
  }, [deck])

  return (
    <div className="showcase" aria-hidden="true">
      <div className="stage">
        <div className="card3d" ref={cardRef}>
          {front.custom
            ? <div className="face front face-custom"><CustomFront card={front.custom} /></div>
            : <img className="face front" src={front.src} alt="" draggable={false} />}
          <img className="face back" src={CARD_BACK} alt="" draggable={false} />
        </div>
      </div>
    </div>
  )
}
