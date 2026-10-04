import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { scryfallImage, usd } from '../data'
import type { Backend, CustomCardInput, ListItem, PredictionRow } from '../types'

export const CARD_BACK = 'https://backs.scryfall.io/large/0/a/0aeebaf5-8c7d-4636-9e82-8c27447861f7.jpg'
const ADD_SLOT_DELAY_MS = 420 // the next slot turns up just after the added card starts flipping
const WAVE_STEP_MS = 90

export interface GridLine {
  item: ListItem
  row: PredictionRow | null
  error?: string
}

interface Props {
  lines: GridLine[]
  mode: 'cards' | 'prices'
  backend: Backend
  /** ids of cards just added: they flip in from face-down instead of appearing face-up */
  fresh: Set<string>
  /** ids of cards being removed: they turn away face-down and fade before leaving the grid */
  leaving: Set<string>
  /** clearing the list: the first card turns into the add slot, the rest turn away to nothing */
  clearing: boolean
  /** the list was just cleared: the add slot is already showing, so it appears without flipping */
  cleared: boolean
  onRemove: (id: string) => void
  onQuantity: (id: string, q: number) => void
  onOpen: (line: GridLine) => void
  onEmptySlot: () => void
}

export const fairOf = (r: PredictionRow, backend: Backend) =>
  backend === 'gp' && r.fair_gp != null
    ? { mid: r.fair_gp, lo: r.fair_gp_lo ?? null, hi: r.fair_gp_hi ?? null, model: 'Deep GP' }
    : { mid: r.fair, lo: r.fair_lo, hi: r.fair_hi, model: 'Model A' }

export const range = (lo: number | null | undefined, hi: number | null | undefined) =>
  lo == null || hi == null ? '' : `${usd(lo)} – ${usd(hi)}`

/**
 * One slot. Rotation accumulates (0 -> 180 -> 360 ...) so every flip turns the same way.
 * Even multiples of 180 show the front face. What the BACK face shows (card back vs. prices)
 * only changes while the back is turned away, so content never swaps in view.
 */
type Away = 'blank' | 'addslot'

function AwayFace({ kind }: { kind: Away }) {
  return kind === 'blank'
    ? <div className="away-blank" />
    : <div className="away-addslot"><img src={CARD_BACK} alt="" draggable={false} /><span className="empty-hint">+ Add a card</span></div>
}

function Tile({ front, back, showBack, delay, flipIn, leaving, away, onBackKind }: {
  front: ReactNode
  back: (kind: 'cardback' | 'price') => ReactNode
  showBack: boolean
  delay: number
  flipIn: boolean
  leaving: boolean
  /** turn over to show this instead (clearing the list) */
  away: Away | null
  onBackKind?: 'price'
}) {
  const [deg, setDeg] = useState(flipIn ? 180 : showBack ? 180 : 0)
  const [backKind, setBackKind] = useState<'cardback' | 'price'>(flipIn ? 'cardback' : onBackKind ?? 'cardback')
  // Whichever face is turned away when `away` arrives is replaced, then the tile turns over to it.
  const [awayOn, setAwayOn] = useState<'front' | 'back' | null>(null)

  useEffect(() => {
    if (!away) return
    setDeg((d) => {
      setAwayOn(Math.round(d / 180) % 2 === 0 ? 'back' : 'front')
      return d + 180
    })
  }, [away])

  useEffect(() => { // flip a freshly added card face-up
    if (!flipIn) return
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setDeg(360)))
    return () => cancelAnimationFrame(t)
  }, [flipIn])

  useEffect(() => { // leaving: turn away (face-down if the card face was showing) while the slot fades
    if (!leaving) return
    setDeg((d) => {
      if (Math.round(d / 180) % 2 === 0) setBackKind('cardback') // back is hidden: safe to swap
      return d + 180
    })
  }, [leaving])

  useEffect(() => { // turn to whichever side `showBack` asks for, the short way round
    setDeg((d) => {
      const facingBack = Math.round(d / 180) % 2 === 1
      return facingBack === showBack ? d : d + 180
    })
  }, [showBack])

  const facingFront = Math.round(deg / 180) % 2 === 0
  return (
    <div className="tile">
      <div className={`tile-inner ${facingFront ? 'front-up' : 'back-up'}`}
        style={{ transform: `rotateY(${deg}deg)`, transitionDelay: `${delay}ms` }}
        onTransitionEnd={() => { if (facingFront && onBackKind && !leaving) setBackKind(onBackKind) }}>
        <div className={`tile-face tile-front ${awayOn === 'front' ? `is-${away}` : ''}`}>
          {awayOn === 'front' && away ? <AwayFace kind={away} /> : front}
        </div>
        <div className={`tile-face tile-back ${awayOn === 'back' ? `is-${away}` : ''}`}>
          {awayOn === 'back' && away ? <AwayFace kind={away} /> : back(backKind)}
        </div>
      </div>
    </div>
  )
}

export function CustomFront({ card }: { card: CustomCardInput }) {
  const pt = card.power != null || card.toughness != null ? `${card.power ?? '–'}/${card.toughness ?? '–'}` : card.loyalty ?? ''
  return (
    <div className="proxy">
      <div className="proxy-head"><span>{card.name}</span><span className="proxy-cost">{card.mana_cost}</span></div>
      <div className="proxy-art">custom</div>
      <div className="proxy-type">{card.type_line}</div>
      <div className="proxy-text">{card.oracle_text}</div>
      <div className="proxy-foot"><span>{card.rarity} · {card.finish}</span><span>{pt}</span></div>
    </div>
  )
}

/**
 * The one face-down card waiting for the next addition. It mounts turned away (invisible against
 * the page) and flips up to show the card back; it is keyed by grid position, so each time the list
 * grows or shrinks the slot at the new position flips up while the old one becomes a real card.
 */
function AddSlot({ onClick, instant, hidden }: { onClick: () => void; instant: boolean; hidden: boolean }) {
  const [shown, setUp] = useState(instant)
  useEffect(() => {
    const t = window.setTimeout(() => setUp(true), ADD_SLOT_DELAY_MS)
    return () => window.clearTimeout(t)
  }, [])
  const up = shown && !hidden
  return (
    <div className="slot">
      <button type="button" className={`tile add-slot ${up ? 'up' : ''}`} onClick={onClick} aria-label="Add a card">
        <span className="add-slot-face">
          <img src={CARD_BACK} alt="" draggable={false} />
          <span className="empty-hint">+ Add a card</span>
        </span>
      </button>
    </div>
  )
}

function PriceFace({ line, backend, onOpen }: { line: GridLine; backend: Backend; onOpen: () => void }) {
  const r = line.row
  if (!r) return <div className="price-face muted">{line.error ?? 'Pricing…'}</div>
  const fair = fairOf(r, backend)
  const signal = r.signal ?? null
  return (
    <button type="button" className="price-face" onClick={onOpen} aria-label={`Details for ${r.name}`}>
      <span className="pf-name">{r.name}</span>
      <span className="pf-label">Fair value · {fair.model}</span>
      <span className="pf-big">{usd(fair.mid)}</span>
      <span className="pf-range">{range(fair.lo, fair.hi)}</span>
      <span className="pf-rule" />
      <span className="pf-label">14 days · Model B</span>
      {r.pred_price == null
        ? <span className="pf-range">not traded — no forecast</span>
        : <>
          <span className="pf-mid">
            {usd(r.pred_price)}
            {signal && <span className={`badge ${signal.toLowerCase()}`}>{signal}</span>}
          </span>
          <span className="pf-range">{range(r.price_lo, r.price_hi)}</span>
        </>}
      <span className="pf-foot">
        {r.status === 'custom' ? 'custom card' : r.price_today == null ? 'no market price' : `market ${usd(r.price_today)}`}
        {r.buylist_pred != null && ` · buylist ${usd(r.buylist_pred)}`}
      </span>
    </button>
  )
}

export function CardGrid({ lines, mode, backend, fresh, leaving, clearing, cleared, onRemove, onQuantity, onOpen, onEmptySlot }: Props) {
  const gridRef = useRef<HTMLDivElement>(null)
  const [cols, setCols] = useState(4)
  useLayoutEffect(() => { // column count drives the diagonal wave
    const el = gridRef.current
    if (!el) return
    const measure = () => setCols(getComputedStyle(el).gridTemplateColumns.split(' ').length || 1)
    measure()
    const ro = new ResizeObserver(measure)
    ro.observe(el)
    return () => ro.disconnect()
  }, [])

  const filled = lines.length
  const wave = (i: number) => ((i % cols) + Math.floor(i / cols)) * WAVE_STEP_MS

  return (
    <div className="card-grid" ref={gridRef}>
      {lines.map((line, i) => {
        const { item } = line
        const id = item.kind === 'existing' ? item.printing_key.split(':')[0] : undefined
        const name = item.kind === 'existing' ? (item.name ?? line.row?.name ?? '') : item.card.name
        const front = item.kind === 'existing'
          ? <img src={scryfallImage(id, 'normal')} alt={name} draggable={false} />
          : <CustomFront card={item.card} />
        return (
          <div className={`slot ${leaving.has(item.id) ? 'leaving' : ''}`} key={item.id}>
            <Tile front={front} showBack={mode === 'prices'} delay={fresh.has(item.id) || leaving.has(item.id) ? 0 : wave(i)}
              flipIn={fresh.has(item.id)} leaving={leaving.has(item.id)} onBackKind="price"
              away={clearing ? (i === 0 ? 'addslot' : 'blank') : null}
              back={(kind) => (kind === 'price'
                ? <PriceFace line={line} backend={backend} onOpen={() => onOpen(line)} />
                : <img src={CARD_BACK} alt="" draggable={false} />)} />
            <div className="slot-controls">
              <label className="qty-pill">
                <span className="sr-only">Quantity of {name}</span>×
                <input type="number" min={1} value={item.quantity}
                  onChange={(e) => onQuantity(item.id, Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="button" className="icon" aria-label={`Remove ${name}`} onClick={() => onRemove(item.id)}>Remove</button>
            </div>
          </div>
        )
      })}
      <AddSlot key={`add-${filled}`} onClick={onEmptySlot} instant={cleared} hidden={clearing} />
    </div>
  )
}
