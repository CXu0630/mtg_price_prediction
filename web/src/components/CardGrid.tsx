import { useEffect, useLayoutEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react'
import { pctChange, scryfallImage, usd } from '../data'
import type { Backend, CustomCardInput, ListItem, PredictionRow } from '../types'

export const CARD_BACK = 'https://backs.scryfall.io/large/0/a/0aeebaf5-8c7d-4636-9e82-8c27447861f7.jpg'
const ADD_SLOT_DELAY_MS = 420 // the next slot turns up just after the added card starts flipping
const WAVE_STEP_MS = 90
const SHIFT_STEP_MS = 120 // matches the removal sequence in App.tsx

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
  /** each card turns into the one after it, then the old add slot turns away */
  removal: { id: string; phase: 'slot' | 'wave' } | null
  /** clearing the list: the first card turns into the add slot, the rest turn away to nothing */
  clearing: boolean
  /** the list was just cleared: the add slot is already showing, so it appears without flipping */
  cleared: boolean
  /** a card's details are taking over: every card turns away to nothing in the wave, and back again after */
  retract?: boolean
  /** reports the column count, which sets how long the diagonal wave takes */
  onCols?: (cols: number) => void
  onRemove: (id: string) => void
  onQuantity: (id: string, q: number) => void
  onOpen: (line: GridLine) => void
  onEmptySlot: () => void
}

export const fairOf = (r: PredictionRow, backend: Backend) =>
  backend === 'gp' && r.fair_gp != null
    ? { mid: r.fair_gp, lo: r.fair_gp_lo ?? null, hi: r.fair_gp_hi ?? null, model: 'Deep GP' }
    : { mid: r.fair, lo: r.fair_lo, hi: r.fair_hi, model: 'Boosted trees' }

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

function Tile({ front, back, replacement, showBack, delay, flipIn, shifting, away, onBackKind }: {
  front: ReactNode
  back: (kind: 'cardback' | 'price') => ReactNode
  replacement: ReactNode | null
  showBack: boolean
  delay: number
  flipIn: boolean
  shifting: boolean
  /** turn over to show this instead (clearing the list, or retracting for a card's details); null turns back */
  away: Away | null
  onBackKind?: 'price'
}) {
  // A tile mounted while `away` (e.g. during the page-load intro) starts out already turned away, unseen.
  const [deg, setDeg] = useState(flipIn || away ? 180 : showBack ? 180 : 0)
  const [backKind, setBackKind] = useState<'cardback' | 'price'>(flipIn ? 'cardback' : onBackKind ?? 'cardback')
  // Whichever face is turned away when `away` arrives is replaced, then the tile turns over to it.
  // When `away` clears, the tile turns over once more and the replaced face gets its content back after.
  const [awayOn, setAwayOn] = useState<'front' | 'back' | null>(away ? 'back' : null)
  const [awayKind, setAwayKind] = useState<Away | null>(away)

  useEffect(() => {
    // already showing the away face (mounted away): nothing to turn
    if (away && awayOn && awayOn === (Math.round(deg / 180) % 2 === 0 ? 'front' : 'back')) return
    if (away) {
      setAwayKind(away)
      setDeg((d) => {
        setAwayOn(Math.round(d / 180) % 2 === 0 ? 'back' : 'front')
        return d + 180
      })
    } else if (awayOn) setDeg((d) => d + 180)
  }, [away]) // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => { // flip a freshly added card face-up
    if (!flipIn) return
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setDeg(360)))
    return () => cancelAnimationFrame(t)
  }, [flipIn])

  useEffect(() => { // the replacement is on the hidden side before this turn begins
    if (shifting) setDeg((d) => d + 180)
  }, [shifting])

  useEffect(() => { // turn to whichever side `showBack` asks for, the short way round
    if (awayOn) return // turned away: coming back restores the right side
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
        onTransitionEnd={() => {
          if (facingFront && onBackKind && !shifting && !awayOn) setBackKind(onBackKind)
          if (!away && awayOn && awayOn !== (facingFront ? 'front' : 'back')) { setAwayOn(null); setAwayKind(null) }
        }}>
        <div className={`tile-face tile-front ${awayOn === 'front' ? `is-${awayKind}` : ''}`}>
          {awayOn === 'front' && awayKind ? <AwayFace kind={awayKind} /> : shifting && showBack ? replacement : front}
        </div>
        <div className={`tile-face tile-back ${awayOn === 'back' ? `is-${awayKind}` : ''}`}>
          {awayOn === 'back' && awayKind ? <AwayFace kind={awayKind} /> : shifting && !showBack ? replacement : back(backKind)}
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
 * grows the slot at the new position flips up while the old one becomes a real card.
 */
function AddSlot({ onClick, instant, hidden, disabled }: { onClick: () => void; instant: boolean; hidden: boolean; disabled: boolean }) {
  const [shown, setUp] = useState(instant)
  useEffect(() => {
    const t = window.setTimeout(() => setUp(true), ADD_SLOT_DELAY_MS)
    return () => window.clearTimeout(t)
  }, [])
  const up = shown && !hidden
  // -180 (turned away, unseen) -> 0 (up) -> 180 (away again) -> 360 ...: always turning left to right
  const [deg, setDeg] = useState(instant ? 0 : -180)
  useEffect(() => {
    setDeg((d) => {
      const isUp = Math.round(d / 180) % 2 === 0
      return isUp === up ? d : d + 180
    })
  }, [up])
  return (
    <div className="slot">
      <button type="button" className={`tile add-slot ${up ? 'up' : ''}`} onClick={onClick} aria-label="Add a card" disabled={disabled}>
        <span className="add-slot-face" style={{ transform: `rotateY(${deg}deg)` }}>
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
  const change = pctChange(r.price_today, r.pred_price)
  return (
    <button type="button" className="price-face" onClick={onOpen} aria-label={`Details for ${r.name}`}>
      <span className="pf-name">{r.name}</span>
      <span className="pf-label">Fair value · {fair.model}</span>
      <span className="pf-big">{usd(fair.mid)}</span>
      <span className="pf-range">{range(fair.lo, fair.hi) || '—'}</span>
      <span className="pf-rule" />
      <span className="pf-row">
        <span className="pf-label">Current</span>
        <span className="pf-value">{r.status === 'custom' ? 'custom' : usd(r.price_today)}</span>
      </span>
      <span className="pf-row">
        <span className="pf-label">In 14 days</span>
        <span className="pf-value">
          {change && <span className={`pf-change ${change.value > 0 ? 'up' : change.value < 0 ? 'down' : ''}`}>{change.text}</span>}
          {r.pred_price == null ? 'no forecast' : usd(r.pred_price)}
        </span>
      </span>
      {/* every row is always present (with a dash when empty) so faces line up across the grid */}
      <span className="pf-row pf-sub">
        <span className="pf-range">{range(r.price_lo, r.price_hi) || '—'}</span>
        {signal && <span className={`badge ${signal.toLowerCase()}`}>{signal}</span>}
      </span>
      <span className="pf-foot">CK buylist {usd(r.buylist_pred)}</span>
    </button>
  )
}

export function CardGrid({ lines, mode, backend, fresh, removal, clearing, cleared, retract = false, onCols, onRemove, onQuantity, onOpen, onEmptySlot }: Props) {
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

  useEffect(() => { onCols?.(cols) }, [cols]) // eslint-disable-line react-hooks/exhaustive-deps

  const filled = lines.length
  const wave = (i: number) => ((i % cols) + Math.floor(i / cols)) * WAVE_STEP_MS
  const removeAt = removal ? lines.findIndex((line) => line.item.id === removal.id) : -1

  return (
    <div className="card-grid" ref={gridRef}>
      {lines.map((line, i) => {
        const { item } = line
        const id = item.kind === 'existing' ? item.printing_key.split(':')[0] : undefined
        const name = item.kind === 'existing' ? (item.name ?? line.row?.name ?? '') : item.card.name
        const shifting = !!removal && i >= removeAt
        const next = lines[i + 1]
        const front = item.kind === 'existing'
          ? <img src={scryfallImage(id, 'normal')} alt={name} draggable={false} />
          : <CustomFront card={item.card} />
        const replacement = !shifting ? null : !next ? <AwayFace kind="addslot" /> : mode === 'prices'
          ? <PriceFace line={next} backend={backend} onOpen={() => onOpen(next)} />
          : next.item.kind === 'existing'
            ? <img src={scryfallImage(next.item.printing_key.split(':')[0], 'normal')} alt={next.item.name ?? next.row?.name ?? ''} draggable={false} />
            : <CustomFront card={next.item.card} />
        return (
          <div className={`slot ${shifting ? 'shifting' : ''} ${clearing || retract ? 'turning-away' : ''}`} key={item.id}
            style={{ '--turn-delay': `${shifting ? 0 : wave(i)}ms` } as CSSProperties}>
            <Tile front={front} replacement={replacement} showBack={mode === 'prices'}
              delay={shifting ? (i - removeAt) * SHIFT_STEP_MS : fresh.has(item.id) ? 0 : wave(i)}
              flipIn={fresh.has(item.id)} shifting={shifting} onBackKind="price"
              away={clearing ? (i === 0 ? 'addslot' : 'blank') : retract ? 'blank' : null}
              back={(kind) => (kind === 'price'
                ? <PriceFace line={line} backend={backend} onOpen={() => onOpen(line)} />
                : <img src={CARD_BACK} alt="" draggable={false} />)} />
            <div className="slot-controls">
              <label className="qty-pill">
                <span className="sr-only">Quantity of {name}</span>×
                <input type="number" min={1} value={item.quantity}
                  disabled={clearing || !!removal}
                  onChange={(e) => onQuantity(item.id, Math.max(1, Number(e.target.value) || 1))} />
              </label>
              <button type="button" className="icon" aria-label={`Remove ${name}`} disabled={clearing || !!removal} onClick={() => onRemove(item.id)}>Remove</button>
            </div>
          </div>
        )
      })}
      <AddSlot key={`add-${filled}`} onClick={onEmptySlot} instant={cleared}
        hidden={clearing || retract || removal?.phase === 'slot'} disabled={clearing || retract || !!removal} />
    </div>
  )
}
