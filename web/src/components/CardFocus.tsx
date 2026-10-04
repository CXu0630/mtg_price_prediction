import { useEffect, useState } from 'react'
import { pctChange, scryfallImage, usd } from '../data'
import type { Backend } from '../types'
import { fairOf, range, type GridLine } from './CardGrid'

/** Printed details for the focused card: Scryfall for real printings, the form's fields for custom cards. */
export interface CardInfo {
  name: string
  set_name?: string
  set?: string
  collector_number?: string
  rarity?: string
  finish?: string
  mana_cost?: string
  type_line?: string
  oracle_text?: string
  power?: string
  toughness?: string
  loyalty?: string
  keywords?: string[]
  artist?: string
  released_at?: string
}

const cache = new Map<string, Promise<CardInfo>>()

function fetchScryfall(id: string): Promise<CardInfo> {
  let p = cache.get(id)
  if (!p) {
    p = fetch(`https://api.scryfall.com/cards/${id}`).then(async (r) => {
      if (!r.ok) throw new Error(`Scryfall ${r.status}`)
      const d = await r.json()
      const faces: Record<string, string>[] = d.card_faces ?? []
      const fromFaces = (k: string) => faces.map((f) => f[k]).filter(Boolean).join(' // ') || undefined
      return {
        name: d.name, set_name: d.set_name, set: d.set, collector_number: d.collector_number, rarity: d.rarity,
        mana_cost: d.mana_cost || fromFaces('mana_cost'), type_line: d.type_line,
        oracle_text: d.oracle_text ?? faces.map((f) => f.oracle_text).filter(Boolean).join('\n—\n'),
        power: d.power, toughness: d.toughness, loyalty: d.loyalty, keywords: d.keywords,
        artist: d.artist, released_at: d.released_at,
      }
    })
    p.catch(() => cache.delete(id)) // let a later open retry
    cache.set(id, p)
  }
  return p
}

export function useCardInfo(line: GridLine | null): CardInfo | null {
  const [info, setInfo] = useState<CardInfo | null>(null)
  const item = line?.item
  const key = item ? (item.kind === 'existing' ? item.printing_key : item.id) : ''
  useEffect(() => {
    if (!line) { setInfo(null); return }
    const { item: it, row } = line
    if (it.kind === 'custom') {
      const c = it.card
      setInfo({ name: c.name, rarity: c.rarity, finish: c.finish, mana_cost: c.mana_cost, type_line: c.type_line,
        oracle_text: c.oracle_text, power: c.power, toughness: c.toughness, loyalty: c.loyalty,
        set_name: 'Custom card', released_at: c.released_at })
      return
    }
    // show what the price row already knows straight away, then fill in the printed details
    const base: CardInfo = { name: row?.name ?? it.name ?? '', set: it.set, set_name: row?.set_name,
      collector_number: row?.collector_number, rarity: row?.rarity, finish: row?.finish }
    setInfo(base)
    let live = true
    fetchScryfall(it.printing_key.split(':')[0])
      .then((d) => { if (live) setInfo({ ...d, finish: base.finish }) })
      .catch(() => {})
    return () => { live = false }
  }, [key]) // eslint-disable-line react-hooks/exhaustive-deps
  return info
}

/** The masthead side for a focused card: its name and what's printed on it. */
export function FocusHeader({ info }: { info: CardInfo | null }) {
  if (!info) return null
  const where = [info.set_name && `${info.set_name}${info.set ? ` (${info.set.toUpperCase()})` : ''}`,
    info.collector_number && `#${info.collector_number}`, info.rarity, info.finish].filter(Boolean).join(' · ')
  const pt = info.power != null ? `${info.power}/${info.toughness}` : info.loyalty ? `Loyalty ${info.loyalty}` : ''
  return (
    <div className="focus-head">
      <p className="eyebrow">{where}</p>
      <h1 className="focus-name">{info.name}</h1>
      <p className="focus-type">
        {info.type_line}{info.mana_cost && <span className="focus-cost">{info.mana_cost}</span>}
      </p>
      {info.oracle_text && <p className="focus-text">{info.oracle_text}</p>}
      <p className="focus-meta">
        {[pt, info.keywords?.length ? info.keywords.join(', ') : '', info.artist && `Illus. ${info.artist}`,
          info.released_at].filter(Boolean).join(' · ')}
      </p>
    </div>
  )
}

/** Below the masthead for a focused card: its numbers on a card that turns over into place while the similar
 * cards are dealt in alongside it. `leaving` turns them all away again together (similar cards in reverse). */
export function FocusBody({ line, backend, leaving }: { line: GridLine; backend: Backend; leaving: boolean }) {
  const r = line.row
  const fair = r ? fairOf(r, backend) : null
  const other = r ? fairOf(r, backend === 'gp' ? 'a' : 'gp') : null
  const change = r ? pctChange(r.price_today, r.pred_price) : null
  const comps = r?.comps ?? []
  const [up, setUp] = useState(false)
  useEffect(() => { // mount turned away, then turn over on the next frame so the transition runs
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setUp(true)))
    return () => cancelAnimationFrame(t)
  }, [])
  const shown = up && !leaving
  // each number is its own little card, dealt in one after another (and taken away in reverse)
  const stats: { key: string; cls?: string; label: string; sub: string; value: string; valueCls?: string; range?: string }[] = r ? [
    { key: 'fair', cls: 'lead', label: 'Fair value', sub: fair!.model, value: usd(fair!.mid), range: range(fair!.lo, fair!.hi) || '—' },
    { key: 'change', cls: 'lead-change', label: '14-day change', sub: 'TCGplayer forecast', value: change?.text ?? '—',
      valueCls: change ? (change.value > 0 ? 'up' : change.value < 0 ? 'down' : '') : '' },
    { key: 'now', label: 'Current', sub: 'TCGplayer', value: r.status === 'custom' ? '—' : usd(r.price_today) },
    { key: 'pred', label: 'In 14 days', sub: 'TCGplayer forecast', value: usd(r.pred_price), range: range(r.price_lo, r.price_hi) || '—' },
    { key: 'bl', label: 'CK buylist', sub: 'Buylist forecast', value: usd(r.buylist_pred), range: range(r.buylist_lo, r.buylist_hi) || '—' },
    { key: 'ckr', label: 'CK retail', sub: 'today', value: usd(r.ck_retail) },
    { key: 'ckb', label: 'CK buylist', sub: 'today', value: usd(r.ck_buylist) },
    { key: 'mkm', label: 'Cardmarket', sub: 'today', value: r.cardmarket_eur == null ? '—' : `€${r.cardmarket_eur.toFixed(2)}` },
    ...(other && other.model !== fair!.model ? [{ key: 'other', label: 'Fair value', sub: other.model, value: usd(other.mid) }] : []),
  ] : []
  const turn = (i: number, n: number, step: number) => ({ transitionDelay: `${(leaving ? n - 1 - i : i) * step}ms` })
  const flip = `${shown ? 'face-up' : ''} ${leaving ? 'turned-out' : ''}`
  return (
    <div className="focus-view">
      {!r ? <p className="muted">{line.error ?? 'Pricing…'}</p> : (
        <div className="summary focus-summary">
          <p className={`eyebrow fade-title ${shown ? '' : 'faded'}`}>
            {line.item.quantity > 1 ? `${line.item.quantity} copies · ` : ''}Prices as of {r.price_date}{r.signal ? ` · ${r.signal}` : ''}
          </p>
          <dl>
            {stats.map((st, i) => (
              <div key={st.key} className={`stat-slot ${st.cls ?? ''}`}>
                <div className={`stat-card ${flip}`} style={turn(i, stats.length, STAT_STEP_MS)}>
                  <dt>{st.label}<span>{st.sub}</span></dt>
                  <dd className={st.valueCls}>{st.value}</dd>
                  {st.range && <span className="focus-range">{st.range}</span>}
                </div>
              </div>
            ))}
          </dl>
          {r.caveats && r.caveats.length > 0 && (
            <ul className={`caveats fade-title ${shown ? '' : 'faded'}`}>{r.caveats.map((c) => <li key={c}>{c}</li>)}</ul>
          )}
        </div>
      )}

      <div className="list-header similar-header"><h2 className={`fade-title ${shown ? '' : 'faded'}`}>Similar cards</h2></div>
      {comps.length === 0
        ? <p className={`muted similar-empty fade-title ${shown ? '' : 'faded'}`}>No similar printings for this card.</p>
        : (
          <div className="card-grid similar-grid">
            {comps.map((c, i) => (
              <div className="slot" key={c.printing_key}>
                <div className={`similar-tile ${flip}`} style={turn(i, comps.length, SIMILAR_STEP_MS)}>
                  <img className="similar-art" src={scryfallImage(c.printing_key.split(':')[0], 'normal')}
                    alt={c.name} title={`${c.name} · ${c.set.toUpperCase()} · ${c.finish}`} draggable={false} />
                </div>
              </div>
            ))}
          </div>
        )}
    </div>
  )
}

/** Timings shared with App so it knows when the focus view has finished turning away. */
export const SIMILAR_STEP_MS = 90
export const STAT_STEP_MS = 40 // the number cards are dealt faster than the card-sized ones
export const FOCUS_FLIP_MS = 800 // matches .stat-card / .similar-tile transitions
/** Until the last number / similar card has turned edge-on (gone from view); the grid can turn back then. */
export const focusLeaveMs = (line: GridLine | null) =>
  Math.max(8 * STAT_STEP_MS, Math.max(0, (line?.row?.comps?.length ?? 0) - 1) * SIMILAR_STEP_MS) + 0.14 * FOCUS_FLIP_MS + 120
