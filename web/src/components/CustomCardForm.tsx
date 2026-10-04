import { useMemo, useState } from 'react'
import { search } from '../data'
import type { CustomCardInput, Demand, SearchIndex } from '../types'
import { Dropdown } from './Dropdown'

const DEMANDS: { value: Demand; label: string }[] = [
  { value: 'staple', label: 'Staple (top ~2% of Commander play)' },
  { value: 'popular', label: 'Popular (top ~5%)' },
  { value: 'playable', label: 'Playable (top ~25%)' },
  { value: 'niche', label: 'Niche (~median)' },
  { value: 'unplayed', label: 'Unplayed' },
]
const FRAME_EFFECTS = ['showcase', 'extendedart', 'legendary', 'etched', 'inverted', 'fullart']

const EMPTY: CustomCardInput = {
  name: '', rarity: 'rare', type_line: '', mana_cost: '', oracle_text: '', finish: 'nonfoil',
  set_type: 'expansion', frame: '2015', border_color: 'black', frame_effects: [], demand: 'playable',
}

interface Props {
  index: SearchIndex | null
  onAdd: (card: CustomCardInput, quantity: number) => void
  onCancel: () => void
}

/** Everything printed on a card or declared at design time (docs/APP_DESIGN.md §2). */
export function CustomCardForm({ index, onAdd, onCancel }: Props) {
  const [card, setCard] = useState<CustomCardInput>(EMPTY)
  const [quantity, setQuantity] = useState(1)
  const set = <K extends keyof CustomCardInput>(k: K, v: CustomCardInput[K]) => setCard((c) => ({ ...c, [k]: v }))
  const valid = card.name.trim() && card.type_line.trim()

  // "Reprint of" must be an exact card name: suggest distinct names from the search index.
  const [reprintQuery, setReprintQuery] = useState('')
  const reprintHits = useMemo(() => {
    if (!index || reprintQuery === card.reprint_of) return []
    return [...new Set(search(index, { name: reprintQuery }, 40).map((h) => h.name))].slice(0, 8)
  }, [index, reprintQuery, card.reprint_of])

  return (
    <form
      className="custom-form"
      onSubmit={(e) => {
        e.preventDefault()
        if (valid) onAdd({ ...card, name: card.name.trim() }, quantity)
      }}
    >
      <h2>Custom card</h2>
      <div className="grid">
        <label className="field wide"><span>Name *</span>
          <input required value={card.name} onChange={(e) => set('name', e.target.value)} /></label>
        <label className="field"><span>Mana cost</span>
          <input placeholder="{2}{U}{U}" value={card.mana_cost} onChange={(e) => set('mana_cost', e.target.value)} /></label>
        <label className="field wide"><span>Type line *</span>
          <input required placeholder="Legendary Creature — Human Wizard" value={card.type_line}
            onChange={(e) => set('type_line', e.target.value)} /></label>
        <label className="field full"><span>Rules text</span>
          <textarea rows={3} value={card.oracle_text} onChange={(e) => set('oracle_text', e.target.value)} /></label>
        <label className="field"><span>Power</span>
          <input value={card.power ?? ''} onChange={(e) => set('power', e.target.value || undefined)} /></label>
        <label className="field"><span>Toughness</span>
          <input value={card.toughness ?? ''} onChange={(e) => set('toughness', e.target.value || undefined)} /></label>
        <label className="field"><span>Loyalty</span>
          <input value={card.loyalty ?? ''} onChange={(e) => set('loyalty', e.target.value || undefined)} /></label>
        <div className="field"><span>Rarity</span>
          <Dropdown value={card.rarity} options={['common', 'uncommon', 'rare', 'mythic', 'special', 'bonus'].map((v) => ({ value: v, label: v }))}
            onChange={(v) => set('rarity', v as CustomCardInput['rarity'])} /></div>
        <div className="field"><span>Finish</span>
          <Dropdown value={card.finish} options={['nonfoil', 'foil', 'etched'].map((v) => ({ value: v, label: v }))}
            onChange={(v) => set('finish', v as CustomCardInput['finish'])} /></div>
        <div className="field"><span>Set type</span>
          <Dropdown value={card.set_type} options={['expansion', 'core', 'masters', 'commander', 'draft_innovation', 'funny', 'promo'].map((v) => ({ value: v, label: v }))}
            onChange={(v) => set('set_type', v)} /></div>
        <div className="field"><span>Border</span>
          <Dropdown value={card.border_color} options={['black', 'borderless', 'white', 'silver', 'gold'].map((v) => ({ value: v, label: v }))}
            onChange={(v) => set('border_color', v)} /></div>
        <div className="field"><span>Frame</span>
          <Dropdown value={card.frame} options={['2015', '1997', '1993', 'future'].map((v) => ({ value: v, label: v }))}
            onChange={(v) => set('frame', v)} /></div>
        <label className="field"><span>Release date</span>
          <input type="date" value={card.released_at ?? ''} onChange={(e) => set('released_at', e.target.value || undefined)} /></label>
        <fieldset className="field full"><legend>Frame effects</legend>
          <div className="chips">
            {FRAME_EFFECTS.map((fx) => (
              <label key={fx} className="chip">
                <input type="checkbox" checked={card.frame_effects.includes(fx)}
                  onChange={(e) => set('frame_effects', e.target.checked
                    ? [...card.frame_effects, fx] : card.frame_effects.filter((x) => x !== fx))} />
                {fx}
              </label>
            ))}
          </div>
        </fieldset>
        <div className="field wide"><span>Expected demand</span>
          <Dropdown value={card.demand} options={DEMANDS} onChange={(v) => set('demand', v as Demand)} />
          <small>Stands in for EDHREC rank, the strongest price signal. Ignored if "Reprint of" is set.</small></div>
        <div className="field search"><span>Reprint of</span>
          <input placeholder="search a real card" value={reprintQuery}
            onChange={(e) => { setReprintQuery(e.target.value); set('reprint_of', undefined) }} />
          {reprintHits.length > 0 && (
            <ul className="hits">
              {reprintHits.map((name) => (
                <li key={name}><button type="button" onClick={() => { set('reprint_of', name); setReprintQuery(name) }}>{name}</button></li>
              ))}
            </ul>
          )}
          {card.reprint_of && <small>Uses {card.reprint_of}'s demand, tags, legality and printing history.</small>}
        </div>
        <label className="field"><span>In set (code)</span>
          <input placeholder="optional, e.g. fdn" value={card.set ?? ''} onChange={(e) => set('set', e.target.value || undefined)} /></label>
        <label className="field"><span>Quantity</span>
          <input type="number" min={1} value={quantity} onChange={(e) => setQuantity(Math.max(1, Number(e.target.value) || 1))} /></label>
      </div>
      <div className="actions">
        <button type="button" className="secondary" onClick={onCancel}>Cancel</button>
        <button type="submit" disabled={!valid}>Add to list</button>
      </div>
    </form>
  )
}
