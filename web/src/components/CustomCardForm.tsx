import { useState } from 'react'
import type { Backend, CustomCardInput, Demand } from '../types'

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
  backend: Backend
  onAdd: (card: CustomCardInput, quantity: number) => void
  onCancel: () => void
}

/** Everything printed on a card or declared at design time (docs/APP_DESIGN.md §2). */
export function CustomCardForm({ backend, onAdd, onCancel }: Props) {
  const [card, setCard] = useState<CustomCardInput>(EMPTY)
  const [quantity, setQuantity] = useState(1)
  const set = <K extends keyof CustomCardInput>(k: K, v: CustomCardInput[K]) => setCard((c) => ({ ...c, [k]: v }))
  const valid = card.name.trim() && card.type_line.trim()

  const readArt = (file: File | undefined) => {
    if (!file) return set('art', undefined)
    const reader = new FileReader()
    reader.onload = () => set('art', reader.result as string)
    reader.readAsDataURL(file)
  }

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
        <label className="field"><span>Rarity</span>
          <select value={card.rarity} onChange={(e) => set('rarity', e.target.value as CustomCardInput['rarity'])}>
            {['common', 'uncommon', 'rare', 'mythic', 'special', 'bonus'].map((r) => <option key={r}>{r}</option>)}
          </select></label>
        <label className="field"><span>Finish</span>
          <select value={card.finish} onChange={(e) => set('finish', e.target.value as CustomCardInput['finish'])}>
            {['nonfoil', 'foil', 'etched'].map((f) => <option key={f}>{f}</option>)}
          </select></label>
        <label className="field"><span>Set type</span>
          <select value={card.set_type} onChange={(e) => set('set_type', e.target.value)}>
            {['expansion', 'core', 'masters', 'commander', 'draft_innovation', 'funny', 'promo'].map((s) => <option key={s}>{s}</option>)}
          </select></label>
        <label className="field"><span>Border</span>
          <select value={card.border_color} onChange={(e) => set('border_color', e.target.value)}>
            {['black', 'borderless', 'white', 'silver', 'gold'].map((b) => <option key={b}>{b}</option>)}
          </select></label>
        <label className="field"><span>Frame</span>
          <select value={card.frame} onChange={(e) => set('frame', e.target.value)}>
            {['2015', '1997', '1993', 'future'].map((f) => <option key={f}>{f}</option>)}
          </select></label>
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
        <label className="field wide"><span>Expected demand</span>
          <select value={card.demand} onChange={(e) => set('demand', e.target.value as Demand)}>
            {DEMANDS.map((d) => <option key={d.value} value={d.value}>{d.label}</option>)}
          </select>
          <small>Stands in for EDHREC rank, the strongest price signal. Ignored if "Reprint of" is set.</small></label>
        <label className="field"><span>Reprint of</span>
          <input placeholder="exact card name" value={card.reprint_of ?? ''} onChange={(e) => set('reprint_of', e.target.value || undefined)} /></label>
        <label className="field"><span>In set (code)</span>
          <input placeholder="optional, e.g. fdn" value={card.set ?? ''} onChange={(e) => set('set', e.target.value || undefined)} /></label>
        {backend === 'gp' && (
          <label className="field wide"><span>Art (deep GP only)</span>
            <input type="file" accept="image/*" onChange={(e) => readArt(e.target.files?.[0])} /></label>
        )}
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
