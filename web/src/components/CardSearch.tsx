import { useEffect, useMemo, useState } from 'react'
import { search, treatmentKey, treatmentLabel, type SearchHit } from '../data'
import type { SearchIndex } from '../types'
import { Dropdown, type Option } from './Dropdown'

interface Props {
  index: SearchIndex | null
  onAdd: (hit: SearchHit) => void
  onCustom: () => void
  onCancel: () => void
}

/** Select a printing, adjust its set and treatment, then add it explicitly. */
export function CardSearch({ index, onAdd, onCustom, onCancel }: Props) {
  const [name, setName] = useState('')
  const [selectedName, setSelectedName] = useState<string | null>(null)
  const [set, setSet] = useState('')
  const [treatment, setTreatment] = useState('')
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
  const [showHits, setShowHits] = useState(true)

  const suggestions = useMemo(() => (index ? search(index, { name }) : []), [index, name])
  // The card being priced: the suggestion picked, or a name typed out in full.
  const typedName = useMemo(() => {
    const t = name.trim().toLowerCase()
    return t ? suggestions.find((hit) => hit.name.toLowerCase() === t)?.name ?? null : null
  }, [suggestions, name])
  const cardName = selectedName ?? typedName
  const nameHits = useMemo(() => {
    if (!index || !cardName) return []
    return search(index, { name: cardName }, Number.MAX_SAFE_INTEGER).filter((hit) => hit.name === cardName)
  }, [index, cardName])
  // Sets come from this card's printings only; treatments from its printings in the chosen set only.
  const setOptions = useMemo<Option[]>(() => {
    const sets = new Map(nameHits.map((hit) => [hit.set, `${hit.set_name} (${hit.set.toUpperCase()})`]))
    return [...sets].map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label))
  }, [nameHits])
  const treatmentsIn = (code: string): Option[] => {
    const found = new Map(nameHits.filter((hit) => hit.set === code)
      .map((hit) => [treatmentKey(hit.finish, hit.treatment), treatmentLabel(hit.finish, hit.treatment)]))
    return [...found].map(([value, label]) => ({ value, label })).sort((a, b) => a.label.localeCompare(b.label))
  }
  const treatmentOptions = useMemo(() => (set ? treatmentsIn(set) : []), [nameHits, set]) // eslint-disable-line react-hooks/exhaustive-deps
  const chooseSet = (code: string) => {
    const options = treatmentsIn(code)
    setSet(code)
    // keep the treatment if this set has it too; a set with only one treatment fills it in
    setTreatment((t) => (options.some((o) => o.value === t) ? t : options.length === 1 ? options[0].value : ''))
    setSelectedKey(null)
    setShowHits(true)
  }
  useEffect(() => { // a card printed in a single set: fill the set in as soon as the name is known
    if (cardName && !set && setOptions.length === 1) chooseSet(setOptions[0].value)
  }, [cardName, setOptions]) // eslint-disable-line react-hooks/exhaustive-deps

  const matching = useMemo(() => (set && treatment
    ? nameHits.filter((hit) => hit.set === set && treatmentKey(hit.finish, hit.treatment) === treatment)
    : []), [nameHits, set, treatment])
  const selected = selectedKey ? matching.find((hit) => hit.printing_key === selectedKey) : null
  const ready = selected ?? (matching.length === 1 ? matching[0] : null)
  // after a card is known, the list only lists same-treatment printings in the set (e.g. alternate collector numbers)
  const visibleHits = cardName ? matching.slice(0, 30) : suggestions

  const choose = (hit: SearchHit) => {
    setName(hit.name)
    setSelectedName(hit.name)
    setSet(hit.set)
    setTreatment(treatmentKey(hit.finish, hit.treatment))
    setSelectedKey(hit.printing_key)
    setShowHits(false)
  }

  return (
    <div className="search">
      <div className="search-picker">
        <div className="search-fields">
        <label className="field"><span>Card name, Scryfall ID or link</span>
          <input id="card-search" type="search" placeholder={index ? 'Name, ID, or Scryfall card URL' : 'Loading cards…'}
            value={name} disabled={!index}
            onFocus={() => setShowHits(true)}
            onChange={(e) => {
              setName(e.target.value)
              setSelectedName(null)
              setSet('')
              setTreatment('')
              setSelectedKey(null)
              setShowHits(true)
            }} /></label>
        <div className="field"><span>Set</span>
          <Dropdown value={set} options={setOptions} onChange={chooseSet}
            placeholder={cardName ? `${setOptions.length} sets` : 'Choose a card first'} /></div>
        <div className="field"><span>Treatment</span>
          <Dropdown value={treatment} options={treatmentOptions}
            onChange={(t) => { setTreatment(t); setSelectedKey(null); setShowHits(true) }}
            placeholder={!cardName ? 'Choose a card first' : !set ? 'Choose a set first' : `${treatmentOptions.length} treatments`} /></div>
        </div>
        {showHits && visibleHits.length > 0 && (!cardName || (matching.length > 1 && !selected)) && (
          <ul className="hits" role="listbox">
            {visibleHits.map((hit) => (
              <li key={hit.printing_key}>
                <button type="button" onClick={() => choose(hit)}>
                  <span className="hit-name">{hit.name}</span>
                  <span className="hit-meta">
                    {hit.set_name} ({hit.set.toUpperCase()}) #{hit.collector_number} · {treatmentLabel(hit.finish, hit.treatment)}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {ready && <p className="search-selection">Ready to add: {ready.name} · {ready.set_name} #{ready.collector_number} · {treatmentLabel(ready.finish, ready.treatment)}</p>}
        <div className="panel-actions">
          <button type="button" className="add-card" disabled={!ready} onClick={() => ready && onAdd(ready)}>Add</button>
          <button type="button" className="secondary" onClick={onCustom}>+ Custom card</button>
          <button type="button" className="link" onClick={onCancel}>Cancel</button>
        </div>
      </div>
    </div>
  )
}
