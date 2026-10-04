import { useMemo, useState } from 'react'
import { search, type SearchHit } from '../data'
import type { SearchIndex } from '../types'

interface Props {
  index: SearchIndex | null
  onAdd: (hit: SearchHit) => void
}

/** Search existing printings by name (optionally "set:abc") and add them to the list. */
export function CardSearch({ index, onAdd }: Props) {
  const [query, setQuery] = useState('')
  const hits = useMemo(() => (index ? search(index, query) : []), [index, query])

  return (
    <div className="search">
      <label className="field">
        <span>Add an existing card</span>
        <input id="card-search"
          type="search"
          placeholder={index ? 'Card name, e.g. "Sol Ring" or "bolt set:2x2"' : 'Loading card index…'}
          value={query}
          disabled={!index}
          onChange={(e) => setQuery(e.target.value)}
        />
      </label>
      {hits.length > 0 && (
        <ul className="hits" role="listbox">
          {hits.map((h) => (
            <li key={h.printing_key}>
              <button
                type="button"
                onClick={() => {
                  onAdd(h)
                  setQuery('')
                }}
              >
                <span className="hit-name">{h.name}</span>
                <span className="hit-meta">
                  {h.set_name} ({h.set.toUpperCase()}) #{h.collector_number} · {h.finish} · {h.rarity}
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
