import type { Meta, PredictionRow, SearchIndex } from './types'

const DATA = `${import.meta.env.BASE_URL}data`

async function getJSON<T>(url: string): Promise<T> {
  const r = await fetch(url)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText} (${url})`)
  return r.json() as Promise<T>
}

export const loadMeta = () => getJSON<Meta>(`${DATA}/meta.json`)
export const loadIndex = () => getJSON<SearchIndex>(`${DATA}/index.json`)

const setCache = new Map<string, Promise<Map<string, PredictionRow>>>()
/** Prediction rows for one set, fetched once and cached. */
export function loadSet(set: string): Promise<Map<string, PredictionRow>> {
  let p = setCache.get(set)
  if (!p) {
    p = getJSON<PredictionRow[]>(`${DATA}/sets/${encodeURIComponent(set)}.json`).then(
      (rows) => new Map(rows.map((r) => [r.printing_key, r])),
    )
    setCache.set(set, p)
  }
  return p
}

const RARITY: Record<string, string> = {
  c: 'common', u: 'uncommon', r: 'rare', m: 'mythic', s: 'special', b: 'bonus',
}

export interface SearchHit {
  i: number
  printing_key: string
  name: string
  set: string
  set_name: string
  collector_number: string
  finish: string
  rarity: string
}

/** Case-insensitive search: name prefix matches first, then substring; optional "set:" / "#" filters. */
export function search(index: SearchIndex, query: string, limit = 30): SearchHit[] {
  const setFilter = query.match(/\bset:(\S+)/i)?.[1]?.toLowerCase()
  const q = query.replace(/\bset:\S+/i, '').trim().toLowerCase()
  if (q.length < 2) return []
  const prefix: number[] = []
  const contains: number[] = []
  for (let i = 0; i < index.name.length; i++) {
    if (setFilter && index.set[i] !== setFilter) continue
    const n = index.name[i].toLowerCase()
    if (n.startsWith(q)) prefix.push(i)
    else if (n.includes(q)) contains.push(i)
    if (prefix.length >= limit) break
  }
  return [...prefix, ...contains].slice(0, limit).map((i) => ({
    i,
    printing_key: index.printing_key[i],
    name: index.name[i],
    set: index.set[i],
    set_name: index.sets[index.set[i]] ?? index.set[i],
    collector_number: index.collector_number[i],
    finish: index.finish[i],
    rarity: RARITY[index.rarity[i]] ?? index.rarity[i],
  }))
}

export function scryfallImage(scryfallId: string | undefined, size: 'small' | 'normal' = 'small'): string | undefined {
  if (!scryfallId || scryfallId.startsWith('custom')) return undefined
  return `https://cards.scryfall.io/${size}/front/${scryfallId[0]}/${scryfallId[1]}/${scryfallId}.jpg`
}

export const usd = (v: number | null | undefined) =>
  v == null ? '—' : v >= 100 ? `$${Math.round(v).toLocaleString()}` : `$${v.toFixed(2)}`
