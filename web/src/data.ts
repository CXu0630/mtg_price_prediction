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
  treatment: string
  rarity: string
}

export interface SearchFilters {
  name?: string
  set?: string
  treatment?: string
}

export const treatmentKey = (finish: string, visual: string) => JSON.stringify([finish, visual])
export const treatmentLabel = (finish: string, visual: string) => visual ? `${finish} · ${visual}` : finish

const UUID = /^[0-9a-f]{8}-(?:[0-9a-f]{4}-){3}[0-9a-f]{12}$/i

type ScryfallLookup = { id: string } | { set: string; collector: string }

function scryfallLookup(input: string): ScryfallLookup | null {
  const value = input.trim()
  if (UUID.test(value)) return { id: value.toLowerCase() }
  const urlText = /^https?:\/\//i.test(value) ? value
    : /^(?:www\.)?(?:scryfall\.com|api\.scryfall\.com|cards\.scryfall\.io)\//i.test(value) ? `https://${value}` : null
  if (!urlText) return null
  try {
    const url = new URL(urlText)
    const host = url.hostname.toLowerCase()
    if (!['scryfall.com', 'www.scryfall.com', 'api.scryfall.com', 'cards.scryfall.io'].includes(host)) return null
    const parts = url.pathname.split('/').filter(Boolean).map(decodeURIComponent)
    const id = parts.map((p) => p.replace(/\.(?:jpg|png|webp)$/i, '')).find((p) => UUID.test(p))
    if (id) return { id: id.toLowerCase() }
    if ((host === 'scryfall.com' || host === 'www.scryfall.com') && parts[0] === 'card' && parts.length >= 3) {
      return { set: parts[1].toLowerCase(), collector: parts[2].toLowerCase() }
    }
  } catch {
    return null
  }
  return null
}

/** Case-insensitive search across name/Scryfall reference, set and treatment. Name prefixes rank first. */
export function search(index: SearchIndex, filters: SearchFilters, limit = 30): SearchHit[] {
  const lookup = scryfallLookup(filters.name ?? '')
  const name = lookup ? '' : filters.name?.trim().toLowerCase() ?? ''
  const set = filters.set?.trim().toLowerCase() ?? ''
  const treatment = (filters.treatment ?? '').toLowerCase().replace(/[\s_-]+/g, '')
  if (!lookup && ((!name && !set && !treatment) || (name.length < 2 && !set && !treatment))) return []
  const matchingSets = set
    ? new Set(Object.entries(index.sets).filter(([code, label]) =>
      code.toLowerCase().includes(set) || label.toLowerCase().includes(set)).map(([code]) => code))
    : null
  const prefix: number[] = []
  const contains: number[] = []
  for (let i = 0; i < index.name.length; i++) {
    if (lookup && ('id' in lookup
      ? !index.printing_key[i].toLowerCase().startsWith(`${lookup.id}:`)
      : index.set[i].toLowerCase() !== lookup.set || index.collector_number[i].toLowerCase() !== lookup.collector)) continue
    if (matchingSets && !matchingSets.has(index.set[i]) && !index.set[i].toLowerCase().includes(set)) continue
    if (treatment) {
      const finish = index.finish[i].toLowerCase().replace(/[\s_-]+/g, '')
      const art = (index.treatment?.[i] ?? '').toLowerCase().replace(/[\s_-]+/g, '')
      if (!(treatment === 'foil'
        ? finish !== 'nonfoil' && finish.includes('foil')
        : finish.includes(treatment) || art.includes(treatment))) continue
    }
    const n = index.name[i].toLowerCase()
    if (lookup || !name || n.startsWith(name)) prefix.push(i)
    else if (n.includes(name)) contains.push(i)
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
    treatment: index.treatment?.[i] ?? '',
    rarity: RARITY[index.rarity[i]] ?? index.rarity[i],
  }))
}

export function scryfallImage(scryfallId: string | undefined, size: 'small' | 'normal' = 'small'): string | undefined {
  if (!scryfallId || scryfallId.startsWith('custom')) return undefined
  return `https://cards.scryfall.io/${size}/front/${scryfallId[0]}/${scryfallId[1]}/${scryfallId}.jpg`
}

/** Percentage change from `from` to `to`, signed and rounded (e.g. "+3.2%"); null when either is missing. */
export const pctChange = (from: number | null | undefined, to: number | null | undefined) => {
  if (from == null || to == null || from <= 0) return null
  const p = ((to - from) / from) * 100
  return { value: p, text: `${p > 0 ? '+' : p < 0 ? '−' : '±'}${Math.abs(p).toFixed(Math.abs(p) < 10 ? 1 : 0)}%` }
}

export const usd = (v: number | null | undefined) =>
  v == null ? '—' : v >= 100 ? `$${Math.round(v).toLocaleString()}` : `$${v.toFixed(2)}`
