import type { CustomCardInput, ListItem, SearchIndex } from './types'

/** RFC 4180-ish CSV parsing: quoted fields, doubled quotes, commas/newlines inside quotes. */
export function parseCSV(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const c = text[i]
    if (quoted) {
      if (c === '"' && text[i + 1] === '"') { field += '"'; i++ }
      else if (c === '"') quoted = false
      else field += c
    } else if (c === '"') quoted = true
    else if (c === ',') { row.push(field); field = '' }
    else if (c === '\n' || c === '\r') {
      if (c === '\r' && text[i + 1] === '\n') i++
      row.push(field); field = ''
      if (row.some((f) => f.trim() !== '')) rows.push(row)
      row = []
    } else field += c
  }
  row.push(field)
  if (row.some((f) => f.trim() !== '')) rows.push(row)
  return rows
}

type NewItem = ListItem extends infer T ? T extends ListItem ? Omit<T, 'id'> : never : never
export interface ImportResult { items: NewItem[]; unmatched: string[] }

/**
 * Turn a CSV into list items. Reads this app's own export (printing_key / custom columns) and plain
 * card lists: `name` is required; `set`, `collector_number`, `finish` and `quantity` narrow or fill it in.
 */
export function importCSV(text: string, index: SearchIndex): ImportResult {
  const [header, ...rows] = parseCSV(text.replace(/^﻿/, ''))
  if (!header) return { items: [], unmatched: [] }
  const col = (...names: string[]) => header.findIndex((h) => names.includes(h.trim().toLowerCase()))
  const c = {
    key: col('printing_key'), custom: col('custom'), name: col('name', 'card', 'card name'),
    set: col('set', 'set code', 'edition'), num: col('collector_number', 'collector number', 'number'),
    finish: col('finish', 'foil'), qty: col('quantity', 'qty', 'count'),
  }
  if (c.name < 0 && c.key < 0 && c.custom < 0) throw new Error('No "name" column found')

  const byKey = new Map<string, number>()
  const byName = new Map<string, number[]>()
  index.printing_key.forEach((k, i) => {
    byKey.set(k, i)
    const n = index.name[i].toLowerCase()
    const list = byName.get(n)
    if (list) list.push(i)
    else byName.set(n, [i])
  })

  const items: ImportResult['items'] = []
  const unmatched: string[] = []
  for (const r of rows) {
    const get = (j: number) => (j >= 0 ? (r[j] ?? '').trim() : '')
    const quantity = Math.max(1, Math.round(Number(get(c.qty)) || 1))
    const custom = get(c.custom)
    if (custom) {
      try {
        items.push({ kind: 'custom', card: JSON.parse(custom) as CustomCardInput, quantity })
      } catch {
        unmatched.push(get(c.name) || 'custom card')
      }
      continue
    }
    let i = byKey.get(get(c.key))
    if (i == null) {
      const name = get(c.name).toLowerCase()
      const set = get(c.set).toLowerCase()
      const num = get(c.num).toLowerCase()
      const finishRaw = get(c.finish).toLowerCase()
      const finish = finishRaw === 'true' || finishRaw === 'yes' ? 'foil' : finishRaw === 'false' || finishRaw === 'no' ? 'nonfoil' : finishRaw
      const candidates = (byName.get(name) ?? []).filter((k) =>
        (!set || index.set[k] === set) && (!num || index.collector_number[k].toLowerCase() === num))
      // Prefer the asked-for finish, then nonfoil.
      i = candidates.find((k) => (finish ? index.finish[k] === finish : index.finish[k] === 'nonfoil')) ?? candidates[0]
    }
    if (i == null) {
      unmatched.push(get(c.name) || get(c.key) || '(blank)')
      continue
    }
    items.push({ kind: 'existing', printing_key: index.printing_key[i], set: index.set[i], name: index.name[i], quantity })
  }
  return { items, unmatched }
}
