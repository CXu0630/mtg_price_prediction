import { useCallback, useEffect, useMemo, useState } from 'react'
import { CardSearch } from './components/CardSearch'
import { CustomCardForm } from './components/CustomCardForm'
import { ResultsTable, type ResultLine } from './components/ResultsTable'
import { loadIndex, loadMeta, loadSet, usd } from './data'
import { loadModels, priceCustom } from './model'
import type { Backend, CustomCardInput, ListItem, Meta, PredictionRow, SearchIndex } from './types'

const STORAGE_KEY = 'mtg-price-list-v1'

function loadList(): ListItem[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? (JSON.parse(raw) as ListItem[]) : []
  } catch {
    return []
  }
}

function saveList(items: ListItem[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(items))
  } catch {
    /* storage unavailable: the list just won't persist */
  }
}

const newId = () => Math.random().toString(36).slice(2, 10)

function toCSV(lines: ResultLine[], backend: Backend): string {
  const cols = ['name', 'set', 'collector_number', 'finish', 'quantity', 'status', 'price_today', 'fair', 'fair_lo',
    'fair_hi', 'pred_price', 'price_lo', 'price_hi', 'signal', 'buylist_pred', 'buylist_lo', 'buylist_hi', 'price_date']
  const esc = (v: unknown) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  const rows = lines.filter((l) => l.row).map((l) => {
    const r = l.row as PredictionRow
    const fair = backend === 'gp' && r.fair_gp != null
      ? { fair: r.fair_gp, fair_lo: r.fair_gp_lo, fair_hi: r.fair_gp_hi } : { fair: r.fair, fair_lo: r.fair_lo, fair_hi: r.fair_hi }
    const rec: Record<string, unknown> = { ...r, ...fair, quantity: l.quantity }
    return cols.map((c) => esc(rec[c])).join(',')
  })
  return [cols.join(','), ...rows].join('\n')
}

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [index, setIndex] = useState<SearchIndex | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [backend, setBackend] = useState<Backend>('a')
  const [items, setItems] = useState<ListItem[]>(loadList)
  const [existingRows, setExistingRows] = useState<Record<string, PredictionRow>>({})
  const [customRows, setCustomRows] = useState<Record<string, PredictionRow | { error: string }>>({})
  const [pricing, setPricing] = useState(false)
  const [showForm, setShowForm] = useState(false)

  useEffect(() => {
    Promise.all([loadMeta(), loadIndex()])
      .then(([m, i]) => { setMeta(m); setIndex(i) })
      .catch((e: Error) => setLoadError(e.message))
  }, [])

  useEffect(() => saveList(items), [items])

  // Existing cards: rows come straight from the static per-set files.
  useEffect(() => {
    const missing = items.filter((i) => i.kind === 'existing' && !existingRows[i.printing_key])
    if (missing.length === 0) return
    const sets = [...new Set(missing.map((i) => (i.kind === 'existing' ? i.set : '')))]
    Promise.all(sets.map((s) => loadSet(s))).then((maps) => {
      const found: Record<string, PredictionRow> = {}
      for (const m of maps) for (const i of missing) {
        if (i.kind === 'existing' && m.has(i.printing_key)) found[i.printing_key] = m.get(i.printing_key) as PredictionRow
      }
      setExistingRows((prev) => ({ ...prev, ...found }))
    }).catch((e: Error) => setLoadError(e.message))
  }, [items, existingRows])

  // Custom cards: priced in the browser by Models A and B (exported nightly; src/model/).
  const priceCustomCards = useCallback(async () => {
    const custom = items.filter((i): i is Extract<ListItem, { kind: 'custom' }> => i.kind === 'custom')
    if (custom.length === 0) return
    setPricing(true)
    try {
      const models = await loadModels(custom.some((c) => c.card.reprint_of))
      const rows = priceCustom(custom.map((c) => c.card), models)
      setCustomRows(Object.fromEntries(custom.map((c, k) => [c.id, rows[k]])))
    } catch (e) {
      const msg = (e as Error).message
      setCustomRows(Object.fromEntries(custom.map((c) => [c.id, { error: msg }])))
    } finally {
      setPricing(false)
    }
  }, [items])

  const customKey = JSON.stringify(items.filter((i) => i.kind === 'custom').map((i) => i.id))
  useEffect(() => { void priceCustomCards() }, [customKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const lines: ResultLine[] = useMemo(() => items.map((i) => {
    if (i.kind === 'existing') return { id: i.id, quantity: i.quantity, row: existingRows[i.printing_key] ?? null, pending: true }
    const r = customRows[i.id]
    if (!r) return { id: i.id, quantity: i.quantity, row: null, pending: true }
    if ('error' in r) return { id: i.id, quantity: i.quantity, row: null, error: r.error }
    return { id: i.id, quantity: i.quantity, row: r }
  }), [items, existingRows, customRows, pricing])

  const download = () => {
    const blob = new Blob([toCSV(lines, backend)], { type: 'text/csv' })
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'mtg-price-list.csv' })
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>MTG Price Range</h1>
          <p className="muted">
            Fair value, 14-day TCGplayer forecast and Card Kingdom buylist forecast for any list of printings —
            real or custom.{meta && ` Prices as of ${meta.price_date}.`}
          </p>
        </div>
        <fieldset className="toggle">
          <legend>Fair-value model</legend>
          <label><input type="radio" name="backend" checked={backend === 'a'} onChange={() => setBackend('a')} /> Model A</label>
          <label><input type="radio" name="backend" checked={backend === 'gp'} onChange={() => setBackend('gp')} /> Deep GP (text + art; existing cards)</label>
        </fieldset>
      </header>

      {loadError && <p className="error" role="alert">Couldn't load price data: {loadError}</p>}

      <section className="editor">
        <CardSearch index={index} onAdd={(h) => setItems((it) => [...it,
          { id: newId(), kind: 'existing', printing_key: h.printing_key, set: h.set, quantity: 1 }])} />
        <button type="button" onClick={() => setShowForm(true)}>+ Custom card</button>
      </section>

      {showForm && (
        <div className="modal" role="dialog" aria-modal="true" aria-label="Custom card">
          <CustomCardForm index={index} onCancel={() => setShowForm(false)}
            onAdd={(card: CustomCardInput, quantity: number) => {
              setItems((it) => [...it, { id: newId(), kind: 'custom', card, quantity }])
              setShowForm(false)
            }} />
        </div>
      )}

      <section>
        <div className="list-header">
          <h2>Your list</h2>
          <div className="actions">
            {pricing && <span className="muted">Pricing custom cards…</span>}
            <button type="button" className="secondary" disabled={items.length === 0} onClick={download}>Download CSV</button>
            <button type="button" className="secondary" disabled={items.length === 0} onClick={() => setItems([])}>Clear</button>
          </div>
        </div>
        <ResultsTable lines={lines} backend={backend}
          onQuantity={(id, q) => setItems((it) => it.map((i) => (i.id === id ? { ...i, quantity: q } : i)))}
          onRemove={(id) => setItems((it) => it.filter((i) => i.id !== id))} />
      </section>

      <footer className="muted">
        {meta && <>
          Model A: out-of-fold error ×{Math.exp(meta.model_a.oof_log_mae).toFixed(2)} typical, {Math.round(meta.model_a.oof_coverage80 * 100)}% of
          prices inside its 80% range. Models B/C trained through {meta.model_b.cutoff}. {meta.n_scored.toLocaleString()} printings
          tradeable today ({usd(0.5)}+).{' '}
        </>}
        Custom cards are priced in your browser by Models A and B (the deep GP covers existing printings only). Ranges are 80% intervals. Not financial advice. Magic: The Gathering is a trademark of
        Wizards of the Coast; card images © Wizards of the Coast, via Scryfall. Unofficial fan project.
      </footer>
    </div>
  )
}
