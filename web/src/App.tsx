import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Detail } from './components/CardDetail'
import { CardGrid, fairOf, type GridLine } from './components/CardGrid'
import { CardSearch } from './components/CardSearch'
import { FlipPanel } from './components/FlipPanel'
import { importCSV } from './csv'
import { CardShowcase, FEATURED, TREASURE, type Face } from './components/CardShowcase'
import { CustomCardForm } from './components/CustomCardForm'
import { loadIndex, loadMeta, loadSet, scryfallImage, usd } from './data'
import { loadModels, priceCustom } from './model'
import type { Backend, CustomCardInput, ListItem, Meta, PredictionRow, SearchIndex } from './types'

const STORAGE_KEY = 'mtg-price-list-v1'
const FRESH_MS = 1600 // how long a just-added card counts as "flipping in"
const LEAVE_MS = 650 // the removed card turns away and fades before the grid closes up
const WAVE_STEP_MS = 90 // matches the grid's diagonal wave
const FLIP_MS = 750 // matches .tile-inner's transition

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

function toCSV(lines: GridLine[], backend: Backend): string {
  // printing_key / custom make the file re-importable.
  const cols = ['name', 'set', 'collector_number', 'finish', 'quantity', 'status', 'price_today', 'fair', 'fair_lo',
    'fair_hi', 'pred_price', 'price_lo', 'price_hi', 'signal', 'buylist_pred', 'buylist_lo', 'buylist_hi', 'price_date',
    'printing_key', 'custom']
  const esc = (v: unknown) => (v == null ? '' : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v))
  const rows = lines.map(({ item, row }) => {
    const f = row ? fairOf(row, backend) : null
    const rec: Record<string, unknown> = {
      ...row, fair: f?.mid, fair_lo: f?.lo, fair_hi: f?.hi, quantity: item.quantity,
      ...(item.kind === 'existing'
        ? { printing_key: item.printing_key, name: row?.name ?? item.name, set: row?.set ?? item.set }
        : { custom: JSON.stringify(item.card), name: item.card.name, finish: item.card.finish }),
    }
    return cols.map((c) => esc(rec[c])).join(',')
  })
  return [cols.join(','), ...rows].join('\n')
}

type Panel = 'title' | 'summary' | 'search' | 'custom'

export default function App() {
  const [meta, setMeta] = useState<Meta | null>(null)
  const [index, setIndex] = useState<SearchIndex | null>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const [backend, setBackend] = useState<Backend>('a')
  const [items, setItems] = useState<ListItem[]>(loadList)
  const [existingRows, setExistingRows] = useState<Record<string, PredictionRow>>({})
  const [customRows, setCustomRows] = useState<Record<string, PredictionRow | { error: string }>>({})
  const [mode, setMode] = useState<'cards' | 'prices'>('cards')
  const [fresh, setFresh] = useState<Set<string>>(new Set())
  const [leaving, setLeaving] = useState<Set<string>>(new Set())
  const [clearing, setClearing] = useState(false)
  const [cleared, setCleared] = useState(false)
  const [detail, setDetail] = useState<PredictionRow | null>(null)
  const [panel, setPanel] = useState<Panel>('title')
  const [burst, setBurst] = useState<{ id: number; face?: Face } | null>(null)
  const freshTimers = useRef<number[]>([])

  useEffect(() => {
    Promise.all([loadMeta(), loadIndex()])
      .then(([m, i]) => { setMeta(m); setIndex(i) })
      .catch((e: Error) => setLoadError(e.message))
    const timers = freshTimers.current
    return () => timers.forEach(clearTimeout)
  }, [])

  useEffect(() => saveList(items), [items])

  // Prices load quietly in the background; they're only revealed by "Predict".
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
    try {
      const models = await loadModels(custom.some((c) => c.card.reprint_of))
      const rows = priceCustom(custom.map((c) => c.card), models)
      setCustomRows(Object.fromEntries(custom.map((c, k) => [c.id, rows[k]])))
    } catch (e) {
      const msg = (e as Error).message
      setCustomRows(Object.fromEntries(custom.map((c) => [c.id, { error: msg }])))
    }
  }, [items])
  const customKey = JSON.stringify(items.filter((i) => i.kind === 'custom').map((i) => i.id))
  useEffect(() => { void priceCustomCards() }, [customKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const lines: GridLine[] = useMemo(() => items.map((item) => {
    if (item.kind === 'existing') return { item, row: existingRows[item.printing_key] ?? null }
    const r = customRows[item.id]
    if (r && 'error' in r) return { item, row: null, error: r.error }
    return { item, row: r ?? null }
  }), [items, existingRows, customRows])

  const faceOf = (i: ListItem): Face => (i.kind === 'existing'
    ? { key: i.id, src: scryfallImage(i.printing_key.split(':')[0], 'normal') }
    : { key: i.id, custom: i.card })
  const spin = (face?: Face) => setBurst((b) => ({ id: (b?.id ?? 0) + 1, face }))

  // The masthead turns over between the title, the card search and the custom-card form;
  // every turn also spins the showcase card.
  const turnTo = (p: Panel) => {
    if (p === panel) return
    spin()
    setPanel(p)
    if (p === 'search') {
      if (window.scrollY > 200) window.scrollTo({ top: 0, behavior: 'smooth' })
      window.setTimeout(() => document.getElementById('card-search')?.focus({ preventScroll: true }), 450)
    }
  }
  const openEditor = () => turnTo('search')
  // Where the masthead rests: the title, or the price summary while predictions are showing.
  const home: Panel = mode === 'prices' ? 'summary' : 'title'
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !detail) setPanel((p) => { if (p !== home) spin(); return home })
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [detail, home])

  const addMany = (added: ListItem[]) => {
    if (!added.length) return
    setCleared(false)
    setMode('cards') // new cards are shown face-up; Predict again to price the whole list
    setPanel('title')
    spin(faceOf(added[0]))
    setItems((it) => [...it, ...added])
    const ids = added.map((i) => i.id)
    setFresh((f) => new Set([...f, ...ids]))
    freshTimers.current.push(window.setTimeout(() => setFresh((f) => {
      const n = new Set(f)
      for (const id of ids) n.delete(id)
      return n
    }), FRESH_MS))
  }
  const add = (item: ListItem) => addMany([item])

  const fileInput = useRef<HTMLInputElement>(null)
  const [importNote, setImportNote] = useState<string | null>(null)
  const importFile = async (file: File) => {
    if (!index) return
    try {
      const { items: found, unmatched } = importCSV(await file.text(), index)
      addMany(found.map((i) => ({ ...i, id: newId() }) as ListItem))
      setImportNote(unmatched.length
        ? `Imported ${found.length}; couldn't match ${unmatched.length}: ${unmatched.slice(0, 5).join(', ')}${unmatched.length > 5 ? '…' : ''}`
        : null)
    } catch (e) {
      setImportNote(`Couldn't read ${file.name}: ${(e as Error).message}`)
    }
  }

  // The showcase cycles through the list (custom cards as proxies); while prices are up it holds the Treasure.
  const deckKey = mode + items.map((i) => i.id).join(',')
  const deck = useMemo<Face[]>(() => (mode === 'prices' ? [TREASURE] : items.map(faceOf)),
    [deckKey]) // eslint-disable-line react-hooks/exhaustive-deps

  const remove = (id: string) => {
    if (leaving.has(id)) return
    // The showcase spins onto whatever takes the removed card's place.
    const at = items.findIndex((i) => i.id === id)
    const rest = items.filter((i) => i.id !== id && !leaving.has(i.id))
    spin(mode === 'prices' ? TREASURE : rest.length ? faceOf(rest[Math.min(at, rest.length - 1)]) : FEATURED[0])
    setLeaving((l) => new Set(l).add(id))
    freshTimers.current.push(window.setTimeout(() => {
      setItems((it) => it.filter((i) => i.id !== id))
      setLeaving((l) => {
        const n = new Set(l)
        n.delete(id)
        return n
      })
    }, LEAVE_MS))
  }

  const clearList = () => {
    if (clearing || items.length === 0) return
    // Every card turns over in the wave: the first becomes the add slot, the rest turn to nothing.
    spin(FEATURED[0])
    setPanel('title')
    setClearing(true)
    freshTimers.current.push(window.setTimeout(() => {
      setItems([])
      setMode('cards')
      setCleared(true)
      setClearing(false)
    }, FLIP_MS + (items.length - 1) * WAVE_STEP_MS + 50))
  }

  const togglePredict = () => {
    // The fast spin doubles as the "working on it" signal; the masthead turns over to the totals.
    spin(mode === 'cards' ? TREASURE : undefined)
    setPanel(mode === 'cards' ? 'summary' : 'title')
    setMode((m) => (m === 'cards' ? 'prices' : 'cards'))
  }

  const total = (f: (r: PredictionRow) => number | null | undefined) =>
    lines.reduce((acc, l) => acc + (l.row && f(l.row) != null ? (f(l.row) as number) * l.item.quantity : 0), 0)

  const download = () => {
    const blob = new Blob([toCSV(lines, backend)], { type: 'text/csv' })
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'mtg-price-list.csv' })
    a.click()
    URL.revokeObjectURL(a.href)
  }

  return (
    <div className="shell">
      <a className="brand" href="./" aria-label="MTG Price Intelligence — home">
        <img src="./logo.svg" alt="" width={30} height={30} />
        <span>MTG Price Intelligence</span>
      </a>
      <CardShowcase deck={deck} burst={burst} />
      <main className="content">
        <header className="masthead">
          <FlipPanel view={panel} rest="title" boxed={(v) => v === 'search' || v === 'custom'} render={(v) => {
            if (v === 'search') return (
              <div className="panel-search">
                <p className="eyebrow">Add a card</p>
                <CardSearch index={index} onAdd={(h) =>
                  add({ id: newId(), kind: 'existing', printing_key: h.printing_key, set: h.set, name: h.name, quantity: 1 })} />
                <div className="panel-actions">
                  <button type="button" className="secondary" onClick={() => turnTo('custom')}>+ Custom card</button>
                  <button type="button" className="link" onClick={() => turnTo(home)}>Cancel</button>
                </div>
              </div>
            )
            if (v === 'summary') return (
              <div className="summary">
                <p className="eyebrow">
                  Your list · {items.reduce((n, i) => n + i.quantity, 0)} cards{meta ? ` · ${meta.price_date}` : ''}
                </p>
                <dl>
                  <div className="lead"><dt>Fair value<span>{backend === 'gp' ? 'Deep GP' : 'Model A'}</span></dt><dd>{usd(total((r) => fairOf(r, backend).mid))}</dd></div>
                  <div><dt>Market<span>today</span></dt><dd>{usd(total((r) => r.price_today))}</dd></div>
                  <div><dt>In 14 days<span>Model B</span></dt><dd>{usd(total((r) => r.pred_price))}</dd></div>
                  <div><dt>CK buylist<span>Model C</span></dt><dd>{usd(total((r) => r.buylist_pred))}</dd></div>
                </dl>
              </div>
            )
            if (v === 'custom') return (
              <CustomCardForm index={index} onCancel={() => turnTo('search')}
                onAdd={(card: CustomCardInput, quantity: number) => add({ id: newId(), kind: 'custom', card, quantity })} />
            )
            return <>
              <p className="eyebrow">Magic: The Gathering</p>
              <h1>MTG Price Intelligence</h1>
              <p className="lede">
                Fair value, a 14-day TCGplayer forecast and a Card Kingdom buylist forecast for any list of
                printings — real or custom.
              </p>
            </>
          }} />
          <div className="masthead-row">
            <div className="segmented" role="radiogroup" aria-label="Fair-value model">
              <button type="button" role="radio" aria-checked={backend === 'a'} className={backend === 'a' ? 'on' : ''}
                onClick={() => setBackend('a')}>Model A</button>
              <button type="button" role="radio" aria-checked={backend === 'gp'} className={backend === 'gp' ? 'on' : ''}
                onClick={() => setBackend('gp')}>Deep GP</button>
            </div>
            {meta && <span className="stamp">Prices as of {meta.price_date}</span>}
          </div>
        </header>

        {loadError && <p className="error" role="alert">Couldn't load price data: {loadError}</p>}

        <section className="list">
          <div className="list-header">
            <h2>Your cards</h2>
            <div className="actions">
              <button type="button" className="link" disabled={!index} onClick={() => fileInput.current?.click()}>Import CSV</button>
              <input ref={fileInput} type="file" accept=".csv,text/csv" hidden
                onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); e.target.value = '' }} />
              <button type="button" className="link" disabled={items.length === 0} onClick={download}>Download CSV</button>
              <button type="button" className="link" disabled={items.length === 0} onClick={clearList}>Clear list</button>
              <button type="button" className="predict" disabled={items.length === 0} onClick={togglePredict}>
                {mode === 'cards' ? 'Predict' : 'Show cards'}
              </button>
            </div>
          </div>

          {importNote && <p className="import-note" role="status">{importNote}</p>}


          <CardGrid lines={lines} mode={mode} backend={backend} fresh={fresh} leaving={leaving}
            clearing={clearing} cleared={cleared}
            onRemove={remove}
            onQuantity={(id, q) => setItems((it) => it.map((i) => (i.id === id ? { ...i, quantity: q } : i)))}
            onOpen={(l) => l.row && setDetail(l.row)}
            onEmptySlot={openEditor} />
        </section>

        {detail && (
          <div className="modal" role="dialog" aria-modal="true" aria-label={`${detail.name} details`}
            onClick={(e) => { if (e.target === e.currentTarget) setDetail(null) }}>
            <div className="detail-modal">
              <div className="detail-head">
                <h2>{detail.name}</h2>
                <button type="button" className="secondary" onClick={() => setDetail(null)}>Close</button>
              </div>
              <Detail row={detail} backend={backend} />
            </div>
          </div>
        )}

        <footer>
          {meta && <>
            Model A: out-of-fold error ×{Math.exp(meta.model_a.oof_log_mae).toFixed(2)} typical, {Math.round(meta.model_a.oof_coverage80 * 100)}% of
            prices inside its 80% range. Models B/C trained through {meta.model_b.cutoff}. {meta.n_scored.toLocaleString()} printings
            tradeable today ({usd(0.5)}+).{' '}
          </>}
          Custom cards are priced in your browser by Models A and B (the deep GP covers existing printings only). Ranges are
          80% intervals. Not financial advice. Magic: The Gathering is a trademark of Wizards of the Coast; card images ©
          Wizards of the Coast, via Scryfall. Unofficial fan project.
        </footer>
      </main>
    </div>
  )
}
