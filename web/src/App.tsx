import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { FocusBody, FocusHeader, focusLeaveMs, useCardInfo } from './components/CardFocus'
import { FlipButton } from './components/FlipButton'
import { DocBody, docLeaveMs } from './components/DocBody'
import { SiteNav } from './components/SiteNav'
import { pageMeta, type PageId } from './content/pages'
import { CardGrid, fairOf, type GridLine } from './components/CardGrid'
import { CardSearch } from './components/CardSearch'
import { FlipPanel } from './components/FlipPanel'
import { importCSV } from './csv'
import { CardShowcase, FEATURED, STATE_CARD, type Face, type ShowcaseState } from './components/CardShowcase'
import { CustomCardForm } from './components/CustomCardForm'
import { loadIndex, loadMeta, loadSet, pctChange, scryfallImage, usd } from './data'
import { loadModels, priceCustom } from './model'
import type { Backend, CustomCardInput, ListItem, Meta, PredictionRow, SearchIndex } from './types'

const STORAGE_KEY = 'mtg-price-list-v1'
const FRESH_MS = 1600 // how long a just-added card counts as "flipping in"
const SLOT_FLIP_MS = 750 // allow the add-card slot's 700ms turn to finish
const SHIFT_STEP_MS = 120 // each following card starts its turn after the previous one
const WAVE_STEP_MS = 90 // matches the grid's diagonal wave
const FLIP_MS = 750 // matches .tile-inner's transition
const INTRO_MS = 900 // the left card spins alone for this long before the page arrives
const EDGE_ON = 0.14 // .tile-inner's cubic-bezier(0.2, 0.7, 0.15, 1) passes 90° at ~14% of its duration
const STEP_PAUSE_MS = 120 // a beat of empty space between one view leaving and the next arriving

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

type DocPage = Exclude<PageId, 'predictor'>
type Panel = 'title' | 'summary' | 'search' | 'card' | `doc-${DocPage}`

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
  const [removal, setRemoval] = useState<{ id: string; phase: 'slot' | 'wave' } | null>(null)
  const [gridEpoch, setGridEpoch] = useState(0)
  const [clearing, setClearing] = useState(false)
  const [cleared, setCleared] = useState(false)
  // the card whose results take over the right page (clicked from its price face)
  const [focusId, setFocusId] = useState<string | null>(null)
  /** retract: the grid turns away · open: details shown · leave: details turn away · restore: the grid turns back */
  const [focusPhase, setFocusPhase] = useState<'retract' | 'open' | 'leave' | 'restore'>('open')
  const [gridCols, setGridCols] = useState(3)
  const [panel, setPanel] = useState<Panel>('title')
  // Site pages: the brand opens a row of page cards; a documentation page takes over the right half
  const [navOpen, setNavOpen] = useState(false)
  const [site, setSite] = useState<PageId>('predictor')
  const [docShown, setDocShown] = useState<DocPage | null>(null) // the page whose content is mounted
  const [docLeaving, setDocLeaving] = useState(false)
  const [holdRetract, setHoldRetract] = useState(false) // keeps the grid turned away for one frame after it's shown again
  const [burst, setBurst] = useState<{ id: number; face?: Face } | null>(null)
  // Page-load intro: the left card spins on its own first, then the title turns over out of the background
  // while the rest of the text and buttons fade in and the grid deals itself in. The page is inert meanwhile.
  const [intro, setIntro] = useState(() => !(window.matchMedia?.('(prefers-reduced-motion: reduce)').matches ?? false))
  useEffect(() => {
    if (!intro) return
    setBurst({ id: 1 })
    const t = window.setTimeout(() => setIntro(false), INTRO_MS)
    return () => window.clearTimeout(t)
  }, []) // eslint-disable-line react-hooks/exhaustive-deps
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

  // The masthead turns over between the title, the price summary and the card search.
  const turnTo = (p: Panel) => {
    if (p === panel) return
    setPanel(p)
    if (p === 'search') {
      if (window.scrollY > 200) window.scrollTo({ top: 0, behavior: 'smooth' })
      window.setTimeout(() => document.getElementById('card-search')?.focus({ preventScroll: true }), 450)
    }
  }
  const openEditor = () => turnTo('search')

  // The custom-card form lives on the back of the whole right half: the page turns over like a card.
  const [customOpen, setCustomOpen] = useState(false)
  const [pageTurning, setPageTurning] = useState(false)
  // Every turn adds 180deg, so the page always turns the same way (left to right) like every other flip.
  // At rest on the front the transform is dropped altogether and the angle starts again from 0.
  const [pageDeg, setPageDeg] = useState(0)
  const startPageTurn = (open: boolean) => {
    setPageTurning(true)
    setPageDeg((d) => d + 180)
    setCustomOpen(open)
  }
  const turnPage = (open: boolean) => {
    if (open === customOpen) return
    startPageTurn(open)
    // bring the top of the right half into view; the form card then fills the window and scrolls inside
    const top = (document.querySelector('.page') as HTMLElement | null)?.offsetTop ?? 0
    if (open && window.scrollY !== top) window.scrollTo({ top })
  }
  useEffect(() => { // the page itself stays put while the form card is up
    if (!customOpen) return
    const root = document.documentElement
    const prev = root.style.overflow
    root.style.overflow = 'hidden'
    return () => { root.style.overflow = prev }
  }, [customOpen])
  // Where the masthead rests: the title, or the price summary while predictions are showing.
  const home: Panel = site !== 'predictor' ? `doc-${site}` : mode === 'prices' ? 'summary' : 'title'

  const focusLine = focusId ? lines.find((l) => l.item.id === focusId) ?? null : null
  const focusInfo = useCardInfo(focusLine)
  // A card's details take over in steps: the grid turns away in its wave, the details card turns over into
  // the empty space and the similar cards are dealt in. Going back runs the same steps in reverse.
  const focusActive = !!focusLine && (focusPhase === 'retract' || focusPhase === 'open')
  // The next step starts once the last card has turned edge-on (gone from view), not when its turn ends:
  // the flip curve eases out, so a card disappears ~14% of the way through and spends the rest turning unseen.
  let lastWave = 0
  for (let i = 0; i < items.length; i++) lastWave = Math.max(lastWave, (i % gridCols) + Math.floor(i / gridCols))
  const gridWaveMs = lastWave * WAVE_STEP_MS + EDGE_ON * FLIP_MS + STEP_PAUSE_MS
  const openFocus = (id: string) => {
    if (focusId) return
    setFocusId(id)
    setFocusPhase('retract')
    setPanel('card')
    if (window.scrollY > 0) window.scrollTo({ top: 0, behavior: 'smooth' })
    freshTimers.current.push(window.setTimeout(() => setFocusPhase('open'), gridWaveMs))
  }
  const closeFocus = (immediate = false) => {
    if (!focusId || focusPhase === 'leave' || focusPhase === 'restore') return
    setPanel(home)
    if (immediate) { setFocusId(null); setFocusPhase('open'); return }
    setFocusPhase('leave')
    freshTimers.current.push(window.setTimeout(() => {
      setFocusPhase('restore') // the grid is shown again, still turned away...
      freshTimers.current.push(window.setTimeout(() => {
        setFocusId(null) // ...then turns back in its wave
        setFocusPhase('open')
      }, 60))
    }, focusLeaveMs(focusLine)))
  }
  useEffect(() => { if (focusId && !focusLine) closeFocus(true) }, [focusId, focusLine]) // eslint-disable-line react-hooks/exhaustive-deps

  // Switching pages: what's on the right turns away (the grid in its wave, a card's details, or the current
  // page's blocks), the masthead turns over to the new page's title, then the new content is dealt in.
  const pageBusy = useRef(false)
  const goPage = (target: PageId) => {
    if (target === site || pageBusy.current || customOpen || clearing || removal) return
    pageBusy.current = true
    const after = (ms: number, f: () => void) => freshTimers.current.push(window.setTimeout(f, ms))
    setSite(target)
    setPanel(target === 'predictor' ? (mode === 'prices' ? 'summary' : 'title') : `doc-${target}`)
    if (window.scrollY > 0) window.scrollTo({ top: 0, behavior: 'smooth' })
    const show = () => {
      if (target === 'predictor') { // the grid comes back: shown still turned away, then turns back in its wave
        setDocShown(null)
        setHoldRetract(true)
        after(60, () => { setHoldRetract(false); pageBusy.current = false })
      } else {
        setDocShown(target)
        pageBusy.current = false
      }
    }
    if (docShown) { // a page's blocks turn away first
      setDocLeaving(true)
      after(docLeaveMs(docShown), () => { setDocLeaving(false); show() })
    } else if (focusId) { // a card's details turn away; the grid stays turned away underneath
      setFocusPhase('leave')
      after(focusLeaveMs(focusLine), () => { setFocusId(null); setFocusPhase('open'); show() })
    } else { // the grid turns away in its wave (retract follows `site`)
      after(gridWaveMs, show)
    }
  }

  // ---- Showcase state machine (see STATE_CARD): derived from what the page is showing ----
  const showcase: ShowcaseState = customOpen ? 'custom'
    : clearing ? 'cards'
      : panel === 'search' ? 'search'
        : site !== 'predictor' ? site
        : focusActive ? 'card'
          : mode === 'prices' ? 'prices' : 'cards'
  const stateFace = (st: ShowcaseState): Face | null => (st === 'card' ? (focusLine ? faceOf(focusLine.item) : null)
    : st === 'cards' ? null : STATE_CARD[st as keyof typeof STATE_CARD])
  const deckKey = `${showcase}|${clearing}|${focusId}|${items.map((i) => i.id).join(',')}`
  const deck = useMemo<Face[]>(() => {
    const f = stateFace(showcase)
    return f ? [f] : clearing ? FEATURED : items.map(faceOf)
  }, [deckKey]) // eslint-disable-line react-hooks/exhaustive-deps
  // Entering a state spins onto its card; entering `cards` lands on `landOn` (e.g. the card just added) or the list's first.
  const landOn = useRef<Face | null>(null)
  const lastShowcase = useRef(showcase)
  useEffect(() => {
    if (lastShowcase.current === showcase) return
    lastShowcase.current = showcase
    spin(stateFace(showcase) ?? landOn.current ?? (items[0] ? faceOf(items[0]) : FEATURED[0]))
    landOn.current = null
  }, [showcase]) // eslint-disable-line react-hooks/exhaustive-deps
  /** Land on `face` in state `next`: via the state change if there is one, otherwise with a spin of its own. */
  const land = (face: Face, next: ShowcaseState) => {
    if (next === showcase) spin(face)
    else landOn.current = face
  }
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (customOpen) turnPage(false)
      else if (focusActive && panel === 'card') closeFocus()
      else setPanel(focusActive ? 'card' : home)
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [home, customOpen, focusActive, panel]) // eslint-disable-line react-hooks/exhaustive-deps

  const addMany = (added: ListItem[]) => {
    if (!added.length) return
    land(faceOf(added[0]), 'cards')
    setFocusId(null)
    setCleared(false)
    setMode('cards') // new cards are shown face-up; Predict again to price the whole list
    setPanel('title')
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


  const remove = (id: string) => {
    if (removal || clearing) return
    // The showcase spins onto whatever takes the removed card's place.
    const at = items.findIndex((i) => i.id === id)
    const rest = items.filter((i) => i.id !== id)
    if (showcase === 'cards') spin(rest.length ? faceOf(rest[Math.min(at, rest.length - 1)]) : FEATURED[0])
    setRemoval({ id, phase: 'wave' })
    freshTimers.current.push(window.setTimeout(() => {
      setRemoval({ id, phase: 'slot' })
      freshTimers.current.push(window.setTimeout(() => {
        setItems((it) => it.filter((i) => i.id !== id))
        setRemoval(null)
        setGridEpoch((n) => n + 1)
        setFresh(new Set())
        setCleared(true)
      }, SLOT_FLIP_MS + 50))
    }, (items.length - at) * SHIFT_STEP_MS))
  }

  const clearList = () => {
    if (clearing || removal || items.length === 0) return
    // Every card turns over in the wave: the first becomes the add slot, the rest turn to nothing.
    land(FEATURED[0], 'cards')
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
    // The masthead turns over to the totals; the showcase's spin onto the Treasure doubles as "working on it".
    setPanel(mode === 'cards' ? 'summary' : 'title')
    setMode((m) => (m === 'cards' ? 'prices' : 'cards'))
  }

  const total = (f: (r: PredictionRow) => number | null | undefined) =>
    lines.reduce((acc, l) => acc + (l.row && f(l.row) != null ? (f(l.row) as number) * l.item.quantity : 0), 0)
  // Expected move of the whole list, over the cards that have both a price today and a forecast.
  const forecastable = lines.filter((l) => l.row?.price_today != null && l.row?.pred_price != null)
  const listChange = pctChange(
    forecastable.reduce((a, l) => a + (l.row!.price_today as number) * l.item.quantity, 0),
    forecastable.reduce((a, l) => a + (l.row!.pred_price as number) * l.item.quantity, 0))

  const download = () => {
    const blob = new Blob([toCSV(lines, backend)], { type: 'text/csv' })
    const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(blob), download: 'mtg-price-list.csv' })
    a.click()
    URL.revokeObjectURL(a.href)
  }

  const modelControls = () => (
    <div className="masthead-row">
      <div className="segmented" role="radiogroup" aria-label="Fair-value model">
        <button type="button" role="radio" aria-checked={backend === 'a'} className={backend === 'a' ? 'on' : ''}
          onClick={() => setBackend('a')}>Boosted trees</button>
        <button type="button" role="radio" aria-checked={backend === 'gp'} className={backend === 'gp' ? 'on' : ''}
          onClick={() => setBackend('gp')}>Deep GP</button>
      </div>
      {meta && <span className="stamp">Prices as of {meta.price_date}</span>}
    </div>
  )

  return (
    <div className={`shell ${intro ? 'intro' : ''}`} inert={intro}>
      <button type="button" className="brand" aria-expanded={navOpen} aria-label="MTG Price Intelligence — pages"
        onClick={() => setNavOpen((o) => !o)}>
        <img src="./logo.svg" alt="" width={30} height={30} />
        <span>MTG Price Intelligence</span>
      </button>
      <SiteNav open={navOpen} page={site} onPick={goPage} />
      <CardShowcase deck={deck} burst={burst} />
      <div className={`page ${customOpen ? 'back' : ''} ${pageTurning ? 'turning' : ''}`}>
      <div className="page-inner" style={customOpen || pageTurning ? { transform: `rotateY(${pageDeg}deg)` } : undefined}
        onTransitionEnd={(e) => {
          if (e.target !== e.currentTarget) return
          setPageTurning(false)
          if (!customOpen) setPageDeg(0)
        }}>
      <main className="content page-face page-front" inert={customOpen}>
        <header className="masthead">
          <FlipPanel view={panel} rest="title" boxed={(v) => v === 'search'} render={(v) => {
            if (v === 'search') return (
              <div className="panel-search">
                <p className="eyebrow">Add a card</p>
                <CardSearch index={index} onAdd={(h) =>
                  add({ id: newId(), kind: 'existing', printing_key: h.printing_key, set: h.set, name: h.name, quantity: 1 })}
                  onCustom={() => turnPage(true)} onCancel={() => turnTo(home)} />
              </div>
            )
            if (v.startsWith('doc-')) {
              const m = pageMeta(v.slice(4) as DocPage)
              return <>
                <p className="eyebrow">{m.eyebrow}</p>
                <h1>{m.title}</h1>
                <p className="lede">{m.lede}</p>
              </>
            }
            if (v === 'card') return <><FocusHeader info={focusInfo} />{modelControls()}</>
            if (v === 'summary') return (
              <div className="summary">
                <p className="eyebrow">
                  Your list · {items.reduce((n, i) => n + i.quantity, 0)} cards{meta ? ` · ${meta.price_date}` : ''}
                </p>
                <dl>
                  <div className="lead"><dt>Fair value<span>{backend === 'gp' ? 'Deep GP' : 'Boosted trees'}</span></dt><dd>{usd(total((r) => fairOf(r, backend).mid))}</dd></div>
                  <div className="lead-change"><dt>14-day change<span>TCGplayer forecast</span></dt>
                    <dd className={listChange ? (listChange.value > 0 ? 'up' : listChange.value < 0 ? 'down' : '') : ''}>{listChange?.text ?? '—'}</dd></div>
                  <div><dt>Market<span>today</span></dt><dd>{usd(total((r) => r.price_today))}</dd></div>
                  <div><dt>In 14 days<span>TCGplayer forecast</span></dt><dd>{usd(total((r) => r.pred_price))}</dd></div>
                  <div><dt>CK buylist<span>Buylist forecast</span></dt><dd>{usd(total((r) => r.buylist_pred))}</dd></div>
                </dl>
                {modelControls()}
              </div>
            )
            return <>
              <p className="eyebrow">Magic: The Gathering</p>
              <div className="title-turn"><h1>MTG Price Intelligence</h1></div>
              <p className="lede">
                Fair value, a 14-day TCGplayer forecast and a Card Kingdom buylist forecast for any list of
                printings — real or custom.
              </p>
              {modelControls()}
            </>
          }} />
        </header>

        {loadError && <p className="error" role="alert">Couldn't load price data: {loadError}</p>}

        <section className="list">
          <div className={`list-header ${site !== 'predictor' ? 'away' : ''}`} hidden={!!docShown} inert={site !== 'predictor'}>
            {/* the heading fades out while a view turns away; it changes while invisible and fades back in */}
            <h2 className={`fade-title ${focusPhase === 'retract' || focusPhase === 'leave' ? 'faded' : ''}`}>
              {focusLine && (focusPhase === 'open' || focusPhase === 'leave') ? 'Prediction' : 'Your cards'}
            </h2>
            <div className="actions">
              <span className={`list-links ${focusLine ? 'muted-out' : ''}`} inert={!!focusLine}>
                <button type="button" className="link" disabled={!index} onClick={() => fileInput.current?.click()}>Import CSV</button>
                <input ref={fileInput} type="file" accept=".csv,text/csv" hidden
                  onChange={(e) => { const f = e.target.files?.[0]; if (f) void importFile(f); e.target.value = '' }} />
                <button type="button" className="link" disabled={items.length === 0} onClick={download}>Download CSV</button>
                <button type="button" className="link" disabled={items.length === 0 || !!removal} onClick={clearList}>Clear list</button>
              </span>
              <FlipButton labels={['Predict', 'Show cards', 'Back to cards']}
                label={focusActive ? 'Back to cards' : mode === 'cards' ? 'Predict' : 'Show cards'}
                disabled={items.length === 0 || !!removal || (!!focusLine && !focusActive) || focusPhase === 'retract'}
                onClick={() => (focusActive ? closeFocus() : togglePredict())} />
            </div>
          </div>

          {importNote && !docShown && <p className="import-note" role="status">{importNote}</p>}


          {docShown && <DocBody key={docShown} page={docShown} leaving={docLeaving} />}
          {focusLine && (focusPhase === 'open' || focusPhase === 'leave') && (
            <FocusBody key={focusLine.item.id} line={focusLine} backend={backend} leaving={focusPhase === 'leave'} />
          )}
          <div hidden={(!!focusLine && (focusPhase === 'open' || focusPhase === 'leave')) || !!docShown}>
          <CardGrid key={gridEpoch} lines={lines} mode={mode} backend={backend} fresh={fresh} removal={removal}
            clearing={clearing} cleared={cleared} retract={!!focusLine || intro || site !== 'predictor' || !!docShown || holdRetract} onCols={setGridCols}
            onRemove={remove}
            onQuantity={(id, q) => setItems((it) => it.map((i) => (i.id === id ? { ...i, quantity: q } : i)))}
            onOpen={(l) => openFocus(l.item.id)}
            onEmptySlot={openEditor} />
          </div>
        </section>

      </main>

      <section className="content page-face page-back" inert={!customOpen} aria-hidden={!customOpen}>
        <div className="custom-sheet">
          {(customOpen || pageTurning) && (
            <CustomCardForm index={index} onCancel={() => turnPage(false)}
              onAdd={(card: CustomCardInput, quantity: number) => {
                startPageTurn(false)
                add({ id: newId(), kind: 'custom', card, quantity })
              }} />
          )}
        </div>
      </section>
      </div>
      </div>


    </div>
  )
}
