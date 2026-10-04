/**
 * Custom cards, priced in the browser. A port of the Python path the API used:
 *   cards.custom_rows            -> customRow
 *   fair_value.with_history      -> history (distinct release days per card, as of the price date)
 *   fair_value.static_features   -> featuresA  (float32, like the training frame)
 *   market.custom_day0_rows      -> featuresB
 * scripts/app/export_web_models.py writes the parity fixture this must reproduce
 * (src/model/parity.test.ts).
 */
import { predictHGB, type HGB } from './hgb'

export interface ModelA {
  as_of: string
  q: number
  sigma_floor: number
  columns: string[]
  point: HGB
  sigma: HGB
  vocab: {
    lists: Record<string, string[]>
    cats: Record<string, string[]>
    rank_fill: Record<string, number>
    set_size: Record<string, number>
    slot_size: Record<string, number>
    set_price: { mean: Record<string, number | null>; chase: Record<string, number | null> }
    rate_cols: string[]
    unknown_set: Record<string, Record<string, number | null> & { slot: Record<string, number | null> }>
  }
}

export interface ModelB {
  cutoff: string
  q: number
  features: string[]
  cats: string[]
  cat_levels: Record<string, string[]>
  models: [HGB, HGB, HGB]
  sib: Record<string, [number | null, number | null]>
  set_move: Record<string, number | null>
  constants: { horizon: number; new_release_days: number; edhrec_worst: number }
}

export interface CustomDefaults {
  as_of_day: number
  legal_formats: string[]
  legal_codes: Record<string, string>
  salt_median: number
  demand_rank: Record<string, number | null>
  booster_set_types: string[]
  eternal: string[]
  standard_era: string[]
  rate_cols: string[]
  pull_defaults: Record<string, Record<string, number | null>>
}

export interface Reprints {
  names: Record<string, string>
  oracles: Record<string, { rank: number | null; salt: number | null; tags: string[]; legal: string; days: number[]; n: number }>
}

/** Mirrors mtg_price_prediction.cards.CustomCard (API field names). */
export interface CustomCard {
  name: string
  rarity: string
  type_line: string
  mana_cost?: string
  oracle_text?: string
  power?: string | null
  toughness?: string | null
  loyalty?: string | null
  defense?: string | null
  finish?: string
  set_type?: string
  frame?: string
  border_color?: string
  frame_effects?: string[]
  promo_types?: string[]
  keywords?: string[]
  full_art?: boolean
  textless?: boolean
  released_at?: string | null
  demand?: string
  reprint_of?: string | null
  set?: string | null
}

const DAY_MS = 86_400_000
const toDay = (iso: string) => Math.round(Date.parse(`${iso.slice(0, 10)}T00:00:00Z`) / DAY_MS)
const symbols = (text: string | undefined) => [...(text ?? '').matchAll(/\{([^}]*)\}/g)].map((m) => m[1])
const isDigits = (s: string) => /^\d+$/.test(s)

export function manaValue(cost: string | undefined): number {
  let total = 0
  for (const s of symbols(cost)) {
    if (isDigits(s)) total += Number(s)
    else if (s === 'X' || s === 'Y' || s === 'Z') continue
    else {
      const digits = s.split('/').filter(isDigits).map(Number)
      total += digits.length ? Math.max(...digits) : 1
    }
  }
  return total
}

export const colorsOf = (text: string | undefined) =>
  [...'WUBRG'].filter((c) => symbols(text).some((s) => s.split('/').includes(c)))

/** fair_value._mana: colored pips, max pips of one color, has X, hybrid, Phyrexian. */
function manaFeatures(cost: string | undefined): [number, number, number, number, number] {
  const syms = symbols(cost).map((s) => s.split('/'))
  const colored = syms.filter((s) => s.some((p) => 'WUBRG'.includes(p) && p.length === 1))
  const perColor = [...'WUBRG'].map((ch) => colored.filter((s) => s.includes(ch)).length)
  const hasX = syms.some((s) => s.length === 1 && s[0] === 'X')
  const hybrid = colored.some((s) => new Set(s.filter((p) => p !== 'P')).size > 1)
  const phyrexian = syms.some((s) => s.includes('P'))
  return [colored.length, Math.max(...perColor), hasX ? 1 : 0, hybrid ? 1 : 0, phyrexian ? 1 : 0]
}

/** A dataset-shaped row for one custom card (cards.custom_rows). */
export interface CustomRow {
  [k: string]: unknown
  printing_key: string
  oracle_id: string
  reprint_of_oracle: string | null
  set: string
  set_type: string
  rarity: string
  finish: string
  released_day: number
  edhrec_rank: number | null
  edhrec_salt: number | null
  lists: Record<string, string[]>
  legal: Record<string, string>
  rates: Record<string, number | null>
}

export function customRow(i: number, c: CustomCard, d: CustomDefaults, rp: Reprints | null): CustomRow {
  const finish = c.finish ?? 'nonfoil'
  const setType = c.set_type ?? 'expansion'
  const demand = c.demand ?? 'playable'
  let refOracle: string | null = null
  if (c.reprint_of) {
    if (!rp) throw new Error('reprint lookups not loaded')
    refOracle = rp.oracles[c.reprint_of] ? c.reprint_of : (rp.names[c.reprint_of] ?? null)
    if (!refOracle) throw new Error(`reprint_of ${JSON.stringify(c.reprint_of)}: no card with that oracle_id or name`)
  }
  const ref = refOracle && rp ? rp.oracles[refOracle] : null
  const typeLine = c.type_line ?? ''
  const text = c.oracle_text ?? ''
  const legendaryCreature = typeLine.includes('Legendary') && typeLine.includes('Creature')
  const produced = new Set<string>()
  for (const m of text.matchAll(/Add ((?:\{[WUBRGC]\})+)/g)) for (const s of symbols(m[1])) produced.add(s)
  const identity = new Set([...colorsOf(c.mana_cost), ...colorsOf(text)])
  const legal: Record<string, string> = {}
  d.legal_formats.forEach((fmt, k) => {
    legal[fmt] = ref
      ? (d.legal_codes[ref.legal[k]] ?? 'NA')
      : d.eternal.includes(fmt) || (d.standard_era.includes(fmt) && (setType === 'expansion' || setType === 'core'))
        ? 'legal' : 'not_legal'
  })
  const boosters = d.booster_set_types.includes(setType) && `${c.rarity}|${finish}` in d.pull_defaults
  const rates: Record<string, number | null> = {}
  for (const col of d.rate_cols) {
    const product = col.split('_rate')[0]
    rates[col] = boosters && (product === 'play' || product === 'collector')
      ? (d.pull_defaults[`${c.rarity}|${finish}`][col] ?? null) : null
  }
  return {
    printing_key: `custom:${i}`,
    name: c.name,
    oracle_id: refOracle ?? `custom:${i}`,
    reprint_of_oracle: refOracle,
    set: (c.set ?? '').toLowerCase() || 'custom',
    set_type: setType,
    rarity: c.rarity,
    finish,
    released_day: c.released_at ? toDay(c.released_at) : d.as_of_day,
    mana_cost: c.mana_cost ?? '',
    conv_mana_cost: manaValue(c.mana_cost),
    oracle_text: text,
    power: c.power ?? null,
    toughness: c.toughness ?? null,
    loyalty: c.loyalty ?? null,
    defense: c.defense ?? null,
    frame: c.frame ?? '2015',
    border_color: c.border_color ?? 'black',
    layout: 'normal',
    lang: 'en',
    full_art: !!c.full_art,
    textless: !!c.textless,
    borderless: (c.border_color ?? 'black') === 'borderless',
    reserved: false,
    game_changer: false,
    promo: (c.promo_types ?? []).length > 0,
    reprint: ref !== null,
    booster: d.booster_set_types.includes(setType),
    story_spotlight: false,
    can_lead_commander: legendaryCreature || text.includes('can be your commander'),
    can_lead_oathbreaker: typeLine.includes('Planeswalker'),
    can_lead_brawl: legendaryCreature || typeLine.includes('Planeswalker'),
    edhrec_rank: ref ? ref.rank : (d.demand_rank[demand] ?? null),
    edhrec_salt: ref ? ref.salt : d.salt_median,
    lists: {
      colors: colorsOf(c.mana_cost),
      color_identity: [...'WUBRG'].filter((x) => identity.has(x)),
      produced_mana: [...produced].sort(),
      keywords: c.keywords ?? [],
      type_tags: typeLine.split(/[\s—-]+/).filter((w) => w && w !== '//').map((w) => w.toLowerCase()),
      oracle_tags: ref ? ref.tags : [],
      promo_types: c.promo_types ?? [],
      frame_effects: c.frame_effects ?? [],
    },
    legal,
    rates,
  }
}

/** with_history / print_history: distinct (card, release day) events up to the price date. */
export function history(rows: CustomRow[], asOfDay: number, rp: Reprints | null) {
  return rows.map((r) => {
    const days = new Set<number>(rp && r.reprint_of_oracle ? (rp.oracles[r.reprint_of_oracle]?.days ?? []) : [])
    for (const o of rows) if (o.oracle_id === r.oracle_id) days.add(o.released_day) // the batch's own releases count too
    const seen = [...days].filter((x) => x <= asOfDay)
    const known = seen.length > 0
    return {
      printing_age_days: asOfDay - r.released_day,
      card_age_days: known ? asOfDay - Math.min(...seen) : NaN,
      days_since_last_printing: known ? asOfDay - Math.max(...seen) : NaN,
      printings_to_date: seen.length,
    }
  })
}

const num = (v: unknown) => {
  if (v == null || v === '') return NaN
  const n = Number(v)
  return /^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$/.test(String(v)) ? n : NaN
}

/** Model A's static_features for custom rows, reindexed to the model's columns, float32. */
export function featuresA(rows: CustomRow[], hist: ReturnType<typeof history>, m: ModelA): Float64Array[] {
  const v = m.vocab
  const mainRates = v.rate_cols.filter((c) => ['play', 'draft', 'set', 'default'].includes(c.split('_rate')[0]))
  return rows.map((r, k) => {
    const h = hist[k]
    const f: Record<string, number> = {
      conv_mana_cost: r.conv_mana_cost as number,
      edhrec_salt: r.edhrec_salt ?? NaN,
      ...h,
    }
    for (const b of ['reserved', 'game_changer', 'promo', 'reprint', 'full_art', 'textless', 'booster', 'story_spotlight',
      'can_lead_commander', 'can_lead_oathbreaker', 'can_lead_brawl', 'borderless']) f[b] = r[b] ? 1 : 0
    for (const c of ['power', 'toughness', 'loyalty', 'defense']) f[c] = num(r[c])
    f.log_edhrec_rank = Math.log(r.edhrec_rank ?? v.rank_fill.edhrec_rank)
    f.log_penny_rank = Math.log(v.rank_fill.penny_rank)
    const mana = manaFeatures(r.mana_cost as string)
    ;['colored_pips', 'max_pips_one_color', 'has_x', 'hybrid', 'phyrexian'].forEach((c, i) => { f[c] = mana[i] })
    f.oracle_text_len = (r.oracle_text as string).length
    f.has_all_parts = 0
    f.is_variation = 0
    const R = v.rate_cols.map((c) => r.rates[c] ?? 0)
    v.rate_cols.forEach((c, i) => { f[c] = R[i] })
    const known = r.set in v.set_size
    f.set_size = known ? v.set_size[r.set] : NaN
    f.rarity_slot_size = v.slot_size[`${r.set}|${r.rarity}`] ?? NaN
    f.n_products = R.filter((x) => x > 0).length
    f.max_rate = Math.max(...R)
    f.sum_rate = R.reduce((a, b) => a + b, 0)
    const collector = r.rates.collector_rate_percent ?? 0
    f.collector_only = collector > 0 && mainRates.reduce((a, c) => a + (r.rates[c] ?? 0), 0) === 0 ? 1 : 0
    f.set_price_level = v.set_price.mean[r.set] ?? NaN
    f.set_chase_level = v.set_price.chase[r.set] ?? NaN
    if (!known) {
      const dflt = v.unknown_set[r.set_type] ?? v.unknown_set._all
      f.set_size = dflt.set_size ?? NaN
      f.rarity_slot_size = dflt.slot[r.rarity] ?? dflt.rarity_slot_size ?? NaN
      f.set_price_level = dflt.set_price_level ?? NaN
      f.set_chase_level = dflt.set_chase_level ?? NaN
    }
    for (const [col, levels] of Object.entries(v.lists)) {
      const have = new Set(r.lists[col] ?? [])
      for (const level of levels) f[`${col}=${level}`] = have.has(level) ? 1 : 0
    }
    const age = h.printing_age_days
    const release = age < 0 ? 'preorder' : age <= 30 ? 'new' : 'established'
    for (const [col, levels] of Object.entries(v.cats)) {
      const value = col === 'release_status' ? release
        : col.startsWith('legalities.') ? (r.legal[col.slice('legalities.'.length)] ?? 'NA')
          : r[col] == null ? 'NA' : String(r[col])
      for (const level of levels) f[`${col}=${level}`] = value === level ? 1 : 0
    }
    return Float64Array.from(m.columns, (c) => (c in f ? Math.fround(f[c]) : 0))
  })
}

export interface FairValue { fair_log: number; fair: number; fair_lo: number; fair_hi: number }

export function predictA(x: Float64Array, m: ModelA): FairValue {
  const fairLog = predictHGB(m.point, x)
  const half = m.q * Math.max(predictHGB(m.sigma, x), m.sigma_floor)
  return { fair_log: fairLog, fair: Math.exp(fairLog), fair_lo: Math.exp(fairLog - half), fair_hi: Math.exp(fairLog + half) }
}

/** market.custom_day0_rows: one day of data, the Model A price; no momentum or other markets. */
export function featuresB(r: CustomRow, h: ReturnType<typeof history>[number], day0: number, asOfDay: number, m: ModelB): number[] {
  const logPrice = Math.log(day0)
  const sib = r.reprint_of_oracle ? m.sib[r.reprint_of_oracle] : undefined
  const sibChg = sib?.[0] ?? NaN
  const f: Record<string, number> = {
    log_price: logPrice, change_21d_pct: 0, volatility_21d: 0, jump_7d: 0,
    releases_before_target: r.released_day > asOfDay && r.released_day <= asOfDay + m.constants.horizon ? 1 : 0,
    ...h,
    log_edhrec: Math.log(r.edhrec_rank ?? m.constants.edhrec_worst),
    reprint: r.reprint ? 1 : 0, promo: r.promo ? 1 : 0, reserved: 0,
    sib_chg: sibChg, sib_div: -sibChg, sib_rel_level: logPrice - (sib?.[1] ?? NaN),
    set_peer: m.set_move[r.set] ?? NaN,
    lvl_ck_retail: NaN, lvl_ck_buylist: NaN, lvl_cardmarket: NaN,
  }
  const release = r.released_day > asOfDay ? 'preorder'
    : r.released_day > asOfDay - m.constants.new_release_days ? 'new' : 'established'
  const finishGroup = ['nonfoil', 'foil', 'etched'].includes(r.finish) ? r.finish : 'special'
  const raw: Record<string, string> = { release_status: release, rarity: r.rarity, finish_group: finishGroup, set_type: r.set_type }
  const code = (c: string) => { const i = m.cat_levels[c].indexOf(raw[c]); return i < 0 ? NaN : i }
  return [...m.features.map((c) => f[c] ?? NaN), ...m.cats.map(code)]
}

export interface MoveForecast { move_med: number; move_lo: number; move_hi: number }

export function predictB(x: number[], m: ModelB): MoveForecast {
  const [lo0, med, hi0] = m.models.map((t) => predictHGB(t, x))
  const lo = Math.min(Math.min(lo0, med) - m.q, med)
  const hi = Math.max(Math.max(hi0, med) + m.q, med)
  return { move_med: med, move_lo: lo, move_hi: hi }
}
