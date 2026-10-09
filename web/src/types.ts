export type Signal = 'BUY' | 'SELL' | 'HOLD'
export type Backend = 'a' | 'gp'
export type Demand = 'staple' | 'popular' | 'playable' | 'niche' | 'unplayed'

export interface Comp {
  printing_key: string
  name: string
  set: string
  finish: string
  price_today: number | null
}

/** One prediction row: the schema of the static data in public/data/. */
export interface PredictionRow {
  printing_key: string
  scryfall_id?: string
  name: string
  set: string
  set_name?: string
  collector_number?: string
  finish: string
  rarity: string
  status: 'scored' | 'valued only' | 'unpriced' | 'custom'
  price_date: string
  price_today: number | null
  ck_retail?: number | null
  ck_buylist?: number | null
  cardmarket_eur?: number | null
  fair: number | null
  fair_lo: number | null
  fair_hi: number | null
  fair_gp?: number | null
  fair_gp_lo?: number | null
  fair_gp_hi?: number | null
  model_a_price?: number | null
  pred_price: number | null
  price_lo: number | null
  price_hi: number | null
  signal: Signal | null
  unconfirmed_jump?: boolean | null
  buylist_pred?: number | null
  buylist_lo?: number | null
  buylist_hi?: number | null
  comps?: Comp[]
  caveats?: string[]
}

export interface CustomCardInput {
  name: string
  rarity: 'common' | 'uncommon' | 'rare' | 'mythic' | 'special' | 'bonus'
  type_line: string
  mana_cost: string
  oracle_text: string
  power?: string
  toughness?: string
  loyalty?: string
  finish: 'nonfoil' | 'foil' | 'etched'
  set_type: string
  frame: string
  border_color: string
  frame_effects: string[]
  released_at?: string
  demand: Demand
  reprint_of?: string
  set?: string
}

export type ListItem =
  | { id: string; kind: 'existing'; printing_key: string; set: string; quantity: number; name?: string }
  | { id: string; kind: 'custom'; card: CustomCardInput; quantity: number }

/** Column-wise search index (web/public/data/index.json). */
export interface SearchIndex {
  printing_key: string[]
  name: string[]
  set: string[]
  collector_number: string[]
  finish: string[]
  /** optional visual treatment tags (borderless, showcase, extended art, etc.) */
  treatment?: string[]
  /** first letter of the rarity (c / u / r / m / s / b) */
  rarity: string[]
  /** set code -> set name */
  sets: Record<string, string>
}

export interface Meta {
  price_date: string
  model_a: { as_of: string; oof_log_mae: number; oof_coverage80: number }
  model_b: { cutoff: string }
  model_c: { cutoff: string }
  n_printings: number
  n_scored: number
}
