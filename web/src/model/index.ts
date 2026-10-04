/** Loads the exported models (lazily, once) and prices custom cards in the browser. */
import type { PredictionRow } from '../types'
import { restoreInfinities } from './hgb'
import {
  customRow, featuresA, featuresB, history, predictA, predictB,
  type CustomCard, type CustomDefaults, type ModelA, type ModelB, type Reprints,
} from './custom'

export interface WebModels { a: ModelA; b: ModelB; defaults: CustomDefaults; reprints: Reprints | null }

const BASE = `${import.meta.env.BASE_URL}data/models`
const get = <T,>(name: string) => fetch(`${BASE}/${name}.json`).then((r) => {
  if (!r.ok) throw new Error(`couldn't load ${name} (${r.status})`)
  return r.json() as Promise<T>
})

let core: Promise<Omit<WebModels, 'reprints'>> | null = null
let reprints: Promise<Reprints> | null = null

export async function loadModels(needReprints: boolean): Promise<WebModels> {
  core ??= Promise.all([get<ModelA>('model_a'), get<ModelB>('model_b'), get<CustomDefaults>('custom')])
    .then(([a, b, defaults]) => ({ a: prepareA(a), b: prepareB(b), defaults }))
  if (needReprints) reprints ??= get<Reprints>('reprints')
  const [c, rp] = await Promise.all([core, needReprints ? reprints : Promise.resolve(null)])
  return { ...c, reprints: rp }
}

export const prepareA = (a: ModelA) => { restoreInfinities(a.point); restoreInfinities(a.sigma); return a }
export const prepareB = (b: ModelB) => { b.models.forEach(restoreInfinities); return b }

/** Same rows the Python scoring.score_custom returned (backend "a"). */
export function priceCustom(cards: CustomCard[], m: WebModels): PredictionRow[] {
  const asOf = m.defaults.as_of_day
  const rows = cards.map((c, i) => customRow(i, c, m.defaults, m.reprints))
  const hist = history(rows, asOf, m.reprints)
  const xa = featuresA(rows, hist, m.a)
  return rows.map((r, i) => {
    const a = predictA(xa[i], m.a)
    const b = predictB(featuresB(r, hist[i], a.fair, asOf, m.b), m.b)
    return {
      printing_key: r.printing_key, name: cards[i].name, set: r.set, finish: r.finish, rarity: r.rarity,
      status: 'custom', price_date: m.a.as_of, price_today: null,
      fair: a.fair, fair_lo: a.fair_lo, fair_hi: a.fair_hi, model_a_price: a.fair,
      pred_price: a.fair * Math.exp(b.move_med), price_lo: a.fair * Math.exp(b.move_lo),
      price_hi: a.fair * Math.exp(b.move_hi),
      signal: b.move_lo > 0 ? 'BUY' : b.move_hi < 0 ? 'SELL' : 'HOLD',
      buylist_pred: null, buylist_lo: null, buylist_hi: null,
      caveats: ['Custom card: one day of price data (the Model A estimate), so the 14-day forecast has no momentum or cross-market signal.',
        'Model C n/a: no Card Kingdom buylist. Deep GP n/a for custom cards (it needs text/art embedding models).'],
    }
  })
}
