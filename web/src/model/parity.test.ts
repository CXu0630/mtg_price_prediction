/**
 * The browser port must reproduce Python (scripts/app/export_web_models.py --fixture):
 * Model A's feature vectors exactly, and the final custom-card prices/forecasts.
 * Run after the nightly export:  npm test
 */
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { describe, expect, it } from 'vitest'
import { customRow, featuresA, history, type CustomCard, type CustomDefaults, type ModelA, type ModelB, type Reprints } from './custom'
import { prepareA, prepareB, priceCustom } from './index'

const dir = join(__dirname, '..', '..', 'public', 'data', 'models')
const load = <T,>(name: string) => JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8')) as T
const fixture = JSON.parse(readFileSync(join(__dirname, 'parity.fixture.json'), 'utf8')) as {
  cards: CustomCard[]
  expected: { fair: number; fair_lo: number; fair_hi: number; pred_price: number; price_lo: number; price_hi: number; signal: string }[]
  features_a: (number | null)[][]
}
const models = { a: prepareA(load<ModelA>('model_a')), b: prepareB(load<ModelB>('model_b')), defaults: load<CustomDefaults>('custom'), reprints: load<Reprints>('reprints') }
const close = (a: number, b: number, rel = 1e-6) => Math.abs(a - b) <= rel * Math.max(1, Math.abs(a), Math.abs(b))

describe('custom-card parity with Python', () => {
  it('builds Model A features identically', () => {
    const rows = fixture.cards.map((c, i) => customRow(i, c, models.defaults, models.reprints))
    const xa = featuresA(rows, history(rows, models.defaults.as_of_day, models.reprints), models.a)
    const mismatches: string[] = []
    xa.forEach((x, i) => fixture.features_a[i].forEach((want, j) => {
      const got = x[j]
      const ok = want === null ? Number.isNaN(got) : close(got, want, 1e-6)
      if (!ok) mismatches.push(`card ${i} ${models.a.columns[j]}: got ${got}, want ${want}`)
    }))
    expect(mismatches.slice(0, 20)).toEqual([])
  })

  it('prices custom cards like Python (Models A and B)', () => {
    const got = priceCustom(fixture.cards, models)
    got.forEach((r, i) => {
      const e = fixture.expected[i]
      for (const k of ['fair', 'fair_lo', 'fair_hi', 'pred_price', 'price_lo', 'price_hi'] as const) {
        expect(close(r[k] as number, e[k], 1e-6), `card ${i} ${k}: got ${r[k]}, want ${e[k]}`).toBe(true)
      }
      expect(r.signal, `card ${i} signal`).toBe(e.signal)
    })
  })
})
