import { scryfallImage, usd } from '../data'
import type { Backend, PredictionRow } from '../types'
import { range } from './CardGrid'

/** Everything about one card's prediction (opened from a price face). */
export function Detail({ row, backend }: { row: PredictionRow; backend: Backend }) {
  return (
    <div className="detail-body">
      {row.scryfall_id && <img className="card-img" src={scryfallImage(row.scryfall_id, 'normal')} alt={row.name} />}
      <dl>
        <dt>Status</dt><dd>{row.status}{row.unconfirmed_jump ? ' · unconfirmed price jump (signals held)' : ''}</dd>
        {row.status !== 'custom' && (<>
          <dt>Card Kingdom</dt><dd>retail {usd(row.ck_retail)} · buylist {usd(row.ck_buylist)}</dd>
          <dt>Cardmarket</dt><dd>{row.cardmarket_eur == null ? '—' : `€${row.cardmarket_eur.toFixed(2)}`}</dd>
        </>)}
        <dt>Model A</dt><dd>{usd(row.fair)} ({range(row.fair_lo, row.fair_hi)}, 80% range)</dd>
        {row.fair_gp != null && <><dt>Deep GP</dt><dd>{usd(row.fair_gp)} ({range(row.fair_gp_lo, row.fair_gp_hi)}, 80% range)</dd></>}
        {row.status === 'custom' && <><dt>Day-0 price for Model B</dt><dd>{usd(row.model_a_price)} (from {backend === 'gp' ? 'deep GP' : 'Model A'})</dd></>}
        <dt>Prices as of</dt><dd>{row.price_date}</dd>
      </dl>
      {row.comps && row.comps.length > 0 && (
        <div className="comps">
          <h3>Priced like these (deep GP comps)</h3>
          <ul>
            {row.comps.map((c) => (
              <li key={c.printing_key}>{c.name} <span className="muted">{c.set.toUpperCase()} · {c.finish}</span> {usd(c.price_today)}</li>
            ))}
          </ul>
        </div>
      )}
      {row.caveats && <ul className="caveats">{row.caveats.map((c) => <li key={c}>{c}</li>)}</ul>}
    </div>
  )
}
