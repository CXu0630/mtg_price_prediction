import { Fragment, useState } from 'react'
import { scryfallImage, usd } from '../data'
import type { Backend, PredictionRow } from '../types'

export interface ResultLine {
  id: string
  quantity: number
  row: PredictionRow | null
  error?: string
  pending?: boolean
}

interface Props {
  lines: ResultLine[]
  backend: Backend
  onQuantity: (id: string, q: number) => void
  onRemove: (id: string) => void
}

const fairOf = (r: PredictionRow, backend: Backend) =>
  backend === 'gp' && r.fair_gp != null
    ? { mid: r.fair_gp, lo: r.fair_gp_lo ?? null, hi: r.fair_gp_hi ?? null }
    : { mid: r.fair, lo: r.fair_lo, hi: r.fair_hi }

const range = (lo: number | null | undefined, hi: number | null | undefined) =>
  lo == null || hi == null ? '' : `${usd(lo)} – ${usd(hi)}`

function SignalBadge({ signal }: { signal: PredictionRow['signal'] }) {
  if (!signal) return <span className="muted">—</span>
  const icon = signal === 'BUY' ? '▲' : signal === 'SELL' ? '▼' : '■'
  return <span className={`badge ${signal.toLowerCase()}`}>{icon} {signal}</span>
}

export function ResultsTable({ lines, backend, onQuantity, onRemove }: Props) {
  const [open, setOpen] = useState<string | null>(null)
  const sum = (f: (r: PredictionRow) => number | null | undefined) =>
    lines.reduce((acc, l) => acc + (l.row && f(l.row) != null ? (f(l.row) as number) * l.quantity : 0), 0)

  if (lines.length === 0) {
    return <p className="empty">Your list is empty. Search for an existing card or add a custom one.</p>
  }
  return (
    <div className="table-wrap">
      <table className="results">
        <thead>
          <tr>
            <th scope="col">Card</th>
            <th scope="col" className="num">Qty</th>
            <th scope="col" className="num">Market today</th>
            <th scope="col" className="num">Fair value ({backend === 'gp' ? 'deep GP' : 'Model A'})</th>
            <th scope="col" className="num">In 14 days (Model B)</th>
            <th scope="col">Signal</th>
            <th scope="col" className="num">CK buylist in 14 days (Model C)</th>
            <th scope="col"><span className="sr-only">Actions</span></th>
          </tr>
        </thead>
        <tbody>
          {lines.map((l) => {
            const r = l.row
            const fair = r ? fairOf(r, backend) : null
            return (
              <Fragment key={l.id}>
                <tr className={open === l.id ? 'open' : ''}>
                  <td>
                    <button type="button" className="card-cell" onClick={() => setOpen(open === l.id ? null : l.id)}
                      aria-expanded={open === l.id} disabled={!r}>
                      {r?.scryfall_id && <img src={scryfallImage(r.scryfall_id)} alt="" loading="lazy" />}
                      <span>
                        <strong>{r?.name ?? (l.pending ? 'Pricing…' : 'Unavailable')}</strong>
                        <small>
                          {r ? `${r.set.toUpperCase()}${r.collector_number ? ` #${r.collector_number}` : ''} · ${r.finish} · ${r.rarity}` : l.error}
                          {r?.status === 'custom' && ' · custom'}
                        </small>
                      </span>
                    </button>
                  </td>
                  <td className="num">
                    <input type="number" min={1} value={l.quantity} aria-label="Quantity"
                      onChange={(e) => onQuantity(l.id, Math.max(1, Number(e.target.value) || 1))} />
                  </td>
                  <td className="num">{r?.status === 'custom' ? <span className="muted">n/a</span> : usd(r?.price_today)}</td>
                  <td className="num">{usd(fair?.mid)}<small>{range(fair?.lo, fair?.hi)}</small></td>
                  <td className="num">{usd(r?.pred_price)}<small>{range(r?.price_lo, r?.price_hi)}</small></td>
                  <td><SignalBadge signal={r?.signal ?? null} /></td>
                  <td className="num">{usd(r?.buylist_pred)}<small>{range(r?.buylist_lo, r?.buylist_hi)}</small></td>
                  <td><button type="button" className="icon" aria-label={`Remove ${r?.name ?? 'card'}`} onClick={() => onRemove(l.id)}>✕</button></td>
                </tr>
                {open === l.id && r && (
                  <tr className="detail">
                    <td colSpan={8}>
                      <Detail row={r} backend={backend} />
                    </td>
                  </tr>
                )}
              </Fragment>
            )
          })}
        </tbody>
        <tfoot>
          <tr>
            <th scope="row">Totals</th>
            <td className="num">{lines.reduce((a, l) => a + l.quantity, 0)}</td>
            <td className="num">{usd(sum((r) => r.price_today))}</td>
            <td className="num">{usd(sum((r) => fairOf(r, backend).mid))}</td>
            <td className="num">{usd(sum((r) => r.pred_price))}</td>
            <td />
            <td className="num">{usd(sum((r) => r.buylist_pred))}</td>
            <td />
          </tr>
        </tfoot>
      </table>
    </div>
  )
}

function Detail({ row, backend }: { row: PredictionRow; backend: Backend }) {
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
