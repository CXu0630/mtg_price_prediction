import type { CSSProperties, ReactNode } from 'react'

/* Small monochrome charts for the documentation pages. One hue (the text colour) carries the data; labels and
   values wear text tokens; every bar has a hover tooltip and a readable value, so nothing relies on colour. */

export const money = (v: number) => (v >= 1000 ? `$${Math.round(v).toLocaleString()}` : v >= 100 ? `$${Math.round(v)}`
  : `$${v.toFixed(2)}`)
export const count = (v: number) => v.toLocaleString()

/** A headline number with a label: the form for a single value that needs no plot. */
export function Stats({ items }: { items: { label: string; value: string; note?: string }[] }) {
  return (
    <dl className="doc-stats">
      {items.map((s) => (
        <div key={s.label}><dt>{s.label}</dt><dd>{s.value}</dd>{s.note && <span>{s.note}</span>}</div>
      ))}
    </dl>
  )
}

/** Horizontal bars, one series. `signed` draws negatives to the left of a zero line. */
export function Bars({ rows, format, signed = false, max }: {
  rows: { label: string; value: number; note?: string }[]
  format: (v: number) => string
  signed?: boolean
  max?: number
}) {
  const top = max ?? Math.max(...rows.map((r) => Math.abs(r.value)), 1e-9)
  return (
    <div className={`doc-bars ${signed ? 'signed' : ''}`} role="list">
      {rows.map((r) => {
        const w = `${(Math.abs(r.value) / top) * (signed ? 50 : 100)}%`
        return (
          <div className="bar-row" role="listitem" key={r.label} title={`${r.label}: ${format(r.value)}${r.note ? ` · ${r.note}` : ''}`}>
            <span className="bar-label">{r.label}</span>
            <span className="bar-track">
              {signed && <span className="bar-zero" />}
              <span className={`bar ${r.value < 0 ? 'neg' : ''}`} style={signed
                ? { width: w, [r.value < 0 ? 'right' : 'left']: '50%' } : { width: w }} />
            </span>
            <span className="bar-value">{format(r.value)}</span>
          </div>
        )
      })}
    </div>
  )
}

/** Two series per row (e.g. CatBoost vs Deep GP), with a legend; the second series is a lighter tint. */
export function PairedBars({ rows, series, format }: {
  rows: { label: string; a?: number; b?: number }[]
  series: [string, string]
  format: (v: number) => string
}) {
  const top = Math.max(...rows.flatMap((r) => [r.a ?? 0, r.b ?? 0]), 1e-9)
  return (
    <div className="doc-bars paired">
      <div className="bar-legend">
        <span><i className="swatch a" />{series[0]}</span><span><i className="swatch b" />{series[1]}</span>
      </div>
      {rows.map((r) => (
        <div className="bar-row" key={r.label}>
          <span className="bar-label">{r.label}</span>
          <span className="bar-pair">
            {([['a', r.a, series[0]], ['b', r.b, series[1]]] as const).map(([k, v, name]) => (
              <span className="bar-track" key={k} title={`${r.label} · ${name}: ${v == null ? 'not used' : format(v)}`}>
                {v == null ? <em>not used</em> : <span className={`bar ${k}`} style={{ width: `${(v / top) * 100}%` }} />}
              </span>
            ))}
          </span>
          <span className="bar-value">{r.a == null ? '—' : format(r.a)}<br />{r.b == null ? '—' : format(r.b)}</span>
        </div>
      ))}
    </div>
  )
}

/** Price histogram on a log axis, with p10 / median / p90 markers. */
export function Histogram({ bins, markers }: {
  bins: { lo: number; hi: number; n: number }[]
  markers: { label: string; value: number }[]
}) {
  const peak = Math.max(...bins.map((b) => b.n))
  const lo = Math.log10(bins[0].lo)
  const hi = Math.log10(bins[bins.length - 1].hi)
  const x = (v: number) => `${((Math.log10(v) - lo) / (hi - lo)) * 100}%`
  const ticks = [0.01, 0.1, 1, 10, 100, 1000, 10000].filter((t) => Math.log10(t) >= lo && Math.log10(t) <= hi)
  return (
    <figure className="doc-hist">
      <div className="hist-plot">
        {bins.map((b) => (
          <span key={b.lo} className="hist-bar" style={{ height: `${(b.n / peak) * 100}%` }}
            title={`${money(b.lo)}–${money(b.hi)}: ${count(b.n)} printings`} />
        ))}
        {markers.map((m, i) => ( // labels step down so close markers (p10 / median) never collide
          <span key={m.label} className="hist-marker" style={{ left: x(m.value), '--row': i } as CSSProperties}>
            <span>{m.label}<b>{money(m.value)}</b></span>
          </span>
        ))}
      </div>
      <div className="hist-axis">
        {ticks.map((t) => <span key={t} style={{ left: x(t) }}>{t >= 1 ? `$${t.toLocaleString()}` : `$${t}`}</span>)}
      </div>
    </figure>
  )
}

export function Table({ head, rows, align }: { head: string[]; rows: ReactNode[][]; align?: ('l' | 'r')[] }) {
  return (
    <div className="doc-table-wrap">
      <table className="doc-table">
        <thead><tr>{head.map((h, i) => <th key={h} className={align?.[i] === 'r' ? 'r' : ''}>{h}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => (
          <tr key={i}>{r.map((c, j) => <td key={j} className={align?.[j] === 'r' ? 'r' : ''}>{c}</td>)}</tr>
        ))}</tbody>
      </table>
    </div>
  )
}
