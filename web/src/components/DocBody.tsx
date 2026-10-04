import { useEffect, useState } from 'react'
import { PAGE_BLOCKS, type PageId } from '../content/pages'

export const DOC_STEP_MS = 70
const DOC_TURN_MS = 800 // matches .doc-block's transition

/** Until the last block has turned edge-on on its way out (the flip curve passes 90deg at ~14%). */
export const docLeaveMs = (page: Exclude<PageId, 'predictor'>) =>
  (PAGE_BLOCKS[page].length - 1) * DOC_STEP_MS + 0.14 * DOC_TURN_MS + 120

/** A documentation page's content: each block is a card that turns over into place, one after another. */
export function DocBody({ page, leaving }: { page: Exclude<PageId, 'predictor'>; leaving: boolean }) {
  const blocks = PAGE_BLOCKS[page]
  const [up, setUp] = useState(false)
  useEffect(() => {
    const t = requestAnimationFrame(() => requestAnimationFrame(() => setUp(true)))
    return () => cancelAnimationFrame(t)
  }, [])
  const shown = up && !leaving
  return (
    <div className="doc-body">
      {blocks.map((b, i) => (
        <section key={b.key} className={`doc-block ${shown ? 'face-up' : ''} ${leaving ? 'turned-out' : ''}`}
          style={{ transitionDelay: `${(leaving ? blocks.length - 1 - i : i) * DOC_STEP_MS}ms` }}>
          {b.title && <h3 className="doc-title">{b.title}</h3>}
          {b.body}
          {b.note && <p className="doc-note">{b.note}</p>}
        </section>
      ))}
    </div>
  )
}
