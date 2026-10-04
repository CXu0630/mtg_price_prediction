import { useEffect, useRef, useState, type ReactNode } from 'react'

/**
 * A two-sided panel that turns over whenever `view` changes. Rotation accumulates (0 -> 180 -> 360 ...)
 * so it always turns the same way; the incoming view is rendered on whichever face is about to come
 * round, and the face that went away is emptied once the turn ends so it stops taking up height.
 */
export function FlipPanel<V extends string>({ view, rest, render, boxed }: {
  view: V
  /** the resting view: it is always laid out invisibly so the panel never gets shorter than it */
  rest: V
  render: (v: V) => ReactNode
  /** views drawn inside a faint card-like border */
  boxed: (v: V) => boolean
}) {
  const deg = useRef(0)
  const shown = useRef(view)
  const [angle, setAngle] = useState(0)
  const [faces, setFaces] = useState<[V | null, V | null]>([view, null]) // [front, back]

  useEffect(() => {
    if (view === shown.current) return
    shown.current = view
    deg.current += 180
    const frontUp = Math.round(deg.current / 180) % 2 === 0
    setAngle(deg.current)
    setFaces((f) => (frontUp ? [view, f[1]] : [f[0], view]))
  }, [view])

  const frontUp = Math.round(angle / 180) % 2 === 0
  const face = (v: V | null, side: 'front' | 'back') => {
    const up = (side === 'front') === frontUp
    return (
      <div className={`flip-face flip-${side} ${v && boxed(v) ? 'boxed' : ''}`} inert={!up} aria-hidden={!up}>
        {v && render(v)}
      </div>
    )
  }

  return (
    <div className="flip-panel">
      <div className="flip-inner" style={{ transform: `rotateY(${angle}deg)` }}
        onTransitionEnd={(e) => {
          if (e.target !== e.currentTarget) return
          setFaces((f) => (frontUp ? [f[0], null] : [null, f[1]]))
        }}>
        <div className="flip-sizer" aria-hidden="true" inert>{render(rest)}</div>
        {face(faces[0], 'front')}
        {face(faces[1], 'back')}
      </div>
    </div>
  )
}
