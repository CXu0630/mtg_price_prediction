import { useEffect, useRef, useState } from 'react'

/**
 * The primary list action (Predict / Show cards / Back to cards): one button that turns over whenever its
 * label changes. Every label it can show is laid out invisibly underneath, so its size never changes.
 */
export function FlipButton({ label, labels, onClick, disabled }: {
  label: string
  labels: string[]
  onClick: () => void
  disabled?: boolean
}) {
  const deg = useRef(0)
  const shown = useRef(label)
  const [angle, setAngle] = useState(0)
  const [faces, setFaces] = useState<[string, string]>([label, label]) // [front, back]

  useEffect(() => {
    if (label === shown.current) return
    shown.current = label
    deg.current += 180
    const frontUp = Math.round(deg.current / 180) % 2 === 0
    setAngle(deg.current)
    setFaces((f) => (frontUp ? [label, f[1]] : [f[0], label]))
  }, [label])

  return (
    <button type="button" className="predict" disabled={disabled} onClick={onClick} aria-label={label}>
      <span className="predict-inner" aria-hidden="true" style={{ transform: `rotateY(${angle}deg)` }}>
        {labels.map((l) => <span key={l} className="predict-face predict-sizer">{l}</span>)}
        <span className="predict-face">{faces[0]}</span>
        <span className="predict-face predict-back">{faces[1]}</span>
      </span>
    </button>
  )
}
