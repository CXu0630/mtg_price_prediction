import { useEffect, useId, useRef, useState, type KeyboardEvent } from 'react'

export interface Option { value: string; label: string }

/**
 * A select drawn by the app instead of the OS, so every dropdown (set, treatment, custom-card fields)
 * looks like the card-search suggestions list. Keyboard: arrows/Home/End move, Enter/Space choose, Esc closes.
 */
export function Dropdown({ value, options, onChange, placeholder = 'Select…', disabled, id }: {
  value: string
  options: Option[]
  onChange: (value: string) => void
  placeholder?: string
  disabled?: boolean
  id?: string
}) {
  const [open, setOpen] = useState(false)
  const [active, setActive] = useState(0)
  const root = useRef<HTMLDivElement>(null)
  const list = useRef<HTMLUListElement>(null)
  const listId = useId()
  const current = options.find((o) => o.value === value)

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => { if (!root.current?.contains(e.target as Node)) setOpen(false) }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  useEffect(() => { // keep the highlighted option in view
    if (open) list.current?.children[active]?.scrollIntoView({ block: 'nearest' })
  }, [open, active])

  const show = () => {
    setActive(Math.max(0, options.findIndex((o) => o.value === value)))
    setOpen(true)
  }
  const pick = (i: number) => {
    const o = options[i]
    if (o) onChange(o.value)
    setOpen(false)
  }
  const onKey = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && open) { e.preventDefault(); e.stopPropagation(); setOpen(false); return }
    if (!open) {
      if (['ArrowDown', 'ArrowUp', 'Enter', ' '].includes(e.key)) { e.preventDefault(); show() }
      return
    }
    const last = options.length - 1
    const moves: Record<string, number> = { ArrowDown: Math.min(active + 1, last), ArrowUp: Math.max(active - 1, 0), Home: 0, End: last }
    if (e.key in moves) { e.preventDefault(); setActive(moves[e.key]) }
    else if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(active) }
    else if (e.key === 'Tab') setOpen(false)
  }

  return (
    <div className={`dropdown ${open ? 'open' : ''}`} ref={root}>
      <button type="button" id={id} className="dropdown-button" disabled={disabled || options.length === 0}
        aria-haspopup="listbox" aria-expanded={open} aria-controls={listId}
        onClick={() => (open ? setOpen(false) : show())} onKeyDown={onKey}>
        <span className={current ? '' : 'placeholder'}>{current?.label ?? placeholder}</span>
      </button>
      {open && (
        <ul className="hits dropdown-list" role="listbox" id={listId} ref={list}>
          {options.map((o, i) => (
            <li key={o.value} role="option" aria-selected={o.value === value}>
              <button type="button" tabIndex={-1} className={`${i === active ? 'active' : ''} ${o.value === value ? 'chosen' : ''}`}
                onMouseEnter={() => setActive(i)} onClick={() => pick(i)}>
                {o.label}
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  )
}
