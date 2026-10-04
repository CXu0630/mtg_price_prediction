import { describe, expect, it } from 'vitest'
import { importCSV, parseCSV } from './csv'
import type { SearchIndex } from './types'

const index: SearchIndex = {
  printing_key: ['a:nonfoil', 'a:foil', 'b:nonfoil'],
  name: ['Sol Ring', 'Sol Ring', 'Ragavan, Nimble Pilferer'],
  set: ['c21', 'c21', 'mh2'],
  collector_number: ['263', '263', '138'],
  finish: ['nonfoil', 'foil', 'nonfoil'],
  rarity: ['u', 'u', 'm'],
  sets: {},
}

describe('csv import', () => {
  it('parses quoted fields', () => {
    expect(parseCSV('a,"b, c","d ""e"""\n1,2,3\n')).toEqual([['a', 'b, c', 'd "e"'], ['1', '2', '3']])
  })
  it('matches by name, set and finish, and keeps custom cards', () => {
    const csv = 'Name,Set,Finish,Qty,custom\nsol ring,c21,foil,2,\n"Ragavan, Nimble Pilferer",,,,\nNope,,,,\nX,,,1,"{""name"":""X""}"\n'
    const { items, unmatched } = importCSV(csv, index)
    expect(items).toEqual([
      { kind: 'existing', printing_key: 'a:foil', set: 'c21', name: 'Sol Ring', quantity: 2 },
      { kind: 'existing', printing_key: 'b:nonfoil', set: 'mh2', name: 'Ragavan, Nimble Pilferer', quantity: 1 },
      { kind: 'custom', card: { name: 'X' }, quantity: 1 },
    ])
    expect(unmatched).toEqual(['Nope'])
  })
})
