import { describe, expect, it } from 'vitest'
import { search } from './data'
import type { SearchIndex } from './types'

const lotus = 'b0faa7f2-b547-42c4-a810-839da50dadfe'
const solRing = '4cbc6901-6a4a-4d0a-83ea-7eefa3b35021'
const index: SearchIndex = {
  printing_key: [`${lotus}:nonfoil`, `${lotus}:foil`, `${solRing}:foil`],
  name: ['Black Lotus', 'Black Lotus', 'Sol Ring'],
  set: ['lea', 'lea', 'c21'],
  collector_number: ['232', '232', '263'],
  finish: ['nonfoil', 'foil', 'foil'],
  treatment: ['', 'borderless', 'showcase'],
  rarity: ['r', 'r', 'u'],
  sets: { lea: 'Limited Edition Alpha', c21: 'Commander 2021' },
}

describe('card search', () => {
  it('accepts a Scryfall ID and card or image links', () => {
    const keys = (query: string) => search(index, { name: query }).map((hit) => hit.printing_key)
    expect(keys(lotus)).toEqual([`${lotus}:nonfoil`, `${lotus}:foil`])
    expect(keys('https://scryfall.com/card/lea/232/black-lotus?utm_source=test'))
      .toEqual([`${lotus}:nonfoil`, `${lotus}:foil`])
    expect(keys(`https://cards.scryfall.io/normal/front/b/0/${lotus}.jpg`))
      .toEqual([`${lotus}:nonfoil`, `${lotus}:foil`])
  })

  it('combines name, set and treatment filters', () => {
    expect(search(index, { name: 'lotus', set: 'alpha', treatment: 'foil' }).map((hit) => hit.printing_key))
      .toEqual([`${lotus}:foil`])
    expect(search(index, { name: lotus, treatment: 'borderless' }).map((hit) => hit.printing_key))
      .toEqual([`${lotus}:foil`])
  })
})
